import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from dataset_config import (
    RESULTS_ROOT,
    DatasetPaths,
    build_drugs_table,
    build_targets_table,
    ensure_dataset_dirs,
    get_dataset_paths,
    load_feature_array,
    normalize_interaction_dataframe,
    read_table,
    resolve_artifacts_root,
    write_table,
)


def _configure_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    validation_size: float = 0.1
    random_state: int = 42
    early_stopping_patience: int = 5


@dataclass
class PreparationConfig:
    artifacts_dir: str | None = None
    threshold: float = 0.0
    force: bool = False
    kpgt_dir: str | None = None
    kpgt_model_path: str | None = None
    kpgt_python: str | None = None
    kpgt_preprocess_jobs: int = 4
    kpgt_chunk_size: int = 1024
    kpgt_batch_size: int = 32
    esm_model_name: str = "esm2_t33_650M_UR50D"
    esmfold_chunk_size: int | None = None
    skip_esmfold: bool = False
    target_similarity_processes: int | None = None
    target_similarity_chunksize: int | None = None


def _tqdm(iterable=None, **kwargs):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return iterable
    # SSH and notebook subprocess output are more reliable when tqdm writes to
    # stdout and when bars are not auto-disabled by non-TTY streams.
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("disable", False)
    return tqdm(iterable, **kwargs)


def _log(message: str) -> None:
    print(message, flush=True)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _training_run_parameters(
    train_dataset_name: str,
    test_dataset_name: str | None,
    artifacts_dir: str | None,
    config: TrainingConfig,
) -> dict:
    return {
        "train_dataset_name": train_dataset_name,
        "test_dataset_name": test_dataset_name,
        "artifacts_root": str(resolve_artifacts_root(artifacts_dir)),
        "training_config": asdict(config),
    }


def _training_run_dir(
    train_dataset_name: str,
    test_dataset_name: str | None,
    artifacts_dir: str | None,
    config: TrainingConfig,
) -> tuple[Path, dict]:
    parameters = _training_run_parameters(train_dataset_name, test_dataset_name, artifacts_dir, config)
    encoded = json.dumps(parameters, sort_keys=True, default=_json_default).encode("utf-8")
    run_hash = hashlib.sha256(encoded).hexdigest()[:12]
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in train_dataset_name)
    run_dir = resolve_artifacts_root(artifacts_dir) / "training_runs" / f"{safe_name}_{run_hash}"
    return run_dir, parameters


def _load_torch_checkpoint(torch_module, path: Path, map_location):
    try:
        return torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=map_location)


def _iterate_with_progress(iterable, *, total: int, desc: str, unit: str):
    progress = _tqdm(iterable, total=total, desc=desc, unit=unit)
    step = max(1, total // 20) if total else 1
    if total:
        _log(f"{desc}: 0/{total} {unit} (0.0%)")

    try:
        for index, item in enumerate(progress, start=1):
            yield item
            if total and (index == total or index % step == 0):
                _log(f"{desc}: {index}/{total} {unit} ({(index / total) * 100:.1f}%)")
    finally:
        if progress is not None:
            progress.close()


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _load_targets_for_graph_build(paths: DatasetPaths):
    import pandas as pd

    targets_df = pd.read_csv(paths.targets_table)
    with open(paths.protein_features, "rb") as handle:
        protein_features = pickle.load(handle)
    if len(targets_df) != len(protein_features):
        raise ValueError("targets.csv and prot_rep.pkl have different lengths.")
    return targets_df, protein_features


def _run_subprocess(command: list[str], workdir: Path, description: str, extra_pythonpath: Path | None = None) -> None:
    env = os.environ.copy()
    if extra_pythonpath is not None:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(extra_pythonpath) if not existing else os.pathsep.join([str(extra_pythonpath), existing])
    try:
        subprocess.run(command, cwd=workdir, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to {description}. Command: {' '.join(command)}") from exc


def _resolve_kpgt_script(kpgt_dir: Path, script_name: str) -> Path:
    candidates = [kpgt_dir / script_name, kpgt_dir / "scripts" / script_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing KPGT script {script_name} under {kpgt_dir}")


def _patch_kpgt_compatibility(kpgt_dir: Path) -> None:
    descriptor_path = kpgt_dir / "src" / "data" / "descriptors" / "rdNormalizedDescriptors.py"
    if not descriptor_path.exists():
        return

    content = descriptor_path.read_text(encoding="utf-8")
    alias_line = '    st.gilbrat = st.gibrat\n'
    if alias_line in content:
        return

    needle = "import scipy.stats as st\n"
    replacement = (
        "import scipy.stats as st\n\n"
        "if not hasattr(st, \"gilbrat\") and hasattr(st, \"gibrat\"):\n"
        "    st.gilbrat = st.gibrat\n"
    )
    if needle in content:
        descriptor_path.write_text(content.replace(needle, replacement, 1), encoding="utf-8")


def _generate_kpgt_features(paths: DatasetPaths, config: PreparationConfig) -> None:
    if paths.drug_features.exists() and not config.force:
        _log(f"Skipping KPGT features: already exists at {paths.drug_features}")
        return

    _log(f"Generating KPGT features into {paths.drug_features}")

    if not config.kpgt_dir:
        raise ValueError("Missing --kpgt-dir. True KPGT feature generation requires an external KPGT checkout.")
    if not config.kpgt_model_path:
        raise ValueError("Missing --kpgt-model-path. True KPGT feature generation requires the pretrained model path.")

    kpgt_dir = Path(config.kpgt_dir).expanduser().resolve()
    model_path = Path(config.kpgt_model_path).expanduser().resolve()
    streaming_script = Path(__file__).resolve().parent / "scripts" / "kpgt_streaming_extract.py"
    _require_file(streaming_script, "streaming KPGT extraction script")
    _require_file(model_path, "KPGT pretrained model")
    _patch_kpgt_compatibility(kpgt_dir)
    kpgt_python = Path(config.kpgt_python).expanduser().resolve() if config.kpgt_python else Path(sys.executable)
    _require_file(kpgt_python, "KPGT python executable")

    _run_subprocess(
        [
            str(kpgt_python),
            str(streaming_script),
            "--input-csv",
            str(paths.drugs_table),
            "--output-npz",
            str(paths.drug_features),
            "--excluded-csv",
            str(paths.excluded_table),
            "--config",
            "base",
            "--model-path",
            str(model_path),
            "--chunk-size",
            str(config.kpgt_chunk_size),
            "--batch-size",
            str(config.kpgt_batch_size),
            "--n-jobs",
            str(config.kpgt_preprocess_jobs),
        ],
        kpgt_dir,
        "stream KPGT features",
        extra_pythonpath=kpgt_dir,
    )
    _require_file(paths.drug_features, "KPGT drug feature output")


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _restrict_to_existing_esmfold_targets(paths: DatasetPaths, config: PreparationConfig) -> None:
    if not config.skip_esmfold:
        return

    import numpy as np
    import pandas as pd

    interactions = pd.read_csv(paths.interaction_table)
    old_drugs = pd.read_csv(paths.drugs_table)
    old_targets = pd.read_csv(paths.targets_table)
    before_interactions = len(interactions)
    available_target_ids = {
        pdb_path.stem
        for pdb_path in paths.esmfold_dir.glob("*.pdb")
        if pdb_path.is_file()
    }
    retained_target_mask = old_targets["Target_ID"].astype(str).isin(available_target_ids)
    if not retained_target_mask.any():
        raise ValueError(
            f"--skip-esmfold found no calculated target PDBs from {paths.targets_table} in {paths.esmfold_dir}"
        )

    retained_target_ids = set(old_targets.loc[retained_target_mask, "Target_ID"].astype(str))
    interactions = interactions[interactions["Target_ID"].astype(str).isin(retained_target_ids)].reset_index(drop=True)
    new_targets = build_targets_table(interactions)
    new_drugs = build_drugs_table(interactions)
    if interactions.empty:
        raise ValueError("--skip-esmfold removed every interaction from the dataset")

    if len(new_targets) == len(old_targets) and len(new_drugs) == len(old_drugs):
        _log(f"Skipping ESMFold generation: all {len(new_targets)} canonical targets already have PDB files")
        return

    old_target_positions = {str(target_id): index for index, target_id in enumerate(old_targets["Target_ID"])}
    old_drug_positions = {str(drug_id): index for index, drug_id in enumerate(old_drugs["Drug_ID"])}
    target_indices = [old_target_positions[str(target_id)] for target_id in new_targets["Target_ID"]]
    drug_indices = [old_drug_positions[str(drug_id)] for drug_id in new_drugs["Drug_ID"]]

    if not config.force and paths.protein_features.exists():
        with open(paths.protein_features, "rb") as handle:
            protein_features = pickle.load(handle)
        if len(protein_features) == len(old_targets):
            with open(paths.protein_features, "wb") as handle:
                pickle.dump([protein_features[index] for index in target_indices], handle)
        elif len(protein_features) != len(new_targets):
            raise ValueError("prot_rep.pkl is aligned with neither the original nor filtered targets.csv")

    if paths.drug_features.exists():
        drug_features = load_feature_array(paths.drug_features)
        if len(drug_features) == len(old_drugs):
            np.savez(paths.drug_features, fps=drug_features[drug_indices])
        elif len(drug_features) != len(new_drugs):
            raise ValueError("kpgt_base.npz is aligned with neither the original nor filtered drugs.csv")

    for matrix_path, old_size, new_size, indices in [
        (paths.target_similarity, len(old_targets), len(new_targets), target_indices),
        (paths.drug_similarity, len(old_drugs), len(new_drugs), drug_indices),
    ]:
        if config.force or not matrix_path.exists():
            continue
        matrix = load_feature_array(matrix_path)
        if matrix.shape == (old_size, old_size):
            np.savez(matrix_path, matrix[np.ix_(indices, indices)])
        elif matrix.shape != (new_size, new_size):
            raise ValueError(f"{matrix_path.name} is aligned with neither the original nor filtered canonical table")

    write_table(interactions, paths.interaction_table)
    write_table(new_drugs, paths.drugs_table)
    write_table(new_targets, paths.targets_table)
    _log(
        "Skipping ESMFold generation and restricting the dataset to existing PDBs: "
        f"targets={len(old_targets)}->{len(new_targets)}, "
        f"drugs={len(old_drugs)}->{len(new_drugs)}, "
        f"interactions={before_interactions}->{len(interactions)}"
    )


def _apply_kpgt_exclusions(paths: DatasetPaths) -> None:
    if not paths.excluded_table.exists():
        return

    import pandas as pd

    excluded = pd.read_csv(paths.excluded_table)
    if excluded.empty:
        _log(f"No KPGT exclusions recorded at {paths.excluded_table}")
        return
    if "Drug_ID" not in excluded.columns:
        raise ValueError(f"KPGT exclusion table is missing Drug_ID: {paths.excluded_table}")

    excluded_drug_ids = set(excluded["Drug_ID"].dropna().astype(str))
    if not excluded_drug_ids:
        _log(f"No KPGT exclusions with Drug_ID recorded at {paths.excluded_table}")
        return

    interactions = pd.read_csv(paths.interaction_table)
    drugs = pd.read_csv(paths.drugs_table)
    old_targets = pd.read_csv(paths.targets_table)
    before_interactions = len(interactions)
    before_drugs = len(drugs)
    matched_drugs = drugs["Drug_ID"].astype(str).isin(excluded_drug_ids)
    matched_interactions = interactions["Drug_ID"].astype(str).isin(excluded_drug_ids)
    if not matched_drugs.any() and not matched_interactions.any():
        _log(f"KPGT exclusions already applied from {paths.excluded_table}")
        return

    interactions = interactions[~matched_interactions].reset_index(drop=True)
    drugs = drugs[~matched_drugs].reset_index(drop=True)
    new_targets = build_targets_table(interactions)
    if interactions.empty:
        raise ValueError(f"KPGT exclusions removed every interaction in {paths.interaction_table}")
    if drugs.empty:
        raise ValueError(f"KPGT exclusions removed every drug in {paths.drugs_table}")

    write_table(interactions, paths.interaction_table)
    write_table(drugs, paths.drugs_table)
    write_table(new_targets, paths.targets_table)

    target_table_changed = not old_targets.reset_index(drop=True).equals(new_targets.reset_index(drop=True))
    _remove_if_exists(paths.drug_similarity)
    if target_table_changed:
        for stale_path in [paths.protein_features, paths.target_similarity]:
            _remove_if_exists(stale_path)
    else:
        _log("KPGT exclusions changed drugs/interactions but targets are unchanged; preserving target-derived artifacts")

    _log(
        "Applied KPGT exclusions: "
        f"excluded_drugs={len(excluded_drug_ids)}, "
        f"drugs={before_drugs}->{len(drugs)}, "
        f"interactions={before_interactions}->{len(interactions)}, "
        f"details={paths.excluded_table}"
    )


def _generate_protein_features(paths: DatasetPaths, config: PreparationConfig) -> None:
    if paths.protein_features.exists() and not config.force:
        _log(f"Skipping ESM-2 embeddings: already exists at {paths.protein_features}")
        return

    import pandas as pd
    import torch

    targets_df = pd.read_csv(paths.targets_table)
    _log(f"Generating ESM-2 embeddings for {len(targets_df)} targets")
    try:
        esm = __import__("esm")
    except ModuleNotFoundError as exc:
        raise RuntimeError("ESM is required to generate true ESM-2 protein features.") from exc

    model_loader = getattr(getattr(esm, "pretrained", None), config.esm_model_name, None)
    if model_loader is None:
        raise ValueError(f"Unsupported ESM model: {config.esm_model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model, alphabet = model_loader()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device)
    esm_model.eval()

    token_representations = []
    repr_layer = 33
    for row in _iterate_with_progress(
        targets_df.itertuples(index=False),
        total=len(targets_df),
        desc="ESM-2 embeddings",
        unit="target",
    ):
        batch_labels, batch_strs, batch_tokens = batch_converter([(row.Target_ID, row.Target)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        token_representations.append(results["representations"][repr_layer].cpu().detach().numpy())

    paths.protein_features.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.protein_features, "wb") as handle:
        pickle.dump(token_representations, handle)


def _generate_esmfold_structures(paths: DatasetPaths, config: PreparationConfig) -> None:
    if config.skip_esmfold:
        _log("Skipping ESMFold structure generation because --skip-esmfold was passed")
        return

    import gc

    import pandas as pd
    import torch

    targets_df = pd.read_csv(paths.targets_table)
    missing_targets = []
    for row in targets_df.itertuples(index=False):
        pdb_path = paths.esmfold_dir / f"{row.Target_ID}.pdb"
        if config.force or not pdb_path.exists():
            missing_targets.append(row)
    missing_targets.sort(key=lambda row: (len(str(row.Target)), str(row.Target_ID)))

    if not missing_targets:
        _log(f"Skipping ESMFold structures: all {len(targets_df)} PDB files already exist in {paths.esmfold_dir}")
        return

    shortest_target = len(str(missing_targets[0].Target))
    longest_target = len(str(missing_targets[-1].Target))
    _log(
        "Generating ESMFold structures for "
        f"{len(missing_targets)} of {len(targets_df)} targets, sorted by length "
        f"({shortest_target} to {longest_target} residues)"
    )

    try:
        esm = __import__("esm")
    except ModuleNotFoundError as exc:
        raise RuntimeError("ESM is required to generate true ESMFold structures.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    current_chunk_size = config.esmfold_chunk_size
    if config.esmfold_chunk_size is not None:
        model.set_chunk_size(config.esmfold_chunk_size)

    paths.esmfold_dir.mkdir(parents=True, exist_ok=True)
    for row in _iterate_with_progress(
        missing_targets,
        total=len(missing_targets),
        desc="ESMFold PDBs",
        unit="target",
    ):
        pdb_path = paths.esmfold_dir / f"{row.Target_ID}.pdb"
        target_length = len(str(row.Target))
        output = None
        while output is None:
            chunk_size_label = current_chunk_size if current_chunk_size is not None else "default"
            _log(f"ESMFold target {row.Target_ID}: generating PDB for {target_length} residues with chunk_size={chunk_size_label}")
            try:
                with torch.no_grad():
                    output = model.infer_pdb(row.Target)
            except torch.OutOfMemoryError as exc:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                if current_chunk_size == 1:
                    raise RuntimeError(
                        "ESMFold ran out of CUDA memory while generating "
                        f"Target_ID={row.Target_ID} ({target_length} residues, chunk_size=1). "
                        "Rerun prepare without --force so completed PDBs are skipped; this target likely needs "
                        "CPU ESMFold, a larger-memory GPU, or removal from the dataset."
                    ) from exc

                next_chunk_size = 128 if current_chunk_size is None else max(1, current_chunk_size // 2)
                _log(
                    "ESMFold CUDA OOM for "
                    f"Target_ID={row.Target_ID} ({target_length} residues) at chunk_size={chunk_size_label}; "
                    f"retrying with chunk_size={next_chunk_size}"
                )
                current_chunk_size = next_chunk_size
                model.set_chunk_size(current_chunk_size)

        with open(pdb_path, "w", encoding="utf-8") as handle:
            handle.write(output)
        output = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _generate_graphs(paths: DatasetPaths, config: PreparationConfig) -> None:
    import numpy as np
    import torch

    from utils import load_predicted_PDB3, protein_graph

    targets_df, protein_features = _load_targets_for_graph_build(paths)
    paths.graph_dir.mkdir(parents=True, exist_ok=True)
    existing_graphs = 0
    if not config.force:
        existing_graphs = sum(1 for target_id in targets_df["Target_ID"] if (paths.graph_dir / f"{target_id}.pt").exists())
        if existing_graphs == len(targets_df):
            _log(f"Skipping protein graphs: all {len(targets_df)} graph files already exist in {paths.graph_dir}")
            return

    pending_graphs = len(targets_df) if config.force else len(targets_df) - existing_graphs
    _log(f"Generating protein graphs for {pending_graphs} of {len(targets_df)} targets")

    graph_inputs = zip(targets_df.iterrows(), protein_features)
    for (_, row), embedding in _iterate_with_progress(
        graph_inputs,
        total=len(targets_df),
        desc="Protein graphs",
        unit="target",
    ):
        graph_path = paths.graph_dir / f"{row['Target_ID']}.pt"
        if graph_path.exists() and not config.force:
            continue

        pdb_path = paths.esmfold_dir / f"{row['Target_ID']}.pdb"
        _require_file(pdb_path, f"ESMFold structure for {row['Target_ID']}")
        dis_map, seq = load_predicted_PDB3(str(pdb_path))
        emb = embedding[0][1 : len(embedding[0]) - 1]
        if len(seq) != len(emb):
            raise ValueError(
                f"Sequence length mismatch for {row['Target_ID']}: ESMFold residues={len(seq)}, ESM-2 residues={len(emb)}"
            )
        row_idx, col_idx = np.where(dis_map <= 8)
        graph = protein_graph(seq, [row_idx, col_idx], emb)
        torch.save(graph, graph_path)


def _generate_similarity_matrices(paths: DatasetPaths, config: PreparationConfig) -> None:
    import pandas as pd

    from sim_matrix import compute_and_save_tm_score_matrix_parallel_optimized, compute_and_save_tanimoto_matrix

    drugs_df = pd.read_csv(paths.drugs_table)
    targets_df = pd.read_csv(paths.targets_table)

    if config.force or not paths.drug_similarity.exists():
        _log(f"Generating drug similarity matrix at {paths.drug_similarity}")
        compute_and_save_tanimoto_matrix(drugs_df, paths.drug_similarity)
    else:
        _log(f"Skipping drug similarity matrix: already exists at {paths.drug_similarity}")

    if config.force or not paths.target_similarity.exists():
        missing_pdbs = [target_id for target_id in targets_df["Target_ID"] if not (paths.esmfold_dir / f"{target_id}.pdb").exists()]
        if missing_pdbs:
            missing_preview = ", ".join(map(str, missing_pdbs[:5]))
            raise FileNotFoundError(f"Missing ESMFold PDB files for TM-score matrix generation: {missing_preview}")
        _log(f"Generating target similarity matrix at {paths.target_similarity}")
        compute_and_save_tm_score_matrix_parallel_optimized(
            targets_df,
            paths.esmfold_dir,
            paths.target_similarity,
            num_processes=config.target_similarity_processes,
            chunksize=config.target_similarity_chunksize,
        )
    else:
        _log(f"Skipping target similarity matrix: already exists at {paths.target_similarity}")


def prepare_dataset(
    dataset_name: str,
    input_path: str | None = None,
    smiles_column: str = "smiles",
    sequence_column: str = "sequence",
    activity_column: str = "activity",
    config: PreparationConfig | None = None,
) -> DatasetPaths:
    import pandas as pd

    config = config or PreparationConfig()
    paths = get_dataset_paths(dataset_name, artifacts_dir=config.artifacts_dir)
    ensure_dataset_dirs(paths)

    if input_path is not None:
        source = Path(input_path).expanduser().resolve()
        interactions = normalize_interaction_dataframe(
            read_table(source),
            smiles_column=smiles_column,
            sequence_column=sequence_column,
            activity_column=activity_column,
            threshold=config.threshold,
            dataset_prefix=dataset_name,
        )
        write_table(interactions, paths.interaction_table)
        write_table(build_drugs_table(interactions), paths.drugs_table)
        write_table(build_targets_table(interactions), paths.targets_table)

    _require_file(paths.interaction_table, "canonical interaction table")
    _require_file(paths.drugs_table, "drug table")
    _require_file(paths.targets_table, "target table")

    pd.read_csv(paths.drugs_table)
    pd.read_csv(paths.targets_table)
    stages = [
        ("KPGT features", _generate_kpgt_features),
        ("ESM-2 embeddings", _generate_protein_features),
        ("ESMFold structures", _generate_esmfold_structures),
        ("Protein graphs", _generate_graphs),
        ("Similarity matrices", _generate_similarity_matrices),
    ]
    progress = _tqdm(total=len(stages), desc=f"Preparing {dataset_name}", unit="stage")
    try:
        _log(f"Preparing dataset {dataset_name} under {paths.root}")
        _log(f"Preparing {dataset_name}: 0/{len(stages)} stages (0.0%)")
        if progress is not None:
            progress.set_postfix(stage="starting")
        for stage_index, (stage_name, stage_fn) in enumerate(stages, start=1):
            _log(f"Starting stage: {stage_name}")
            if progress is not None:
                progress.set_postfix(stage=stage_name)
            stage_fn(paths, config)
            if stage_name == "KPGT features":
                _apply_kpgt_exclusions(paths)
                _restrict_to_existing_esmfold_targets(paths, config)
            if progress is not None:
                progress.update(1)
            _log(f"Preparing {dataset_name}: {stage_index}/{len(stages)} stages ({(stage_index / len(stages)) * 100:.1f}%)")
    finally:
        if progress is not None:
            progress.close()
    return paths


def _load_training_assets(paths: DatasetPaths):
    import pandas as pd

    _log(f"Loading drug features: {paths.drug_features}")
    drug_features = load_feature_array(paths.drug_features)
    _log(f"Loading drugs table: {paths.drugs_table}")
    drug_df = pd.read_csv(paths.drugs_table)
    _log(f"Loading targets table: {paths.targets_table}")
    target_df = pd.read_csv(paths.targets_table)
    drug_dict = dict(zip(drug_df["Drug_ID"], drug_features))
    _log(f"Loading drug similarity matrix: {paths.drug_similarity}")
    tanimoto_matrix = load_feature_array(paths.drug_similarity)
    _log(f"Loading target similarity matrix: {paths.target_similarity}")
    tm_score_matrix = load_feature_array(paths.target_similarity)
    _log(
        "Loaded training assets: "
        f"{len(drug_df)} drugs, {len(target_df)} targets, "
        f"drug_features_shape={getattr(drug_features, 'shape', 'unknown')}, "
        f"drug_similarity_shape={getattr(tanimoto_matrix, 'shape', 'unknown')}, "
        f"target_similarity_shape={getattr(tm_score_matrix, 'shape', 'unknown')}"
    )
    return drug_features, drug_df, target_df, drug_dict, tanimoto_matrix, tm_score_matrix


def _merge_predictions(raw_df, prediction_df):
    import pandas as pd

    prediction_lookup = prediction_df[["Drug_ID", "Target_ID", "predicted_label"]].drop_duplicates()
    merged = raw_df.merge(prediction_lookup, on=["Drug_ID", "Target_ID"], how="left")
    return merged.rename(columns={"Label": "label_true", "predicted_label": "label_predicted", "Drug": "smiles", "Target": "sequence"})


def run_training(
    train_dataset_name: str,
    test_dataset_name: str | None = None,
    training_config: TrainingConfig | None = None,
    artifacts_dir: str | None = None,
    output_dir: str | None = None,
    output_name: str | None = None,
) -> Path:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    from loss import FocalLoss, NTXentContrastiveLoss
    from model import GraphDTI_bi
    from utils import GraphDataset_withsim, custom_collate_fn, custom_collate_fn_test, evaluate_cl, train_cl

    config = training_config or TrainingConfig()
    train_paths = get_dataset_paths(train_dataset_name, artifacts_dir=artifacts_dir)
    test_paths = get_dataset_paths(test_dataset_name, artifacts_dir=artifacts_dir) if test_dataset_name else None

    _log(f"Starting training for dataset '{train_dataset_name}'")
    _log(f"Artifacts root: {resolve_artifacts_root(artifacts_dir)}")
    _log(f"Training interaction table: {train_paths.interaction_table}")
    if config.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if config.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be 0 or greater")

    run_dir, run_parameters = _training_run_dir(train_dataset_name, test_dataset_name, artifacts_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    params_path = run_dir / "params.json"
    last_checkpoint_path = run_dir / "last.pt"
    best_checkpoint_path = run_dir / "best.pt"
    if params_path.exists():
        saved_parameters = json.loads(params_path.read_text(encoding="utf-8"))
        if saved_parameters != run_parameters:
            raise RuntimeError(f"Training run parameter mismatch for {run_dir}")
    else:
        params_path.write_text(json.dumps(run_parameters, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Training run directory: {run_dir}")

    train_drug_feature, train_drug_df, train_target_df, train_drug_dict, tanimoto_matrix, tm_score_matrix = _load_training_assets(train_paths)
    _log(f"Loading training interactions: {train_paths.interaction_table}")
    train_df = pd.read_csv(train_paths.interaction_table)
    _log(f"Loaded {len(train_df)} interactions")
    stratify_labels = train_df["Label"] if train_df["Label"].nunique() > 1 else None
    _log(
        "Splitting train/validation: "
        f"validation_size={config.validation_size}, random_state={config.random_state}, "
        f"stratified={stratify_labels is not None}"
    )
    train_df, validation_df = train_test_split(
        train_df,
        test_size=config.validation_size,
        stratify=stratify_labels,
        random_state=config.random_state,
    )
    _log(f"Split sizes: train={len(train_df)}, validation={len(validation_df)}")

    if test_paths is None:
        test_df = validation_df.copy()
        test_drug_df = train_drug_df
        test_target_df = train_target_df
        test_drug_dict = train_drug_dict
        test_graph_dir = train_paths.graph_dir
        _log("No external test dataset provided; validation split will also be used for final test predictions.")
    else:
        _log(f"Loading external test dataset '{test_dataset_name}'")
        _, test_drug_df, test_target_df, test_drug_dict, _, _ = _load_training_assets(test_paths)
        _log(f"Loading test interactions: {test_paths.interaction_table}")
        test_df = pd.read_csv(test_paths.interaction_table)
        test_graph_dir = test_paths.graph_dir
        _log(f"Loaded {len(test_df)} test interactions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Training device: {device}")
    if device.type == "cuda":
        _log(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    _log("Constructing graph datasets")
    train_dataset = GraphDataset_withsim(train_df, train_drug_df, train_target_df, train_drug_dict, str(train_paths.graph_dir))
    valid_dataset = GraphDataset_withsim(validation_df, train_drug_df, train_target_df, train_drug_dict, str(train_paths.graph_dir))
    test_dataset = GraphDataset_withsim(test_df, test_drug_df, test_target_df, test_drug_dict, str(test_graph_dir))
    _log(
        "Dataset sizes: "
        f"train={len(train_dataset)}, validation={len(valid_dataset)}, test={len(test_dataset)}"
    )

    _log("Building GraphDTI_bi model")
    model = GraphDTI_bi(train_drug_feature[0].shape[0], 1280, 2, surface_feature=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    weights = torch.tensor([1.0, 1.0], device=device)
    focal_criterion = FocalLoss(alpha=1, gamma=2, weight=weights)
    drug_contrastive_criterion = NTXentContrastiveLoss(temperature=0.07, sim_threshold=0.8)
    target_contrastive_criterion = NTXentContrastiveLoss(temperature=0.07, sim_threshold=0.5)

    _log(
        "Training config: "
        f"epochs={config.epochs}, batch_size={config.batch_size}, "
        f"learning_rate={config.learning_rate}, weight_decay={config.weight_decay}, "
        f"early_stopping_patience={config.early_stopping_patience}"
    )
    _log("Constructing dataloaders")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, tanimoto_matrix, tm_score_matrix),
    )
    val_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_test,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_test,
    )
    _log(
        "Dataloader batches: "
        f"train={len(train_loader)}, validation={len(val_loader)}, test={len(test_loader)}"
    )

    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 0

    if last_checkpoint_path.exists():
        checkpoint = _load_torch_checkpoint(torch, last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_f1 = float(checkpoint.get("best_f1", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        _log(
            f"Resuming matching run from epoch {start_epoch}; "
            f"best_epoch={best_epoch}, best_f1={best_f1:.4f}, bad_epochs={bad_epochs}"
        )

    early_stop_already_reached = (
        bool(config.early_stopping_patience)
        and bad_epochs >= config.early_stopping_patience
        and best_checkpoint_path.exists()
    )
    if start_epoch >= config.epochs:
        _log(f"Run already reached requested epochs ({config.epochs}); skipping training loop.")
    elif early_stop_already_reached:
        _log(
            f"Run already early-stopped after {start_epoch} epochs; "
            f"best epoch was {best_epoch}."
        )
    else:
        _log(f"Starting epoch loop at epoch {start_epoch + 1}")
    epoch_range = range(start_epoch, start_epoch) if early_stop_already_reached else range(start_epoch, config.epochs)
    remaining_epochs = len(epoch_range)
    epoch_progress = _tqdm(epoch_range, total=remaining_epochs, desc="Training epochs", unit="epoch")
    for epoch in epoch_range:
        _log(f"Epoch {epoch + 1}/{config.epochs}: training")
        train_loss, train_accuracy, train_acc_0, train_acc_1 = train_cl(
            model,
            train_loader,
            optimizer,
            drug_contrastive_criterion,
            target_contrastive_criterion,
            focal_criterion,
            device,
            epoch,
        )
        _log(f"Epoch {epoch + 1}/{config.epochs}: validation")
        val_loss, val_accuracy, val_acc_0, val_acc_1, val_f1, val_preds, val_labels, val_probs = evaluate_cl(
            model,
            val_loader,
            focal_criterion,
            device,
            desc=f"Epoch {epoch + 1} validation",
        )
        _log(
            f"Epoch {epoch + 1}/{config.epochs} summary: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.2f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.2f}%, val_f1={val_f1:.4f}"
        )
        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "training_parameters": run_parameters,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "metrics": {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_acc_0": train_acc_0,
                "train_acc_1": train_acc_1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_acc_0": val_acc_0,
                "val_acc_1": val_acc_1,
                "val_f1": val_f1,
            },
        }
        if device.type == "cuda":
            checkpoint_payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            bad_epochs = 0
            checkpoint_payload["best_f1"] = best_f1
            checkpoint_payload["best_epoch"] = best_epoch
            checkpoint_payload["bad_epochs"] = bad_epochs
            torch.save(checkpoint_payload, best_checkpoint_path)
            _log(f"New best validation F1: {best_f1:.4f} at epoch {epoch + 1}")
        else:
            bad_epochs += 1
            checkpoint_payload["bad_epochs"] = bad_epochs
            if config.early_stopping_patience:
                _log(
                    f"Validation F1 did not improve; plateau count "
                    f"{bad_epochs}/{config.early_stopping_patience}"
                )
            else:
                _log("Validation F1 did not improve; early stopping is disabled")
        checkpoint_payload["best_f1"] = best_f1
        checkpoint_payload["best_epoch"] = best_epoch
        torch.save(checkpoint_payload, last_checkpoint_path)
        _log(f"Saved checkpoints: last={last_checkpoint_path}, best={best_checkpoint_path}")
        set_postfix = getattr(epoch_progress, "set_postfix", None)
        if callable(set_postfix):
            set_postfix(best_f1=f"{best_f1:.4f}", val_f1=f"{val_f1:.4f}")
        update = getattr(epoch_progress, "update", None)
        if callable(update):
            update(1)
        if config.early_stopping_patience and bad_epochs >= config.early_stopping_patience:
            _log(
                f"Early stopping at epoch {epoch + 1}: validation F1 plateaued "
                f"for {bad_epochs} epochs; best epoch was {best_epoch}."
            )
            break
    close = getattr(epoch_progress, "close", None)
    if callable(close):
        close()

    if not best_checkpoint_path.exists():
        raise RuntimeError("Training did not produce a best checkpoint.")

    _log(f"Loading best checkpoint for evaluation: {best_checkpoint_path}")
    best_checkpoint = _load_torch_checkpoint(torch, best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_f1 = float(best_checkpoint.get("best_f1", best_f1))
    best_epoch = int(best_checkpoint.get("best_epoch", best_epoch))
    _log(f"Best validation checkpoint: epoch={best_epoch}, f1={best_f1:.4f}")

    _log("Recomputing validation predictions from best checkpoint")
    _, _, _, _, best_f1, best_predictions, best_labels, best_probs = evaluate_cl(
        model,
        val_loader,
        focal_criterion,
        device,
        desc="Best checkpoint validation",
    )

    _log("Running final test evaluation")
    _, test_acc, _, _, test_f1, test_preds, test_labels, test_probs = evaluate_cl(
        model,
        test_loader,
        focal_criterion,
        device,
        desc="Final test",
    )
    _log(f"Final test summary: accuracy={test_acc:.2f}%, f1={test_f1:.4f}")

    _log("Merging validation and test predictions")
    validation_predictions = validation_df[["Drug_ID", "Target_ID", "Drug", "Target", "Y"]].copy()
    validation_predictions["predicted_label"] = best_predictions
    test_predictions = test_df[["Drug_ID", "Target_ID", "Drug", "Target", "Y"]].copy()
    test_predictions["predicted_label"] = test_preds

    output_root = Path(output_dir).expanduser().resolve() if output_dir else RESULTS_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{output_name or train_dataset_name}_predictions.csv"
    _log(f"Writing predictions to {output_path}")

    output_df = pd.concat(
        [
            _merge_predictions(validation_df, validation_predictions),
            _merge_predictions(test_df, test_predictions),
        ],
        ignore_index=True,
    )[["sequence", "smiles", "label_true", "label_predicted"]]
    output_df.to_csv(output_path, index=False)

    _log(f"Saved predictions to {output_path}")
    _log(f"Validation F1: {best_f1:.4f}")
    _log(f"Test F1: {test_f1:.4f}, Test Accuracy: {test_acc:.2f}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    formatter = lambda prog: argparse.HelpFormatter(prog, width=180)
    parser = argparse.ArgumentParser(
        description="Unified GS-DTI data preparation and training entrypoint.",
        formatter_class=formatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_prepare_args(cmd):
        cmd.add_argument("--dataset", required=True, help="Dataset name under the artifacts directory")
        cmd.add_argument("--input", help="Custom raw dataset in CSV or Parquet")
        cmd.add_argument("--smiles-column", default="smiles")
        cmd.add_argument("--sequence-column", default="sequence")
        cmd.add_argument("--activity-column", default="activity")
        cmd.add_argument("--threshold", type=float, default=0.0, help="Threshold to derive binary Label from numeric activity")
        cmd.add_argument("--artifacts-dir", help="Root directory for canonical datasets and derived artifacts")
        cmd.add_argument("--kpgt-dir", help="Path to an external KPGT checkout")
        cmd.add_argument("--kpgt-model-path", help="Path to the pretrained KPGT model file")
        cmd.add_argument("--kpgt-python", help="Python executable to use for KPGT preprocessing and feature extraction")
        cmd.add_argument("--kpgt-preprocess-jobs", type=int, default=4, help="Worker count used inside each streaming KPGT chunk")
        cmd.add_argument("--kpgt-chunk-size", type=int, default=1024, help="Number of drugs held in memory during streaming KPGT feature extraction")
        cmd.add_argument("--kpgt-batch-size", type=int, default=32, help="KPGT model inference batch size")
        cmd.add_argument("--esm-model-name", default="esm2_t33_650M_UR50D")
        cmd.add_argument("--esmfold-chunk-size", type=int)
        cmd.add_argument(
            "--skip-esmfold",
            action="store_true",
            help="Skip ESMFold generation and keep only targets with existing targets/esmfold/<Target_ID>.pdb files",
        )
        cmd.add_argument("--target-similarity-processes", type=int, help="Worker process count for TM-score target similarity generation")
        cmd.add_argument("--target-similarity-chunksize", type=int, help="Multiprocessing chunksize for TM-score target similarity generation")
        cmd.add_argument("--force", action="store_true", help="Regenerate derived artifacts")

    prepare = subparsers.add_parser("prepare", help="Prepare a built-in or custom dataset", formatter_class=formatter)
    add_shared_prepare_args(prepare)

    train = subparsers.add_parser("train", help="Train using a prepared dataset", formatter_class=formatter)
    train.add_argument("--dataset", required=True, help="Training dataset name")
    train.add_argument("--test-dataset", help="Optional external evaluation dataset name")
    train.add_argument("--artifacts-dir", help="Root directory for canonical datasets and derived artifacts")
    train.add_argument("--output-dir", help="Directory where prediction CSVs are saved")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--validation-size", type=float, default=0.1)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--early-stopping-patience", type=int, default=5, help="Stop after this many validation-F1 plateau epochs; 0 disables early stopping")
    train.add_argument("--output-name", help="Prediction filename prefix")

    run = subparsers.add_parser("run", help="Prepare then train in one command", formatter_class=formatter)
    add_shared_prepare_args(run)
    run.add_argument("--test-dataset", help="Optional external evaluation dataset name")
    run.add_argument("--test-input", help="Optional raw test dataset in CSV or Parquet")
    run.add_argument("--output-dir", help="Directory where prediction CSVs are saved")
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--learning-rate", type=float, default=5e-5)
    run.add_argument("--weight-decay", type=float, default=1e-4)
    run.add_argument("--validation-size", type=float, default=0.1)
    run.add_argument("--random-state", type=int, default=42)
    run.add_argument("--early-stopping-patience", type=int, default=5, help="Stop after this many validation-F1 plateau epochs; 0 disables early stopping")
    run.add_argument("--output-name", help="Prediction filename prefix")
    return parser


def _preparation_config_from_args(args) -> PreparationConfig:
    return PreparationConfig(
        artifacts_dir=args.artifacts_dir,
        threshold=args.threshold,
        force=args.force,
        kpgt_dir=args.kpgt_dir,
        kpgt_model_path=args.kpgt_model_path,
        kpgt_python=args.kpgt_python,
        kpgt_preprocess_jobs=args.kpgt_preprocess_jobs,
        kpgt_chunk_size=args.kpgt_chunk_size,
        kpgt_batch_size=args.kpgt_batch_size,
        esm_model_name=args.esm_model_name,
        esmfold_chunk_size=args.esmfold_chunk_size,
        skip_esmfold=args.skip_esmfold,
        target_similarity_processes=args.target_similarity_processes,
        target_similarity_chunksize=args.target_similarity_chunksize,
    )


def main() -> None:
    _configure_streams()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        prepare_dataset(
            dataset_name=args.dataset,
            input_path=args.input,
            smiles_column=args.smiles_column,
            sequence_column=args.sequence_column,
            activity_column=args.activity_column,
            config=_preparation_config_from_args(args),
        )
        return

    if args.command == "train":
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_size=args.validation_size,
            random_state=args.random_state,
            early_stopping_patience=args.early_stopping_patience,
        )
        run_training(
            args.dataset,
            test_dataset_name=args.test_dataset,
            training_config=config,
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
            output_name=args.output_name,
        )
        return

    if args.command == "run":
        prep_config = _preparation_config_from_args(args)
        prepare_dataset(
            dataset_name=args.dataset,
            input_path=args.input,
            smiles_column=args.smiles_column,
            sequence_column=args.sequence_column,
            activity_column=args.activity_column,
            config=prep_config,
        )
        test_dataset_name = args.test_dataset
        if args.test_input:
            if not test_dataset_name:
                test_dataset_name = f"{args.dataset}_test"
            prepare_dataset(
                dataset_name=test_dataset_name,
                input_path=args.test_input,
                smiles_column=args.smiles_column,
                sequence_column=args.sequence_column,
                activity_column=args.activity_column,
                config=prep_config,
            )
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_size=args.validation_size,
            random_state=args.random_state,
            early_stopping_patience=args.early_stopping_patience,
        )
        run_training(
            args.dataset,
            test_dataset_name=test_dataset_name,
            training_config=config,
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
            output_name=args.output_name,
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
