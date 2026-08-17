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
    save_feature_array,
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
    esm_checkpoint_seconds: float = 600.0
    esmfold_chunk_size: int | None = None
    skip_esmfold: bool = False
    skip_similarity: bool = False
    target_similarity_processes: int | None = None
    target_similarity_chunksize: int | None = None
    # Append (the default) extends an already-prepared dataset in place:
    # substances and targets that are already canonical keep their ids and
    # their derived artifacts, and only genuinely new ones are processed.
    # rebuild=True restores the old behaviour of rewriting the canonical
    # tables from --input.
    rebuild: bool = False
    # Preparation failures for a *new* entity (KPGT graph construction,
    # ESMFold OOM, protein graph build) normally abort the run. batch-predict
    # turns this on so one bad SMILES or sequence in a screening batch is
    # recorded in excluded.csv and dropped instead of killing the job.
    drop_failed_entities: bool = False


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


# ---------------------------------------------------------------------------
# Append-mode bookkeeping
#
# Every derived artifact is positionally aligned with a canonical table:
# kpgt_base.npz row i <-> drugs.csv row i, prot_rep.pkl entry i <-> targets.csv
# row i, target_simmatrix.npz index i <-> targets.csv row i, and so on. Append
# mode preserves that by only ever adding rows to the *end* of a table and only
# ever extending an artifact's tail. artifact_manifest.json records how many
# leading rows each artifact covers plus a hash of exactly which ids those were,
# so a table that changed underneath an artifact fails loud instead of silently
# scoring the wrong drug against the wrong protein.
# ---------------------------------------------------------------------------

DRUG_ARTIFACT_KEYS = ("drug_features", "drug_similarity")
TARGET_ARTIFACT_KEYS = ("protein_features", "target_similarity")


def _manifest_path(paths: DatasetPaths) -> Path:
    return paths.root / "artifact_manifest.json"


def _load_manifest(paths: DatasetPaths) -> dict:
    path = _manifest_path(paths)
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _log(f"Ignoring unreadable artifact manifest at {path}")
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _save_manifest(paths: DatasetPaths, manifest: dict) -> None:
    path = _manifest_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _id_hash(identifiers) -> str:
    digest = hashlib.sha256()
    for identifier in identifiers:
        encoded = str(identifier).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _record_artifact_prefix(paths: DatasetPaths, key: str, identifiers, rows: int) -> None:
    manifest = _load_manifest(paths)
    manifest[key] = {"rows": int(rows), "id_hash": _id_hash(list(identifiers)[:rows])}
    _save_manifest(paths, manifest)


def _synced_prefix(paths: DatasetPaths, key: str, artifact_rows: int, identifiers) -> int:
    """How many leading table rows the artifact is trusted to already cover."""
    if artifact_rows <= 0:
        return 0
    identifiers = list(identifiers)
    if artifact_rows > len(identifiers):
        raise ValueError(
            f"{key} has {artifact_rows} rows but the canonical table only has {len(identifiers)}; "
            "the table shrank underneath the artifact. Rerun prepare with --force to regenerate it."
        )
    record = _load_manifest(paths).get(key)
    if record is None:
        _log(
            f"No artifact manifest entry for {key}; trusting its existing {artifact_rows}-row prefix "
            "and recording it now (datasets prepared before append mode have no manifest)."
        )
        _record_artifact_prefix(paths, key, identifiers, artifact_rows)
        return artifact_rows
    if int(record.get("rows", -1)) != artifact_rows or record.get("id_hash") != _id_hash(identifiers[:artifact_rows]):
        raise ValueError(
            f"{key} no longer lines up with the canonical table it was generated from "
            f"(manifest rows={record.get('rows')}, artifact rows={artifact_rows}). "
            "Rerun prepare with --force to regenerate the derived artifacts."
        )
    return artifact_rows


def _feature_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(load_feature_array(path).shape[0])


def _matrix_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    matrix = load_feature_array(path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{path.name} must be a square matrix, got shape={matrix.shape}")
    return int(matrix.shape[0])


def _protein_feature_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as handle:
        return len(pickle.load(handle))


def _atomic_pickle_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(obj, handle)
    os.replace(temporary, path)


def _read_canonical_tables(paths: DatasetPaths):
    """Load interactions/drugs/targets, tolerating the inference-only layout.

    batch-predict can target a dataset that has entities but no labelled
    interactions at all, so an absent or empty df_less1000.csv is a valid
    state here rather than an error.
    """
    import pandas as pd

    interaction_columns = ["Drug_ID", "Drug", "Target_ID", "Target", "Y", "Label", "Target_Length"]
    if paths.interaction_table.exists():
        interactions = pd.read_csv(paths.interaction_table)
    else:
        interactions = pd.DataFrame(columns=interaction_columns)
    drugs = pd.read_csv(paths.drugs_table) if paths.drugs_table.exists() else pd.DataFrame(columns=["Drug_ID", "smiles"])
    targets = pd.read_csv(paths.targets_table) if paths.targets_table.exists() else pd.DataFrame(columns=["Target_ID", "Target"])
    return interactions, drugs, targets


def _mint_entity_ids(existing_ids, prefix: str, letter: str, count: int) -> list[str]:
    """Allocate `count` ids that cannot collide with anything already in the table."""
    import re

    taken = {str(identifier) for identifier in existing_ids}
    pattern = re.compile(rf"_{letter}(\d+)$")
    next_index = 0
    for identifier in taken:
        match = pattern.search(str(identifier))
        if match:
            next_index = max(next_index, int(match.group(1)))
    minted: list[str] = []
    for _ in range(count):
        next_index += 1
        candidate = f"{prefix}_{letter}{next_index}"
        while candidate in taken:
            next_index += 1
            candidate = f"{prefix}_{letter}{next_index}"
        taken.add(candidate)
        minted.append(candidate)
    return minted


@dataclass
class AppendResult:
    new_drugs: int = 0
    new_targets: int = 0
    new_interactions: int = 0
    duplicate_interactions: int = 0
    drug_ids_by_smiles: dict | None = None
    target_ids_by_sequence: dict | None = None


def append_entities(
    paths: DatasetPaths,
    dataset_prefix: str,
    smiles_values=(),
    sequence_values=(),
    interactions_to_add=None,
) -> AppendResult:
    """Extend the canonical tables with whatever substances/targets are new.

    Existing SMILES and sequences keep their canonical Drug_ID / Target_ID (and
    therefore all of their expensive derived artifacts); unseen ones are minted
    fresh ids and appended to the end of drugs.csv / targets.csv so the
    positional alignment with kpgt_base.npz and prot_rep.pkl still holds.
    """
    import pandas as pd

    existing_interactions, drugs, targets = _read_canonical_tables(paths)

    drug_ids_by_smiles = {str(smiles): str(drug_id) for drug_id, smiles in zip(drugs["Drug_ID"], drugs["smiles"])}
    target_ids_by_sequence = {str(sequence): str(target_id) for target_id, sequence in zip(targets["Target_ID"], targets["Target"])}

    incoming_smiles = [str(value) for value in smiles_values]
    incoming_sequences = [str(value) for value in sequence_values]
    if interactions_to_add is not None and not interactions_to_add.empty:
        incoming_smiles = incoming_smiles + interactions_to_add["Drug"].astype(str).tolist()
        incoming_sequences = incoming_sequences + interactions_to_add["Target"].astype(str).tolist()

    new_smiles = list(dict.fromkeys(value for value in incoming_smiles if value not in drug_ids_by_smiles))
    new_sequences = list(dict.fromkeys(value for value in incoming_sequences if value not in target_ids_by_sequence))

    if new_smiles:
        minted = _mint_entity_ids(drugs["Drug_ID"], dataset_prefix, "D", len(new_smiles))
        drug_ids_by_smiles.update(dict(zip(new_smiles, minted)))
        drugs = pd.concat(
            [drugs, pd.DataFrame({"Drug_ID": minted, "smiles": new_smiles})],
            ignore_index=True,
            sort=False,
        )
    if new_sequences:
        minted = _mint_entity_ids(targets["Target_ID"], dataset_prefix, "T", len(new_sequences))
        target_ids_by_sequence.update(dict(zip(new_sequences, minted)))
        targets = pd.concat(
            [targets, pd.DataFrame({"Target_ID": minted, "Target": new_sequences})],
            ignore_index=True,
            sort=False,
        )

    new_interactions = 0
    duplicate_interactions = 0
    if interactions_to_add is not None and not interactions_to_add.empty:
        incoming = interactions_to_add.copy()
        incoming["Drug_ID"] = incoming["Drug"].astype(str).map(drug_ids_by_smiles)
        incoming["Target_ID"] = incoming["Target"].astype(str).map(target_ids_by_sequence)
        existing_pairs = set(
            zip(existing_interactions["Drug_ID"].astype(str), existing_interactions["Target_ID"].astype(str))
        ) if not existing_interactions.empty else set()
        incoming_pairs = list(zip(incoming["Drug_ID"].astype(str), incoming["Target_ID"].astype(str)))
        keep_mask = [pair not in existing_pairs for pair in incoming_pairs]
        duplicate_interactions = len(keep_mask) - sum(keep_mask)
        incoming = incoming[keep_mask].reset_index(drop=True)
        new_interactions = len(incoming)
        if new_interactions:
            merged = pd.concat([existing_interactions, incoming], ignore_index=True, sort=False)
            write_table(merged, paths.interaction_table)

    if new_smiles or not paths.drugs_table.exists():
        write_table(drugs, paths.drugs_table)
    if new_sequences or not paths.targets_table.exists():
        write_table(targets, paths.targets_table)
    if not paths.interaction_table.exists():
        write_table(existing_interactions, paths.interaction_table)

    _log(
        f"Appended to {paths.name}: "
        f"new_substances={len(new_smiles)}, new_targets={len(new_sequences)}, "
        f"new_interactions={new_interactions}, duplicate_interactions_skipped={duplicate_interactions}; "
        f"totals now substances={len(drugs)}, targets={len(targets)}"
    )
    return AppendResult(
        new_drugs=len(new_smiles),
        new_targets=len(new_sequences),
        new_interactions=new_interactions,
        duplicate_interactions=duplicate_interactions,
        drug_ids_by_smiles=drug_ids_by_smiles,
        target_ids_by_sequence=target_ids_by_sequence,
    )


def _drop_dataset_entities(
    paths: DatasetPaths,
    *,
    drug_ids=(),
    target_ids=(),
    exclusion_rows=None,
) -> None:
    """Remove entities from the canonical tables and compact every aligned artifact.

    Works no matter which stage discovered the problem: an artifact whose
    prefix already covered a dropped row is compacted with the same positional
    mask, so kpgt_base.npz / prot_rep.pkl / the similarity matrices stay lined
    up with the tables afterwards.
    """
    import numpy as np
    import pandas as pd

    drug_ids = {str(value) for value in drug_ids}
    target_ids = {str(value) for value in target_ids}
    if not drug_ids and not target_ids:
        return

    interactions, drugs, targets = _read_canonical_tables(paths)
    manifest = _load_manifest(paths)

    keep_drugs = ~drugs["Drug_ID"].astype(str).isin(drug_ids) if len(drugs) else pd.Series([], dtype=bool)
    keep_targets = ~targets["Target_ID"].astype(str).isin(target_ids) if len(targets) else pd.Series([], dtype=bool)
    keep_drug_mask = keep_drugs.to_numpy() if len(drugs) else np.zeros(0, dtype=bool)
    keep_target_mask = keep_targets.to_numpy() if len(targets) else np.zeros(0, dtype=bool)

    def compact_feature_file(path: Path, key: str, keep_mask):
        if not path.exists():
            return
        features = load_feature_array(path)
        rows = int(features.shape[0])
        if rows > len(keep_mask):
            raise ValueError(f"{path.name} has {rows} rows but the table only has {len(keep_mask)}")
        prefix_mask = keep_mask[:rows]
        if prefix_mask.all():
            return
        np.savez(path, fps=features[prefix_mask])
        manifest.pop(key, None)
        _log(f"Compacted {path.name}: {rows} -> {int(prefix_mask.sum())} rows")

    def compact_square_matrix(path: Path, key: str, keep_mask):
        if not path.exists():
            return
        matrix = load_feature_array(path)
        rows = int(matrix.shape[0])
        if rows > len(keep_mask):
            raise ValueError(f"{path.name} has {rows} rows but the table only has {len(keep_mask)}")
        prefix_mask = keep_mask[:rows]
        if prefix_mask.all():
            return
        indices = np.flatnonzero(prefix_mask)
        np.savez(path, matrix[np.ix_(indices, indices)])
        manifest.pop(key, None)
        _log(f"Compacted {path.name}: {rows} -> {len(indices)} rows")

    if drug_ids:
        compact_feature_file(paths.drug_features, "drug_features", keep_drug_mask)
        compact_feature_file(paths.drug_similarity, "drug_similarity", keep_drug_mask)
    if target_ids:
        if paths.protein_features.exists():
            with open(paths.protein_features, "rb") as handle:
                protein_features = pickle.load(handle)
            rows = len(protein_features)
            if rows > len(keep_target_mask):
                raise ValueError(f"prot_rep.pkl has {rows} entries but targets.csv only has {len(keep_target_mask)}")
            prefix_mask = keep_target_mask[:rows]
            if not prefix_mask.all():
                _atomic_pickle_dump(
                    [entry for entry, keep in zip(protein_features, prefix_mask) if keep],
                    paths.protein_features,
                )
                manifest.pop("protein_features", None)
                _log(f"Compacted prot_rep.pkl: {rows} -> {int(prefix_mask.sum())} entries")
        compact_square_matrix(paths.target_similarity, "target_similarity", keep_target_mask)

    before_interactions = len(interactions)
    if len(interactions):
        interaction_mask = ~(
            interactions["Drug_ID"].astype(str).isin(drug_ids)
            | interactions["Target_ID"].astype(str).isin(target_ids)
        )
        interactions = interactions[interaction_mask].reset_index(drop=True)
        if before_interactions and interactions.empty:
            raise ValueError(
                f"Preparation exclusions removed every interaction in {paths.interaction_table}; "
                "check excluded.csv for the rejected substances and targets."
            )
        write_table(interactions, paths.interaction_table)

    if drug_ids:
        write_table(drugs[keep_drug_mask].reset_index(drop=True), paths.drugs_table)
    if target_ids:
        write_table(targets[keep_target_mask].reset_index(drop=True), paths.targets_table)

    _save_manifest(paths, manifest)
    if exclusion_rows:
        _record_exclusions(paths, exclusion_rows)
    _log(
        "Dropped failed entities: "
        f"substances={len(drug_ids)}, targets={len(target_ids)}, "
        f"interactions={before_interactions}->{len(interactions)}; details={paths.excluded_table}"
    )


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


def _kpgt_pending_paths(paths: DatasetPaths) -> dict[str, Path]:
    drugs_dir = paths.drugs_table.parent
    output_npz = drugs_dir / "kpgt_pending.npz"
    return {
        "input_csv": drugs_dir / "kpgt_pending.csv",
        "output_npz": output_npz,
        "excluded_csv": drugs_dir / "kpgt_pending_excluded.csv",
        "state": drugs_dir / "kpgt_pending_state.json",
        # Owned by scripts/kpgt_streaming_extract.py, derived from output_npz.
        "temp_npy": output_npz.with_suffix(".fps.tmp.npy"),
        "temp_resume": output_npz.with_suffix(".kpgt_resume.json"),
    }


def _reset_kpgt_pending(pending: dict[str, Path]) -> None:
    for key in ("output_npz", "excluded_csv", "temp_npy", "temp_resume", "state"):
        pending[key].unlink(missing_ok=True)


def _sync_kpgt_features(paths: DatasetPaths, config: PreparationConfig) -> None:
    """Extend kpgt_base.npz so it covers every row of drugs.csv.

    Only the drugs appended since the last run are streamed through KPGT; the
    already-computed prefix is loaded and re-saved untouched. A crashed run is
    recovered twice over: the extractor resumes inside its own chunk temp
    files, and a completed-but-unmerged pending npz is reused as-is.
    """
    import numpy as np
    import pandas as pd

    drugs = pd.read_csv(paths.drugs_table)
    if drugs.empty:
        raise ValueError(f"No substances to featurize in {paths.drugs_table}")

    done = 0 if config.force else _synced_prefix(
        paths, "drug_features", _feature_row_count(paths.drug_features), drugs["Drug_ID"]
    )
    if done == len(drugs):
        _log(f"Skipping KPGT features: all {done} substances already present in {paths.drug_features}")
        return

    pending = drugs.iloc[done:].reset_index(drop=True)
    _log(f"Generating KPGT features for {len(pending)} new substances ({done} already done) into {paths.drug_features}")

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

    pending_paths = _kpgt_pending_paths(paths)
    pending_state = {
        "done": int(done),
        "pending_rows": int(len(pending)),
        "pending_hash": _id_hash(pending["smiles"].astype(str)),
    }
    saved_state = None
    if pending_paths["state"].exists():
        try:
            saved_state = json.loads(pending_paths["state"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved_state = None
    if saved_state != pending_state:
        if saved_state is not None:
            _log("Discarding stale KPGT pending state: the substance table changed since the interrupted run")
        _reset_kpgt_pending(pending_paths)
        write_table(pending, pending_paths["input_csv"])
        pending_paths["state"].write_text(json.dumps(pending_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if pending_paths["output_npz"].exists():
        _log(f"Reusing completed KPGT output from an earlier interrupted run: {pending_paths['output_npz']}")
    else:
        _run_subprocess(
            [
                str(kpgt_python),
                str(streaming_script),
                "--input-csv",
                str(pending_paths["input_csv"]),
                "--output-npz",
                str(pending_paths["output_npz"]),
                "--excluded-csv",
                str(pending_paths["excluded_csv"]),
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
        _require_file(pending_paths["output_npz"], "KPGT drug feature output")

    excluded_positions: list[int] = []
    if pending_paths["excluded_csv"].exists():
        excluded_df = pd.read_csv(pending_paths["excluded_csv"])
        if not excluded_df.empty and "source_row" in excluded_df.columns:
            excluded_positions = sorted({int(value) for value in excluded_df["source_row"]})

    new_features = load_feature_array(pending_paths["output_npz"])
    expected_new = len(pending) - len(excluded_positions)
    if int(new_features.shape[0]) != expected_new:
        raise RuntimeError(
            f"KPGT returned {new_features.shape[0]} feature rows for {len(pending)} substances "
            f"minus {len(excluded_positions)} exclusions; expected {expected_new}. "
            f"Delete {pending_paths['output_npz']} and rerun."
        )

    if excluded_positions:
        excluded_drug_ids = pending.iloc[excluded_positions]["Drug_ID"].astype(str).tolist()
        exclusion_rows = [
            {
                "source_row": int(done + position),
                "entity_type": "drug",
                "Drug_ID": pending.iloc[position]["Drug_ID"],
                "smiles": pending.iloc[position]["smiles"],
                "Target_ID": None,
                "Target": None,
                "exclusion_reason": "KPGT graph construction failed",
            }
            for position in excluded_positions
        ]
        if not config.drop_failed_entities:
            raise RuntimeError(
                f"KPGT could not build graphs for {len(excluded_drug_ids)} substances "
                f"(first: {excluded_drug_ids[:3]}). Rerun with --drop-failed-entities to record them in "
                f"{paths.excluded_table} and continue without them."
            )
        # Drop before writing the extended npz: the excluded rows all live past
        # the existing prefix, so nothing already aligned has to be compacted.
        _drop_dataset_entities(paths, drug_ids=excluded_drug_ids, exclusion_rows=exclusion_rows)

    if done:
        existing = load_feature_array(paths.drug_features)[:done]
        features = np.vstack([existing, new_features])
    else:
        features = new_features
    save_feature_array(paths.drug_features, features)
    _reset_kpgt_pending(pending_paths)
    pending_paths["input_csv"].unlink(missing_ok=True)

    final_drugs = pd.read_csv(paths.drugs_table)
    if len(final_drugs) != len(features):
        raise RuntimeError(
            f"KPGT sync ended with {len(features)} feature rows for {len(final_drugs)} substances in "
            f"{paths.drugs_table}; rerun prepare with --force."
        )
    _record_artifact_prefix(paths, "drug_features", final_drugs["Drug_ID"], len(features))
    _log(f"KPGT features now cover {len(features)} substances at {paths.drug_features}")


def _load_pdb_sequence(pdb_path: Path) -> str:
    from Bio.PDB import PDBParser

    amino_acid_map = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y", "SEC": "U",
    }
    structure = PDBParser(QUIET=True).get_structure("protein", str(pdb_path))
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " " and "CA" in residue:
                    residues.append(amino_acid_map.get(residue.resname, "X"))
    return "".join(residues)


def _record_exclusions(paths: DatasetPaths, excluded_rows: list[dict]) -> None:
    if not excluded_rows:
        return

    import pandas as pd

    new_exclusions = pd.DataFrame(excluded_rows)
    if paths.excluded_table.exists():
        exclusions = pd.read_csv(paths.excluded_table)
        exclusions = pd.concat([exclusions, new_exclusions], ignore_index=True, sort=False)
    else:
        exclusions = new_exclusions
    identifying_columns = [
        column
        for column in ["entity_type", "Drug_ID", "Target_ID", "exclusion_reason"]
        if column in exclusions.columns
    ]
    exclusions = exclusions.drop_duplicates(subset=identifying_columns, keep="last")
    write_table(exclusions, paths.excluded_table)


def _restrict_to_existing_esmfold_targets(paths: DatasetPaths, config: PreparationConfig) -> None:
    """With --skip-esmfold, keep only targets that already have a matching PDB.

    Targets whose ESM-2 embeddings already exist were validated by an earlier
    run, so only the appended tail is re-checked; every rejected target is
    recorded in excluded.csv and removed through _drop_dataset_entities, which
    compacts whatever aligned artifacts already covered it.
    """
    if not config.skip_esmfold:
        return

    import pandas as pd

    targets = pd.read_csv(paths.targets_table)
    validated = 0 if config.force else _synced_prefix(
        paths, "protein_features", _protein_feature_count(paths.protein_features), targets["Target_ID"]
    )
    pending = targets.iloc[validated:]
    if pending.empty:
        _log(f"Skipping ESMFold generation: all {len(targets)} canonical targets were already validated")
        return

    _log(f"Validating existing ESMFold PDBs for {len(pending)} new targets ({validated} already validated)")
    excluded_rows = []
    excluded_target_ids = []
    exclusion_counts = {"missing": 0, "unreadable": 0, "mismatched": 0}
    for source_row, row in pending.iterrows():
        target_id = str(row["Target_ID"])
        canonical_sequence = str(row["Target"]).upper()
        pdb_path = paths.esmfold_dir / f"{target_id}.pdb"
        if not pdb_path.exists():
            reason = "Missing ESMFold PDB"
            exclusion_kind = "missing"
        else:
            try:
                pdb_sequence = _load_pdb_sequence(pdb_path)
            except Exception as exc:
                reason = f"Unreadable ESMFold PDB: {type(exc).__name__}: {exc}"
                exclusion_kind = "unreadable"
            else:
                if pdb_sequence == canonical_sequence:
                    continue
                reason = (
                    "ESMFold sequence mismatch: "
                    f"canonical_residues={len(canonical_sequence)}, pdb_residues={len(pdb_sequence)}"
                )
                exclusion_kind = "mismatched"

        exclusion_counts[exclusion_kind] += 1
        if exclusion_kind != "missing":
            _log(f"Excluding target {target_id}: {reason}")
        excluded_target_ids.append(target_id)
        excluded_rows.append(
            {
                "source_row": int(source_row),
                "entity_type": "target",
                "Drug_ID": None,
                "smiles": None,
                "Target_ID": target_id,
                "Target": row["Target"],
                "exclusion_reason": reason,
            }
        )

    if not excluded_target_ids:
        _log(f"Skipping ESMFold generation: all {len(pending)} new targets already have matching PDB files")
        return
    _log(
        "Excluded targets while reusing ESMFold PDBs: "
        f"missing={exclusion_counts['missing']}, "
        f"unreadable={exclusion_counts['unreadable']}, "
        f"sequence_mismatch={exclusion_counts['mismatched']}; "
        f"details={paths.excluded_table}"
    )
    if len(excluded_target_ids) == len(targets):
        raise ValueError(
            f"--skip-esmfold found no matching calculated target PDBs from {paths.targets_table} in {paths.esmfold_dir}"
        )
    _drop_dataset_entities(paths, target_ids=excluded_target_ids, exclusion_rows=excluded_rows)


def _sync_protein_features(paths: DatasetPaths, config: PreparationConfig) -> None:
    """Extend prot_rep.pkl so it covers every row of targets.csv.

    The pickle is rewritten atomically every --esm-checkpoint-seconds rather
    than only at the end, so an interrupted embedding run restarts from the
    last checkpoint instead of from the first target.
    """
    import time

    import pandas as pd
    import torch

    targets_df = pd.read_csv(paths.targets_table)
    if targets_df.empty:
        raise ValueError(f"No targets to embed in {paths.targets_table}")

    done = 0 if config.force else _synced_prefix(
        paths, "protein_features", _protein_feature_count(paths.protein_features), targets_df["Target_ID"]
    )
    if done == len(targets_df):
        _log(f"Skipping ESM-2 embeddings: all {done} targets already present in {paths.protein_features}")
        return

    if done:
        with open(paths.protein_features, "rb") as handle:
            token_representations = pickle.load(handle)[:done]
    else:
        token_representations = []

    pending = targets_df.iloc[done:]
    _log(f"Generating ESM-2 embeddings for {len(pending)} new targets ({done} already done)")
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

    repr_layer = 33
    last_checkpoint = time.monotonic()
    for row in _iterate_with_progress(
        pending.itertuples(index=False),
        total=len(pending),
        desc="ESM-2 embeddings",
        unit="target",
    ):
        batch_labels, batch_strs, batch_tokens = batch_converter([(row.Target_ID, row.Target)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        token_representations.append(results["representations"][repr_layer].cpu().detach().numpy())
        if config.esm_checkpoint_seconds and time.monotonic() - last_checkpoint >= config.esm_checkpoint_seconds:
            _atomic_pickle_dump(token_representations, paths.protein_features)
            _record_artifact_prefix(paths, "protein_features", targets_df["Target_ID"], len(token_representations))
            _log(f"Checkpointed ESM-2 embeddings at {len(token_representations)}/{len(targets_df)} targets")
            last_checkpoint = time.monotonic()

    _atomic_pickle_dump(token_representations, paths.protein_features)
    _record_artifact_prefix(paths, "protein_features", targets_df["Target_ID"], len(token_representations))
    _log(f"ESM-2 embeddings now cover {len(token_representations)} targets at {paths.protein_features}")


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
    failed_rows = []
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
                    message = (
                        "ESMFold ran out of CUDA memory while generating "
                        f"Target_ID={row.Target_ID} ({target_length} residues, chunk_size=1). "
                        "Rerun prepare without --force so completed PDBs are skipped; this target likely needs "
                        "CPU ESMFold, a larger-memory GPU, or removal from the dataset."
                    )
                    if not config.drop_failed_entities:
                        raise RuntimeError(message) from exc
                    _log(f"Dropping target {row.Target_ID}: {message}")
                    failed_rows.append(
                        {
                            "source_row": None,
                            "entity_type": "target",
                            "Drug_ID": None,
                            "smiles": None,
                            "Target_ID": str(row.Target_ID),
                            "Target": row.Target,
                            "exclusion_reason": f"ESMFold CUDA OOM at chunk_size=1 ({target_length} residues)",
                        }
                    )
                    break

                next_chunk_size = 128 if current_chunk_size is None else max(1, current_chunk_size // 2)
                _log(
                    "ESMFold CUDA OOM for "
                    f"Target_ID={row.Target_ID} ({target_length} residues) at chunk_size={chunk_size_label}; "
                    f"retrying with chunk_size={next_chunk_size}"
                )
                current_chunk_size = next_chunk_size
                model.set_chunk_size(current_chunk_size)

        if output is None:
            continue
        with open(pdb_path, "w", encoding="utf-8") as handle:
            handle.write(output)
        output = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if failed_rows:
        _drop_dataset_entities(
            paths,
            target_ids=[row["Target_ID"] for row in failed_rows],
            exclusion_rows=failed_rows,
        )


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

    failed_rows = []
    graph_inputs = zip(targets_df.iterrows(), protein_features)
    for (source_row, row), embedding in _iterate_with_progress(
        graph_inputs,
        total=len(targets_df),
        desc="Protein graphs",
        unit="target",
    ):
        graph_path = paths.graph_dir / f"{row['Target_ID']}.pt"
        if graph_path.exists() and not config.force:
            continue

        try:
            pdb_path = paths.esmfold_dir / f"{row['Target_ID']}.pdb"
            _require_file(pdb_path, f"ESMFold structure for {row['Target_ID']}")
            dis_map, seq = load_predicted_PDB3(str(pdb_path))
            emb = embedding[0][1 : len(embedding[0]) - 1]
            if len(seq) != len(emb):
                raise ValueError(
                    f"Sequence length mismatch for {row['Target_ID']}: ESMFold residues={len(seq)}, ESM-2 residues={len(emb)}"
                )
        except Exception as exc:
            if not config.drop_failed_entities:
                raise
            _log(f"Dropping target {row['Target_ID']}: protein graph build failed: {type(exc).__name__}: {exc}")
            failed_rows.append(
                {
                    "source_row": int(source_row),
                    "entity_type": "target",
                    "Drug_ID": None,
                    "smiles": None,
                    "Target_ID": str(row["Target_ID"]),
                    "Target": row["Target"],
                    "exclusion_reason": f"Protein graph build failed: {type(exc).__name__}: {exc}",
                }
            )
            continue

        row_idx, col_idx = np.where(dis_map <= 8)
        graph = protein_graph(seq, [row_idx, col_idx], emb)
        torch.save(graph, graph_path)

    if failed_rows:
        _drop_dataset_entities(
            paths,
            target_ids=[row["Target_ID"] for row in failed_rows],
            exclusion_rows=failed_rows,
        )


def _generate_similarity_matrices(paths: DatasetPaths, config: PreparationConfig) -> None:
    """Extend the contrastive-learning similarity artifacts to cover new entities.

    Both are training-only inputs (`utils.custom_collate_fn`); inference and
    batch-predict never touch them, which is why --skip-similarity exists for
    screening runs that would otherwise pay for an O(n^2) TM-score job.
    """
    import numpy as np
    import pandas as pd

    from sim_matrix import compute_and_extend_tm_score_matrix, compute_drug_fingerprints

    drugs_df = pd.read_csv(paths.drugs_table)
    targets_df = pd.read_csv(paths.targets_table)

    if config.skip_similarity:
        _log(
            "Skipping similarity artifacts because --skip-similarity was passed. "
            "Inference and batch-predict do not need them, but this dataset is not train-ready "
            "until prepare is rerun without the flag."
        )
        return

    drugs_done = 0 if config.force else _synced_prefix(
        paths, "drug_similarity", _feature_row_count(paths.drug_similarity), drugs_df["Drug_ID"]
    )
    if drugs_done == len(drugs_df):
        _log(f"Skipping drug fingerprints: all {drugs_done} substances already present in {paths.drug_similarity}")
    else:
        _log(f"Generating drug fingerprints for {len(drugs_df) - drugs_done} new substances at {paths.drug_similarity}")
        new_fingerprints = compute_drug_fingerprints(drugs_df.iloc[drugs_done:])
        if drugs_done:
            fingerprints = np.vstack([load_feature_array(paths.drug_similarity)[:drugs_done], new_fingerprints])
        else:
            fingerprints = new_fingerprints
        save_feature_array(paths.drug_similarity, fingerprints)
        _record_artifact_prefix(paths, "drug_similarity", drugs_df["Drug_ID"], len(fingerprints))

    targets_done = 0 if config.force else _synced_prefix(
        paths, "target_similarity", _matrix_row_count(paths.target_similarity), targets_df["Target_ID"]
    )
    if targets_done == len(targets_df):
        _log(f"Skipping target similarity matrix: all {targets_done} targets already present in {paths.target_similarity}")
        return

    missing_pdbs = [target_id for target_id in targets_df["Target_ID"] if not (paths.esmfold_dir / f"{target_id}.pdb").exists()]
    if missing_pdbs:
        missing_preview = ", ".join(map(str, missing_pdbs[:5]))
        raise FileNotFoundError(f"Missing ESMFold PDB files for TM-score matrix generation: {missing_preview}")
    _log(f"Extending target similarity matrix from {targets_done} to {len(targets_df)} targets at {paths.target_similarity}")
    compute_and_extend_tm_score_matrix(
        targets_df,
        paths.esmfold_dir,
        paths.target_similarity,
        start_row=targets_done,
        num_processes=config.target_similarity_processes,
        chunksize=config.target_similarity_chunksize,
    )
    _record_artifact_prefix(paths, "target_similarity", targets_df["Target_ID"], len(targets_df))


def prepare_dataset(
    dataset_name: str,
    input_path: str | None = None,
    smiles_column: str = "smiles",
    sequence_column: str = "sequence",
    activity_column: str = "activity",
    config: PreparationConfig | None = None,
    extra_smiles=None,
    extra_sequences=None,
) -> DatasetPaths:
    """Prepare a dataset, extending it in place when it already exists.

    `--input` supplies labelled interactions; `extra_smiles`/`extra_sequences`
    supply bare entities (what batch-predict passes). Either way, substances
    and targets that are already canonical keep their ids and their derived
    artifacts, and only the new ones are pushed through KPGT / ESM-2 / ESMFold
    / graph building. Pass `rebuild` (or `force`) to get the old behaviour of
    rewriting the canonical tables from `--input`.
    """
    import pandas as pd

    config = config or PreparationConfig()
    paths = get_dataset_paths(dataset_name, artifacts_dir=config.artifacts_dir)
    ensure_dataset_dirs(paths)

    already_prepared = paths.drugs_table.exists() and paths.targets_table.exists()
    rebuild = config.rebuild or config.force or not already_prepared

    incoming_interactions = None
    if input_path is not None:
        source = Path(input_path).expanduser().resolve()
        incoming_interactions = normalize_interaction_dataframe(
            read_table(source),
            smiles_column=smiles_column,
            sequence_column=sequence_column,
            activity_column=activity_column,
            threshold=config.threshold,
            dataset_prefix=dataset_name,
        )

    if rebuild:
        if incoming_interactions is not None:
            _log(f"Rebuilding canonical tables for {dataset_name} from {input_path}")
            write_table(incoming_interactions, paths.interaction_table)
            write_table(build_drugs_table(incoming_interactions), paths.drugs_table)
            write_table(build_targets_table(incoming_interactions), paths.targets_table)
            # The manifest is deliberately left alone: rerunning a rebuild over
            # identical input reproduces identical ids and should reuse the
            # existing artifacts, while a rebuild over changed input makes
            # _synced_prefix fail loud instead of pairing the wrong rows.
        if extra_smiles or extra_sequences:
            append_entities(
                paths,
                dataset_name,
                smiles_values=extra_smiles or (),
                sequence_values=extra_sequences or (),
            )
    else:
        _log(f"Appending to the already-prepared dataset {dataset_name} under {paths.root}")
        append_entities(
            paths,
            dataset_name,
            smiles_values=extra_smiles or (),
            sequence_values=extra_sequences or (),
            interactions_to_add=incoming_interactions,
        )

    _require_file(paths.drugs_table, "drug table")
    _require_file(paths.targets_table, "target table")
    if not paths.interaction_table.exists():
        # batch-predict can prepare a dataset that has entities but no labelled
        # interactions at all; keep the canonical file present but empty.
        write_table(
            pd.DataFrame(columns=["Drug_ID", "Drug", "Target_ID", "Target", "Y", "Label", "Target_Length"]),
            paths.interaction_table,
        )

    stages = [
        ("KPGT features", _sync_kpgt_features),
        ("ESM-2 embeddings", _sync_protein_features),
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
                _restrict_to_existing_esmfold_targets(paths, config)
            if progress is not None:
                progress.update(1)
            _log(f"Preparing {dataset_name}: {stage_index}/{len(stages)} stages ({(stage_index / len(stages)) * 100:.1f}%)")
    finally:
        if progress is not None:
            progress.close()
    return paths


def repair_interaction_labels(
    dataset_name: str,
    input_path: str,
    smiles_column: str = "smiles",
    sequence_column: str = "sequence",
    activity_column: str = "activity",
    threshold: float = 0.0,
    artifacts_dir: str | None = None,
) -> DatasetPaths:
    """Repair only Y/Label in an already-prepared canonical interaction table."""
    import pandas as pd

    paths = get_dataset_paths(dataset_name, artifacts_dir=artifacts_dir)
    _require_file(paths.interaction_table, "canonical interaction table")

    source = Path(input_path).expanduser().resolve()
    repaired_source = normalize_interaction_dataframe(
        read_table(source),
        smiles_column=smiles_column,
        sequence_column=sequence_column,
        activity_column=activity_column,
        threshold=threshold,
        dataset_prefix=dataset_name,
    )[["Drug", "Target", "Y", "Label"]]
    existing = pd.read_csv(paths.interaction_table)

    key_columns = ["Drug", "Target"]
    occurrence_column = "_pair_occurrence"
    repaired_source[occurrence_column] = repaired_source.groupby(key_columns, sort=False).cumcount()
    existing[occurrence_column] = existing.groupby(key_columns, sort=False).cumcount()
    repaired_values = repaired_source[key_columns + [occurrence_column, "Y", "Label"]]
    repaired = existing.drop(columns=["Y", "Label"]).merge(
        repaired_values,
        on=key_columns + [occurrence_column],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    missing = repaired["Y"].isna() | repaired["Label"].isna()
    if missing.any():
        sample = repaired.loc[missing, key_columns].head(3).to_dict("records")
        raise ValueError(
            f"Could not match {int(missing.sum())} existing interaction rows to the raw input; "
            f"examples: {sample}"
        )

    output_columns = existing.columns.drop(occurrence_column)
    repaired = repaired.drop(columns=[occurrence_column])[output_columns]
    write_table(repaired, paths.interaction_table)
    _log(
        f"Repaired Y and Label for {len(repaired)} rows in {paths.interaction_table}; "
        "all derived drug and target artifacts were left unchanged"
    )
    return paths


def _load_training_assets(paths: DatasetPaths):
    drug_features, drug_df, target_df, drug_dict = _load_model_assets(paths)
    _log(f"Loading drug fingerprints: {paths.drug_similarity}")
    drug_fingerprints = load_feature_array(paths.drug_similarity)
    _log(f"Loading target similarity matrix: {paths.target_similarity}")
    tm_score_matrix = load_feature_array(paths.target_similarity)
    _log(
        "Loaded training assets: "
        f"{len(drug_df)} drugs, {len(target_df)} targets, "
        f"drug_features_shape={getattr(drug_features, 'shape', 'unknown')}, "
        f"drug_fingerprints_shape={getattr(drug_fingerprints, 'shape', 'unknown')}, "
        f"target_similarity_shape={getattr(tm_score_matrix, 'shape', 'unknown')}"
    )
    return drug_features, drug_df, target_df, drug_dict, drug_fingerprints, tm_score_matrix


def _load_model_assets(paths: DatasetPaths):
    import pandas as pd

    _log(f"Loading drug features: {paths.drug_features}")
    drug_features = load_feature_array(paths.drug_features)
    _log(f"Loading drugs table: {paths.drugs_table}")
    drug_df = pd.read_csv(paths.drugs_table)
    _log(f"Loading targets table: {paths.targets_table}")
    target_df = pd.read_csv(paths.targets_table)
    if len(drug_df) != len(drug_features):
        raise ValueError("drugs.csv and kpgt_base.npz have different lengths.")
    drug_dict = dict(zip(drug_df["Drug_ID"], drug_features))
    _log(
        "Loaded model assets: "
        f"{len(drug_df)} drugs, {len(target_df)} targets, "
        f"drug_features_shape={getattr(drug_features, 'shape', 'unknown')}"
    )
    return drug_features, drug_df, target_df, drug_dict


def _build_prediction_export(raw_df, predictions, probabilities):
    import numpy as np

    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    if len(raw_df) != len(predictions) or len(raw_df) != len(probabilities):
        raise ValueError("Prediction output length does not match the scored interaction table.")
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(f"Expected two-class probabilities, got shape {probabilities.shape}.")

    output = raw_df[["Drug", "Target", "Y"]].copy()
    output.columns = ["smiles", "sequence", "real_value"]
    output["label"] = predictions.astype(int)
    output["probability"] = probabilities[:, 1]
    return output[["smiles", "sequence", "label", "probability", "real_value"]]


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

    train_drug_feature, train_drug_df, train_target_df, train_drug_dict, drug_fingerprints, tm_score_matrix = _load_training_assets(train_paths)
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
        collate_fn=lambda batch: custom_collate_fn(batch, drug_fingerprints, tm_score_matrix),
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

    output_root = Path(output_dir).expanduser().resolve() if output_dir else RESULTS_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{output_name or train_dataset_name}_predictions.parquet"
    _log(f"Writing predictions to {output_path}")

    prediction_exports = [_build_prediction_export(validation_df, best_predictions, best_probs)]
    if test_paths is not None:
        prediction_exports.append(_build_prediction_export(test_df, test_preds, test_probs))
    output_df = pd.concat(prediction_exports, ignore_index=True)
    output_df.to_parquet(output_path, index=False)

    _log(f"Saved predictions to {output_path}")
    _log(f"Validation F1: {best_f1:.4f}")
    _log(f"Test F1: {test_f1:.4f}, Test Accuracy: {test_acc:.2f}")
    return output_path


def run_inference(
    dataset_name: str,
    checkpoint_path: str,
    batch_size: int = 64,
    artifacts_dir: str | None = None,
    output_dir: str | None = None,
    output_name: str | None = None,
) -> Path:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from loss import FocalLoss
    from model import GraphDTI_bi
    from utils import GraphDataset_withsim, custom_collate_fn_test, evaluate_cl

    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    _require_file(resolved_checkpoint, "inference checkpoint")

    paths = get_dataset_paths(dataset_name, artifacts_dir=artifacts_dir)
    _log(f"Starting inference for prepared dataset '{dataset_name}'")
    _log(f"Checkpoint: {resolved_checkpoint}")
    _log(f"Artifacts root: {resolve_artifacts_root(artifacts_dir)}")
    _log(f"Loading inference interactions: {paths.interaction_table}")
    interactions = pd.read_csv(paths.interaction_table)
    drug_features, drug_df, target_df, drug_dict = _load_model_assets(paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Inference device: {device}")
    if device.type == "cuda":
        _log(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    dataset = GraphDataset_withsim(
        interactions,
        drug_df,
        target_df,
        drug_dict,
        str(paths.graph_dir),
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_test,
    )
    _log(f"Inference rows={len(dataset)}, batches={len(loader)}, batch_size={batch_size}")

    model = GraphDTI_bi(drug_features[0].shape[0], 1280, 2, surface_feature=False).to(device)
    checkpoint = _load_torch_checkpoint(torch, resolved_checkpoint, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {resolved_checkpoint}")
    model.load_state_dict(checkpoint["model_state_dict"])
    _log(f"Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")

    weights = torch.tensor([1.0, 1.0], device=device)
    criterion = FocalLoss(alpha=1, gamma=2, weight=weights)
    _, accuracy, _, _, f1, predictions, _, probabilities = evaluate_cl(
        model,
        loader,
        criterion,
        device,
        desc="Inference",
    )

    output_root = Path(output_dir).expanduser().resolve() if output_dir else RESULTS_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{output_name or dataset_name}_predictions.parquet"
    _log(f"Writing inference predictions to {output_path}")
    _build_prediction_export(interactions, predictions, probabilities).to_parquet(output_path, index=False)
    _log(f"Saved inference predictions to {output_path}")
    _log(f"Inference F1: {f1:.4f}, accuracy: {accuracy:.2f}%")
    return output_path


@dataclass
class BatchPredictConfig:
    pair_batch_size: int = 64
    chunk_rows: int = 20_000_000
    probability_dtype: str = "float16"
    graph_cache_size: int = 256
    target_major: bool = True
    resume: bool = True
    output_dir: str | None = None
    progress_file: str | None = None


class _GraphCache:
    """Bounded LRU over the serialized protein graphs.

    A target graph is a residues x 1280 tensor on disk, so re-reading it for
    every pair is what makes naive cartesian scoring unusable. With the default
    target-major pair order each graph is deserialized exactly once even when
    the cache is far smaller than the target set.
    """

    def __init__(self, graph_dir: Path, capacity: int):
        from collections import OrderedDict

        self.graph_dir = Path(graph_dir)
        self.capacity = max(1, capacity)
        self._entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, target_id: str):
        import torch

        if target_id in self._entries:
            self._entries.move_to_end(target_id)
            self.hits += 1
            return self._entries[target_id]

        path = self.graph_dir / f"{target_id}.pt"
        _require_file(path, f"protein graph for {target_id}")
        graph = torch.load(path, map_location="cpu")
        self._entries[target_id] = graph
        self.misses += 1
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return graph


def _read_entity_values(
    json_path: str | None,
    parquet_path: str | None,
    value_column: str,
    id_column: str,
    label: str,
) -> list[str]:
    """Load the batch-predict entity list from either JSON or Parquet.

    A Parquet table that carries its own id column must be keyed 0..N-1, since
    those ids are what the prediction chunks are written against.
    """
    if bool(json_path) == bool(parquet_path):
        raise ValueError(f"Provide exactly one of --{label}-json or --{label}-parquet.")

    if json_path:
        raw = json.loads(Path(json_path).expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{label} JSON must contain a non-empty array of strings.")
        values = raw
    else:
        import pyarrow.parquet as pq

        table = pq.read_table(Path(parquet_path).expanduser().resolve())
        if value_column not in table.column_names:
            raise ValueError(f"{parquet_path} is missing the '{value_column}' column.")
        if id_column in table.column_names:
            frame = table.to_pandas().sort_values(id_column).reset_index(drop=True)
            expected = list(range(len(frame)))
            if frame[id_column].tolist() != expected:
                raise ValueError(
                    f"{parquet_path} must have contiguous {id_column} values 0..N-1; "
                    "regenerate it with scripts/prepare_batch_tables.py."
                )
            values = frame[value_column].tolist()
        else:
            values = table.column(value_column).to_pylist()

    cleaned = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} entry {index} is not a non-empty string.")
        cleaned.append(value.strip())
    if not cleaned:
        raise ValueError(f"No {label} values were supplied.")
    return cleaned


def run_batch_predict(
    dataset_name: str,
    checkpoint_path: str,
    substances: list[str],
    targets: list[str],
    batch_config: BatchPredictConfig | None = None,
    preparation_config: PreparationConfig | None = None,
    skip_prepare: bool = False,
) -> Path:
    """Prepare whatever is new, then score the full substances x targets product.

    The product is written as resumable, minimum-size Parquet chunks: ids are
    delta-encoded int32, the label is int8, and only the positive-class
    probability is stored (the negative one is 1 - p). A progress file records
    the next unscored pair so an interrupted run restarts where it stopped
    instead of rescoring everything.
    """
    import gc
    import itertools

    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.data import Batch

    from model import GraphDTI_bi

    from scripts.batch_predict_utils import (
        chunk_path,
        hash_strings,
        load_progress,
        pair_stream,
        save_progress,
        write_entity_map,
        write_prediction_chunk,
    )

    batch_config = batch_config or BatchPredictConfig()
    preparation_config = preparation_config or PreparationConfig(drop_failed_entities=True, skip_similarity=True)

    if batch_config.pair_batch_size < 1:
        raise ValueError("--pair-batch-size must be at least 1")
    if batch_config.chunk_rows < batch_config.pair_batch_size:
        raise ValueError("--chunk-rows must be at least --pair-batch-size")

    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    _require_file(resolved_checkpoint, "batch-predict checkpoint")

    # Input order is the id space the output chunks are keyed on, so duplicates
    # are kept (and scored twice) rather than silently renumbering every id.
    substances = [str(value) for value in substances]
    targets = [str(value) for value in targets]
    duplicate_substances = len(substances) - len(set(substances))
    duplicate_targets = len(targets) - len(set(targets))
    if duplicate_substances or duplicate_targets:
        _log(
            f"Input contains {duplicate_substances} repeated substances and {duplicate_targets} repeated targets; "
            "they keep their own ids and are scored once per occurrence."
        )
    _log(f"batch-predict: {len(substances):,} substances x {len(targets):,} targets = {len(substances) * len(targets):,} pairs")

    if skip_prepare:
        _log("Skipping the preparation phase because --skip-prepare was passed")
        paths = get_dataset_paths(dataset_name, artifacts_dir=preparation_config.artifacts_dir)
    else:
        paths = prepare_dataset(
            dataset_name=dataset_name,
            config=preparation_config,
            extra_smiles=substances,
            extra_sequences=targets,
        )

    # ---- resolve input positions to canonical, actually-scorable entities ----
    _require_file(paths.drugs_table, "prepared drug table")
    _require_file(paths.targets_table, "prepared target table")
    _require_file(paths.drug_features, "prepared KPGT drug features")
    drug_features = load_feature_array(paths.drug_features)
    drugs_df = pd.read_csv(paths.drugs_table)
    targets_df = pd.read_csv(paths.targets_table)
    if len(drugs_df) != len(drug_features):
        raise ValueError(
            f"{paths.drugs_table} has {len(drugs_df)} rows but {paths.drug_features} has {len(drug_features)}; "
            "rerun prepare."
        )

    drug_row_by_smiles = {str(smiles): index for index, smiles in enumerate(drugs_df["smiles"])}
    target_row_by_sequence = {str(sequence): index for index, sequence in enumerate(targets_df["Target"])}

    scorable_substances: list[tuple[int, int]] = []  # (input id, drugs.csv row)
    dropped_substances: list[dict] = []
    for input_id, smiles in enumerate(substances):
        row = drug_row_by_smiles.get(smiles)
        if row is None:
            dropped_substances.append({"substance_id": input_id, "reason": "excluded during preparation"})
            continue
        scorable_substances.append((input_id, row))

    scorable_targets: list[tuple[int, int]] = []  # (input id, targets.csv row)
    dropped_targets: list[dict] = []
    for input_id, sequence in enumerate(targets):
        row = target_row_by_sequence.get(sequence)
        if row is None:
            dropped_targets.append({"target_id": input_id, "reason": "excluded during preparation"})
            continue
        target_id = str(targets_df.iloc[row]["Target_ID"])
        if not (paths.graph_dir / f"{target_id}.pt").exists():
            dropped_targets.append({"target_id": input_id, "reason": f"missing protein graph {target_id}.pt"})
            continue
        scorable_targets.append((input_id, row))

    if not scorable_substances or not scorable_targets:
        raise ValueError(
            "batch-predict has nothing to score: "
            f"{len(scorable_substances)} usable substances and {len(scorable_targets)} usable targets. "
            f"See {paths.excluded_table}."
        )
    if dropped_substances or dropped_targets:
        _log(
            f"Excluding {len(dropped_substances)} substances and {len(dropped_targets)} targets from the product; "
            f"reasons are recorded alongside the chunks and in {paths.excluded_table}"
        )

    substance_ids = np.asarray([input_id for input_id, _ in scorable_substances], dtype=np.int32)
    substance_rows = np.asarray([row for _, row in scorable_substances], dtype=np.int64)
    target_ids = np.asarray([input_id for input_id, _ in scorable_targets], dtype=np.int32)
    target_canonical = [str(targets_df.iloc[row]["Target_ID"]) for _, row in scorable_targets]

    num_substances = len(substance_ids)
    num_targets = len(target_ids)
    total_pairs = num_substances * num_targets

    entity_hash = hashlib.sha256(
        f"{hash_strings(substances)}:{hash_strings(targets)}".encode()
    ).hexdigest()[:12]
    run_slug = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in f"batch_{dataset_name}_{resolved_checkpoint.stem}_{entity_hash}"
    )
    output_dir = (
        Path(batch_config.output_dir).expanduser().resolve()
        if batch_config.output_dir
        else resolve_artifacts_root(preparation_config.artifacts_dir) / "batch_predict" / run_slug
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sidecars live in meta/ so the chunk directory holds only part-*.parquet
    # files of one schema and can be opened as a single Parquet dataset.
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    write_entity_map(
        meta_dir / "substance_map.parquet",
        substance_ids,
        [str(drugs_df.iloc[row]["Drug_ID"]) for _, row in scorable_substances],
        [substances[input_id] for input_id, _ in scorable_substances],
        "substance_id",
        "Drug_ID",
        "smiles",
    )
    write_entity_map(
        meta_dir / "target_map.parquet",
        target_ids,
        target_canonical,
        [targets[input_id] for input_id, _ in scorable_targets],
        "target_id",
        "Target_ID",
        "sequence",
    )
    (meta_dir / "unscored_entities.json").write_text(
        json.dumps({"substances": dropped_substances, "targets": dropped_targets}, indent=2),
        encoding="utf-8",
    )

    total_chunks = -(-total_pairs // batch_config.chunk_rows)
    digits = max(5, len(str(total_chunks)))
    fingerprint = hashlib.sha256(
        (
            f"{entity_hash}:{hash_strings(substance_ids.tolist())}:{hash_strings(target_ids.tolist())}:"
            f"{resolved_checkpoint}:{batch_config.chunk_rows}:{batch_config.probability_dtype}:"
            f"{'target' if batch_config.target_major else 'substance'}"
        ).encode()
    ).hexdigest()
    progress_path = Path(batch_config.progress_file).expanduser() if batch_config.progress_file else output_dir / "progress.json"
    progress = load_progress(progress_path, fingerprint, total_pairs, batch_config.resume)
    start_index = int(progress["next_pair_index"])
    chunk_index = int(progress["next_chunk_index"])
    if not 0 <= start_index <= total_pairs:
        raise ValueError(f"Invalid next_pair_index in progress file: {start_index}")

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"batch-predict device: {device}")
    if device.type == "cuda":
        _log(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    model = GraphDTI_bi(drug_features[0].shape[0], 1280, 2, surface_feature=False).to(device)
    checkpoint = _load_torch_checkpoint(torch, resolved_checkpoint, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {resolved_checkpoint}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _log(f"Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")

    graph_cache = _GraphCache(paths.graph_dir, batch_config.graph_cache_size)
    drug_matrix = np.asarray(drug_features)
    current_batch_size = batch_config.pair_batch_size

    def score_batch(pairs):
        """Score one list of (substance_position, target_position) pairs.

        Halves the forward-pass batch on CUDA OOM and retries instead of losing
        the whole run, mirroring the ESMFold chunk-size backoff in prepare.
        """
        rows = drug_matrix[substance_rows[[pair[0] for pair in pairs]]]
        drugs_tensor = torch.from_numpy(np.ascontiguousarray(rows)).to(torch.float32).to(device)
        graphs = Batch.from_data_list([graph_cache.get(target_canonical[pair[1]]) for pair in pairs]).to(device)
        with torch.no_grad():
            outputs, _, _ = model(drugs_tensor, graphs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predicted = torch.argmax(outputs, dim=1)
        return predicted.cpu().numpy().astype(np.int8), probabilities.cpu().numpy().astype(np.float32)

    def score_batch_with_backoff(pairs):
        nonlocal current_batch_size

        while True:
            try:
                return score_batch(pairs)
            except torch.OutOfMemoryError:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                if len(pairs) == 1:
                    raise
                half = max(1, len(pairs) // 2)
                current_batch_size = min(current_batch_size, half)
                _log(f"CUDA OOM scoring {len(pairs)} pairs; retrying in halves of {half}")
                first_labels, first_probabilities = score_batch_with_backoff(pairs[:half])
                second_labels, second_probabilities = score_batch_with_backoff(pairs[half:])
                return (
                    np.concatenate([first_labels, second_labels]),
                    np.concatenate([first_probabilities, second_probabilities]),
                )

    bar = _tqdm(
        total=total_pairs,
        initial=start_index,
        desc="GS-DTI batch-predict",
        unit="pair",
        dynamic_ncols=True,
    )
    pair_iter = pair_stream(start_index, num_substances, num_targets, target_major=batch_config.target_major)
    processed = start_index
    _log(
        f"Scoring {total_pairs:,} pairs from {start_index:,}; chunks -> {output_dir}; progress: {progress_path}"
    )

    while processed < total_pairs:
        target_chunk_path = chunk_path(output_dir, chunk_index, digits)
        chunk_end = min(processed + batch_config.chunk_rows, total_pairs)
        rows_needed = chunk_end - processed

        if batch_config.resume and target_chunk_path.exists():
            # A previous run finished this chunk but the progress file was not updated afterward.
            for _ in itertools.islice(pair_iter, rows_needed):
                pass
            if bar is not None:
                bar.update(rows_needed)
        else:
            substance_buffer = np.empty(rows_needed, dtype=np.int32)
            target_buffer = np.empty(rows_needed, dtype=np.int32)
            label_buffer = np.empty(rows_needed, dtype=np.int8)
            probability_buffer = np.empty(rows_needed, dtype=np.float32)
            filled = 0

            while filled < rows_needed:
                pair_batch = list(itertools.islice(pair_iter, min(current_batch_size, rows_needed - filled)))
                labels, probabilities = score_batch_with_backoff(pair_batch)
                batch_len = len(pair_batch)
                substance_buffer[filled : filled + batch_len] = substance_ids[[pair[0] for pair in pair_batch]]
                target_buffer[filled : filled + batch_len] = target_ids[[pair[1] for pair in pair_batch]]
                label_buffer[filled : filled + batch_len] = labels
                probability_buffer[filled : filled + batch_len] = probabilities
                filled += batch_len
                if bar is not None:
                    bar.update(batch_len)
                    bar.set_postfix(
                        chunk=chunk_index,
                        batch=current_batch_size,
                        graph_hit=f"{graph_cache.hits / max(1, graph_cache.hits + graph_cache.misses):.2%}",
                    )

            write_prediction_chunk(
                target_chunk_path,
                substance_buffer,
                target_buffer,
                label_buffer,
                probability_buffer,
                probability_dtype=batch_config.probability_dtype,
            )
            del substance_buffer, target_buffer, label_buffer, probability_buffer
            gc.collect()

        processed = chunk_end
        chunk_index += 1
        progress["next_pair_index"] = processed
        progress["next_chunk_index"] = chunk_index
        save_progress(progress_path, progress)
        _log(f"batch-predict progress: {processed:,}/{total_pairs:,} pairs ({(processed / total_pairs) * 100:.2f}%)")

    close = getattr(bar, "close", None)
    if callable(close):
        close()
    _log(
        f"Wrote {chunk_index:,} chunk file(s) to {output_dir}; "
        f"graph cache hits={graph_cache.hits:,}, misses={graph_cache.misses:,}"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    formatter = lambda prog: argparse.HelpFormatter(prog, width=180)
    parser = argparse.ArgumentParser(
        description="Unified GS-DTI data preparation, training, and inference entrypoint.",
        formatter_class=formatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_raw_input_args(cmd):
        cmd.add_argument("--input", help="Custom raw dataset in CSV or Parquet")
        cmd.add_argument("--smiles-column", default="smiles")
        cmd.add_argument("--sequence-column", default="sequence")
        cmd.add_argument("--activity-column", default="activity")
        cmd.add_argument(
            "--threshold",
            type=float,
            default=0.0,
            help="Threshold to derive Label from non-binary activity; exact 0/1 activity is preserved as Label",
        )
        cmd.add_argument(
            "--rebuild",
            action="store_true",
            help="Rewrite the canonical tables from --input instead of appending new substances/targets to the prepared dataset",
        )

    def add_shared_prepare_args(cmd, with_raw_input=True):
        cmd.add_argument("--dataset", required=True, help="Dataset name under the artifacts directory")
        if with_raw_input:
            add_raw_input_args(cmd)
        cmd.add_argument("--artifacts-dir", help="Root directory for canonical datasets and derived artifacts")
        cmd.add_argument("--kpgt-dir", help="Path to an external KPGT checkout")
        cmd.add_argument("--kpgt-model-path", help="Path to the pretrained KPGT model file")
        cmd.add_argument("--kpgt-python", help="Python executable to use for KPGT preprocessing and feature extraction")
        cmd.add_argument("--kpgt-preprocess-jobs", type=int, default=4, help="Worker count used inside each streaming KPGT chunk")
        cmd.add_argument("--kpgt-chunk-size", type=int, default=1024, help="Number of drugs held in memory during streaming KPGT feature extraction")
        cmd.add_argument("--kpgt-batch-size", type=int, default=32, help="KPGT model inference batch size")
        cmd.add_argument("--esm-model-name", default="esm2_t33_650M_UR50D")
        cmd.add_argument(
            "--esm-checkpoint-seconds",
            type=float,
            default=600.0,
            help="Rewrite prot_rep.pkl at least this often so an interrupted embedding run resumes; 0 disables checkpointing",
        )
        cmd.add_argument("--esmfold-chunk-size", type=int)
        cmd.add_argument(
            "--skip-esmfold",
            action="store_true",
            help="Skip ESMFold generation and keep only targets with existing, sequence-matching targets/esmfold/<Target_ID>.pdb files",
        )
        cmd.add_argument(
            "--skip-similarity",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Skip drug fingerprint and TM-score similarity generation; inference-only datasets do not need them",
        )
        cmd.add_argument("--target-similarity-processes", type=int, help="Worker process count for TM-score target similarity generation")
        cmd.add_argument("--target-similarity-chunksize", type=int, help="Multiprocessing chunksize for TM-score target similarity generation")
        cmd.add_argument(
            "--drop-failed-entities",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Record substances/targets that fail KPGT, ESMFold, or graph building in excluded.csv and continue instead of aborting",
        )
        cmd.add_argument("--force", action="store_true", help="Regenerate derived artifacts")

    prepare = subparsers.add_parser("prepare", help="Prepare a built-in or custom dataset", formatter_class=formatter)
    add_shared_prepare_args(prepare)

    train = subparsers.add_parser("train", help="Train using a prepared dataset", formatter_class=formatter)
    train.add_argument("--dataset", required=True, help="Training dataset name")
    train.add_argument("--test-dataset", help="Optional external evaluation dataset name")
    train.add_argument("--artifacts-dir", help="Root directory for canonical datasets and derived artifacts")
    train.add_argument("--output-dir", help="Directory where prediction Parquet files are saved")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--validation-size", type=float, default=0.1)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--early-stopping-patience", type=int, default=5, help="Stop after this many validation-F1 plateau epochs; 0 disables early stopping")
    train.add_argument("--output-name", help="Prediction filename prefix")

    infer = subparsers.add_parser("infer", help="Run a trained checkpoint on a prepared dataset", formatter_class=formatter)
    infer.add_argument("--dataset", required=True, help="Prepared dataset name")
    infer.add_argument("--checkpoint", required=True, help="Training checkpoint (.pt) containing model_state_dict")
    infer.add_argument("--artifacts-dir", help="Root directory containing the prepared dataset")
    infer.add_argument("--output-dir", help="Directory where the prediction Parquet file is saved")
    infer.add_argument("--output-name", help="Prediction filename prefix")
    infer.add_argument("--batch-size", type=int, default=64)

    run = subparsers.add_parser("run", help="Prepare then train in one command", formatter_class=formatter)
    add_shared_prepare_args(run)
    run.add_argument("--test-dataset", help="Optional external evaluation dataset name")
    run.add_argument("--test-input", help="Optional raw test dataset in CSV or Parquet")
    run.add_argument("--output-dir", help="Directory where prediction Parquet files are saved")
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--learning-rate", type=float, default=5e-5)
    run.add_argument("--weight-decay", type=float, default=1e-4)
    run.add_argument("--validation-size", type=float, default=0.1)
    run.add_argument("--random-state", type=int, default=42)
    run.add_argument("--early-stopping-patience", type=int, default=5, help="Stop after this many validation-F1 plateau epochs; 0 disables early stopping")
    run.add_argument("--output-name", help="Prediction filename prefix")

    batch_predict = subparsers.add_parser(
        "batch-predict",
        help="Prepare whatever is new, then score every substance x target pair into resumable Parquet chunks",
        formatter_class=formatter,
    )
    add_shared_prepare_args(batch_predict, with_raw_input=False)
    # Screening runs never touch the training-only similarity artifacts, and one
    # unparseable SMILES in a million-pair batch should not kill the job.
    batch_predict.set_defaults(skip_similarity=True, drop_failed_entities=True, rebuild=False)
    batch_predict.add_argument("--checkpoint", required=True, help="Training checkpoint (.pt) containing model_state_dict")
    batch_predict.add_argument("--substances-json", help="JSON array of SMILES strings")
    batch_predict.add_argument("--substances-parquet", help="Parquet table with a smiles column (and optionally substance_id)")
    batch_predict.add_argument("--targets-json", help="JSON array of protein sequences")
    batch_predict.add_argument("--targets-parquet", help="Parquet table with a sequence column (and optionally target_id)")
    batch_predict.add_argument("--pair-batch-size", type=int, default=64, help="Maximum cartesian pairs scored in one model forward pass")
    batch_predict.add_argument("--chunk-rows", type=int, default=20_000_000, help="Prediction rows per output Parquet chunk file")
    batch_predict.add_argument(
        "--probability-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for the probability column in output chunks",
    )
    batch_predict.add_argument("--graph-cache-size", type=int, default=256, help="Protein graphs held in memory; larger helps substance-major order")
    batch_predict.add_argument(
        "--pair-order",
        choices=("target-major", "substance-major"),
        default="target-major",
        help="Cartesian iteration order; target-major loads each protein graph once and is much faster",
    )
    batch_predict.add_argument("--output-dir", help="Directory for chunk Parquet files; defaults under the artifacts root")
    batch_predict.add_argument("--progress-file", help="Resumable JSON progress checkpoint; defaults to <output-dir>/progress.json")
    batch_predict.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume the matching progress file (default: true)",
    )
    batch_predict.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Assume every substance and target is already prepared and go straight to scoring",
    )

    repair = subparsers.add_parser(
        "repair-labels",
        help="Repair Y/Label in an existing canonical interaction table without regenerating derived artifacts",
        formatter_class=formatter,
    )
    repair.add_argument("--dataset", required=True)
    repair.add_argument("--input", required=True, help="Original raw dataset in CSV or Parquet")
    repair.add_argument("--smiles-column", default="smiles")
    repair.add_argument("--sequence-column", default="sequence")
    repair.add_argument("--activity-column", default="activity")
    repair.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold to derive Label from non-binary activity; exact 0/1 activity is preserved as Label",
    )
    repair.add_argument("--artifacts-dir", help="Root directory containing the already-prepared dataset")
    return parser


def _preparation_config_from_args(args) -> PreparationConfig:
    return PreparationConfig(
        artifacts_dir=args.artifacts_dir,
        threshold=getattr(args, "threshold", 0.0),
        force=args.force,
        kpgt_dir=args.kpgt_dir,
        kpgt_model_path=args.kpgt_model_path,
        kpgt_python=args.kpgt_python,
        kpgt_preprocess_jobs=args.kpgt_preprocess_jobs,
        kpgt_chunk_size=args.kpgt_chunk_size,
        kpgt_batch_size=args.kpgt_batch_size,
        esm_model_name=args.esm_model_name,
        esm_checkpoint_seconds=args.esm_checkpoint_seconds,
        esmfold_chunk_size=args.esmfold_chunk_size,
        skip_esmfold=args.skip_esmfold,
        skip_similarity=args.skip_similarity,
        target_similarity_processes=args.target_similarity_processes,
        target_similarity_chunksize=args.target_similarity_chunksize,
        rebuild=getattr(args, "rebuild", False),
        drop_failed_entities=args.drop_failed_entities,
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

    if args.command == "batch-predict":
        substances = _read_entity_values(
            args.substances_json, args.substances_parquet, "smiles", "substance_id", "substances"
        )
        targets = _read_entity_values(args.targets_json, args.targets_parquet, "sequence", "target_id", "targets")
        run_batch_predict(
            dataset_name=args.dataset,
            checkpoint_path=args.checkpoint,
            substances=substances,
            targets=targets,
            batch_config=BatchPredictConfig(
                pair_batch_size=args.pair_batch_size,
                chunk_rows=args.chunk_rows,
                probability_dtype=args.probability_dtype,
                graph_cache_size=args.graph_cache_size,
                target_major=args.pair_order == "target-major",
                resume=args.resume,
                output_dir=args.output_dir,
                progress_file=args.progress_file,
            ),
            preparation_config=_preparation_config_from_args(args),
            skip_prepare=args.skip_prepare,
        )
        return

    if args.command == "repair-labels":
        repair_interaction_labels(
            dataset_name=args.dataset,
            input_path=args.input,
            smiles_column=args.smiles_column,
            sequence_column=args.sequence_column,
            activity_column=args.activity_column,
            threshold=args.threshold,
            artifacts_dir=args.artifacts_dir,
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

    if args.command == "infer":
        run_inference(
            dataset_name=args.dataset,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
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
