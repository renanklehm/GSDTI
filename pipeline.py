import argparse
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    validation_size: float = 0.1
    random_state: int = 42


@dataclass
class PreparationConfig:
    artifacts_dir: str | None = None
    threshold: float = 0.0
    force: bool = False
    kpgt_dir: str | None = None
    kpgt_model_path: str | None = None
    kpgt_python: str | None = None
    esm_model_name: str = "esm2_t33_650M_UR50D"
    esmfold_chunk_size: int | None = None


def _tqdm(iterable=None, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, **kwargs)


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
        return

    if not config.kpgt_dir:
        raise ValueError("Missing --kpgt-dir. True KPGT feature generation requires an external KPGT checkout.")
    if not config.kpgt_model_path:
        raise ValueError("Missing --kpgt-model-path. True KPGT feature generation requires the pretrained model path.")

    kpgt_dir = Path(config.kpgt_dir).expanduser().resolve()
    model_path = Path(config.kpgt_model_path).expanduser().resolve()
    preprocess_script = _resolve_kpgt_script(kpgt_dir, "preprocess_downstream_dataset.py")
    extract_script = _resolve_kpgt_script(kpgt_dir, "extract_features.py")
    _require_file(model_path, "KPGT pretrained model")
    _patch_kpgt_compatibility(kpgt_dir)
    kpgt_python = Path(config.kpgt_python).expanduser().resolve() if config.kpgt_python else Path(sys.executable)
    _require_file(kpgt_python, "KPGT python executable")

    dataset_name = paths.root.name
    datasets_root = kpgt_dir / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    kpgt_dataset_dir = datasets_root / dataset_name
    kpgt_dataset_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    shutil.copy2(paths.drugs_table, kpgt_dataset_dir / f"{dataset_name}.csv")
    _run_subprocess(
        [str(kpgt_python), str(preprocess_script), "--data_path", str(datasets_root), "--dataset", dataset_name],
        kpgt_dir,
        "preprocess KPGT dataset",
        extra_pythonpath=kpgt_dir,
    )
    _run_subprocess(
        [
            str(kpgt_python),
            str(extract_script),
            "--config",
            "base",
            "--model_path",
            str(model_path),
            "--data_path",
            str(datasets_root),
            "--dataset",
            dataset_name,
        ],
        kpgt_dir,
        "extract KPGT features",
        extra_pythonpath=kpgt_dir,
    )

    candidates = [
        kpgt_dataset_dir / "kpgt_base.npz",
        kpgt_dir / "datasets" / "bind_drugs" / "kpgt_base.npz",
        kpgt_dir / "datasets" / dataset_name / "kpgt_base.npz",
    ]
    generated = next((candidate for candidate in candidates if candidate.exists()), None)
    if generated is None:
        raise FileNotFoundError("KPGT finished without producing kpgt_base.npz in an expected location.")

    paths.drug_features.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated, paths.drug_features)


def _generate_protein_features(paths: DatasetPaths, config: PreparationConfig) -> None:
    if paths.protein_features.exists() and not config.force:
        return

    import pandas as pd
    import torch

    targets_df = pd.read_csv(paths.targets_table)
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
    for row in _tqdm(targets_df.itertuples(index=False), total=len(targets_df), desc="ESM-2 embeddings", unit="target", leave=False):
        batch_labels, batch_strs, batch_tokens = batch_converter([(row.Target_ID, row.Target)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        token_representations.append(results["representations"][repr_layer].cpu().detach().numpy())

    paths.protein_features.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.protein_features, "wb") as handle:
        pickle.dump(token_representations, handle)


def _generate_esmfold_structures(paths: DatasetPaths, config: PreparationConfig) -> None:
    import pandas as pd
    import torch

    targets_df = pd.read_csv(paths.targets_table)
    missing_targets = []
    for row in targets_df.itertuples(index=False):
        pdb_path = paths.esmfold_dir / f"{row.Target_ID}.pdb"
        if config.force or not pdb_path.exists():
            missing_targets.append(row)

    if not missing_targets:
        return

    try:
        esm = __import__("esm")
    except ModuleNotFoundError as exc:
        raise RuntimeError("ESM is required to generate true ESMFold structures.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    if config.esmfold_chunk_size is not None:
        model.set_chunk_size(config.esmfold_chunk_size)

    paths.esmfold_dir.mkdir(parents=True, exist_ok=True)
    for row in _tqdm(missing_targets, total=len(missing_targets), desc="ESMFold PDBs", unit="target", leave=False):
        pdb_path = paths.esmfold_dir / f"{row.Target_ID}.pdb"
        with torch.no_grad():
            output = model.infer_pdb(row.Target)
        with open(pdb_path, "w", encoding="utf-8") as handle:
            handle.write(output)


def _generate_graphs(paths: DatasetPaths, config: PreparationConfig) -> None:
    import numpy as np
    import torch

    from utils import load_predicted_PDB3, protein_graph

    targets_df, protein_features = _load_targets_for_graph_build(paths)
    paths.graph_dir.mkdir(parents=True, exist_ok=True)

    graph_inputs = zip(targets_df.iterrows(), protein_features)
    for (_, row), embedding in _tqdm(graph_inputs, total=len(targets_df), desc="Protein graphs", unit="target", leave=False):
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
        compute_and_save_tanimoto_matrix(drugs_df, paths.drug_similarity)

    if config.force or not paths.target_similarity.exists():
        missing_pdbs = [target_id for target_id in targets_df["Target_ID"] if not (paths.esmfold_dir / f"{target_id}.pdb").exists()]
        if missing_pdbs:
            missing_preview = ", ".join(map(str, missing_pdbs[:5]))
            raise FileNotFoundError(f"Missing ESMFold PDB files for TM-score matrix generation: {missing_preview}")
        compute_and_save_tm_score_matrix_parallel_optimized(targets_df, paths.esmfold_dir, paths.target_similarity)


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
        if progress is not None:
            progress.set_postfix(stage="starting")
        for stage_name, stage_fn in stages:
            if progress is not None:
                progress.set_postfix(stage=stage_name)
            stage_fn(paths, config)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return paths


def _load_training_assets(paths: DatasetPaths):
    import pandas as pd

    drug_features = load_feature_array(paths.drug_features)
    drug_df = pd.read_csv(paths.drugs_table)
    target_df = pd.read_csv(paths.targets_table)
    drug_dict = dict(zip(drug_df["Drug_ID"], drug_features))
    tanimoto_matrix = load_feature_array(paths.drug_similarity)
    tm_score_matrix = load_feature_array(paths.target_similarity)
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

    train_drug_feature, train_drug_df, train_target_df, train_drug_dict, tanimoto_matrix, tm_score_matrix = _load_training_assets(train_paths)
    train_df = pd.read_csv(train_paths.interaction_table)
    stratify_labels = train_df["Label"] if train_df["Label"].nunique() > 1 else None
    train_df, validation_df = train_test_split(
        train_df,
        test_size=config.validation_size,
        stratify=stratify_labels,
        random_state=config.random_state,
    )

    if test_paths is None:
        test_df = validation_df.copy()
        test_drug_df = train_drug_df
        test_target_df = train_target_df
        test_drug_dict = train_drug_dict
        test_graph_dir = train_paths.graph_dir
    else:
        _, test_drug_df, test_target_df, test_drug_dict, _, _ = _load_training_assets(test_paths)
        test_df = pd.read_csv(test_paths.interaction_table)
        test_graph_dir = test_paths.graph_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = GraphDataset_withsim(train_df, train_drug_df, train_target_df, train_drug_dict, str(train_paths.graph_dir))
    valid_dataset = GraphDataset_withsim(validation_df, train_drug_df, train_target_df, train_drug_dict, str(train_paths.graph_dir))
    test_dataset = GraphDataset_withsim(test_df, test_drug_df, test_target_df, test_drug_dict, str(test_graph_dir))

    model = GraphDTI_bi(train_drug_feature[0].shape[0], 1280, 2, surface_feature=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    weights = torch.tensor([1.0, 1.0], device=device)
    focal_criterion = FocalLoss(alpha=1, gamma=2, weight=weights)
    drug_contrastive_criterion = NTXentContrastiveLoss(temperature=0.07, sim_threshold=0.8)
    target_contrastive_criterion = NTXentContrastiveLoss(temperature=0.07, sim_threshold=0.5)

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

    best_f1 = -1.0
    best_predictions = None
    best_labels = None
    best_probs = None
    for epoch in range(config.epochs):
        train_cl(
            model,
            train_loader,
            optimizer,
            drug_contrastive_criterion,
            target_contrastive_criterion,
            focal_criterion,
            device,
            epoch,
        )
        _, _, _, _, val_f1, val_preds, val_labels, val_probs = evaluate_cl(model, val_loader, focal_criterion, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_predictions = val_preds.copy()
            best_labels = val_labels.copy()
            best_probs = val_probs.copy()

    if best_predictions is None or best_labels is None or best_probs is None:
        raise RuntimeError("Training did not produce validation predictions.")

    _, test_acc, _, _, test_f1, test_preds, test_labels, test_probs = evaluate_cl(model, test_loader, focal_criterion, device)

    validation_predictions = validation_df[["Drug_ID", "Target_ID", "Drug", "Target", "Y"]].copy()
    validation_predictions["predicted_label"] = best_predictions
    test_predictions = test_df[["Drug_ID", "Target_ID", "Drug", "Target", "Y"]].copy()
    test_predictions["predicted_label"] = test_preds

    output_root = Path(output_dir).expanduser().resolve() if output_dir else RESULTS_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{output_name or train_dataset_name}_predictions.csv"

    output_df = pd.concat(
        [
            _merge_predictions(validation_df, validation_predictions),
            _merge_predictions(test_df, test_predictions),
        ],
        ignore_index=True,
    )[["sequence", "smiles", "label_true", "label_predicted"]]
    output_df.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")
    print(f"Validation F1: {best_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}, Test Accuracy: {test_acc:.2f}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified GS-DTI data preparation and training entrypoint.")
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
        cmd.add_argument("--esm-model-name", default="esm2_t33_650M_UR50D")
        cmd.add_argument("--esmfold-chunk-size", type=int)
        cmd.add_argument("--force", action="store_true", help="Regenerate derived artifacts")

    prepare = subparsers.add_parser("prepare", help="Prepare a built-in or custom dataset")
    add_shared_prepare_args(prepare)

    train = subparsers.add_parser("train", help="Train using a prepared dataset")
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
    train.add_argument("--output-name", help="Prediction filename prefix")

    run = subparsers.add_parser("run", help="Prepare then train in one command")
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
        esm_model_name=args.esm_model_name,
        esmfold_chunk_size=args.esmfold_chunk_size,
    )


def main() -> None:
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
