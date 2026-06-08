from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    root: Path
    interaction_table: Path
    drugs_table: Path
    targets_table: Path
    drug_features: Path
    graph_dir: Path
    esmfold_dir: Path
    protein_features: Path
    drug_similarity: Path
    target_similarity: Path


def _resolve_existing_dataset_dir(dataset_name: str, artifacts_root: Path) -> Path:
    candidates = [
        artifacts_root / dataset_name,
        artifacts_root / dataset_name.lower(),
        artifacts_root / dataset_name.upper(),
        artifacts_root / dataset_name.capitalize(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return artifacts_root / dataset_name


def resolve_artifacts_root(artifacts_dir: str | Path | None = None) -> Path:
    if artifacts_dir is None:
        return DATA_ROOT
    return Path(artifacts_dir).expanduser().resolve()


def get_dataset_paths(dataset_name: str, artifacts_dir: str | Path | None = None) -> DatasetPaths:
    artifacts_root = resolve_artifacts_root(artifacts_dir)
    root = _resolve_existing_dataset_dir(dataset_name, artifacts_root)
    drugs_dir = root / "drugs"
    targets_dir = root / "targets"
    return DatasetPaths(
        name=dataset_name,
        root=root,
        interaction_table=root / "df_less1000.csv",
        drugs_table=drugs_dir / "drugs.csv",
        targets_table=targets_dir / "targets.csv",
        drug_features=drugs_dir / "kpgt_base.npz",
        graph_dir=targets_dir / "graph",
        esmfold_dir=targets_dir / "esmfold",
        protein_features=targets_dir / "prot_rep.pkl",
        drug_similarity=drugs_dir / "drug_simmatrix.npz",
        target_similarity=targets_dir / "target_simmatrix.npz",
    )


def ensure_dataset_dirs(paths: DatasetPaths) -> None:
    for directory in [
        paths.root,
        paths.drugs_table.parent,
        paths.targets_table.parent,
        paths.graph_dir,
        paths.esmfold_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def read_table(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path}")


def normalize_interaction_dataframe(
    df,
    smiles_column: str,
    sequence_column: str,
    activity_column: str,
    threshold: float = 0.0,
    dataset_prefix: str = "custom",
) -> "pd.DataFrame":
    required = {smiles_column, sequence_column, activity_column}
    missing = required.difference(df.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_columns}")

    canonical = df[[smiles_column, sequence_column, activity_column]].copy()
    canonical.columns = ["Drug", "Target", "Y"]
    canonical = canonical.dropna(subset=["Drug", "Target", "Y"]).reset_index(drop=True)
    canonical["Drug"] = canonical["Drug"].astype(str)
    canonical["Target"] = canonical["Target"].astype(str)
    canonical["Y"] = canonical["Y"].astype(float)

    drug_ids = {smiles: f"{dataset_prefix}_D{i + 1}" for i, smiles in enumerate(canonical["Drug"].drop_duplicates())}
    target_ids = {seq: f"{dataset_prefix}_T{i + 1}" for i, seq in enumerate(canonical["Target"].drop_duplicates())}

    canonical.insert(0, "Drug_ID", canonical["Drug"].map(drug_ids))
    canonical.insert(2, "Target_ID", canonical["Target"].map(target_ids))
    canonical["Label"] = (canonical["Y"] >= threshold).astype(int)
    canonical["Target_Length"] = canonical["Target"].str.len()
    return canonical[["Drug_ID", "Drug", "Target_ID", "Target", "Y", "Label", "Target_Length"]]


def build_drugs_table(interactions):
    drugs = interactions[["Drug_ID", "Drug"]].drop_duplicates().copy()
    drugs.columns = ["Drug_ID", "smiles"]
    return drugs.reset_index(drop=True)


def build_targets_table(interactions):
    targets = interactions[["Target_ID", "Target"]].drop_duplicates().copy()
    return targets.reset_index(drop=True)


def load_feature_array(npz_path: Path):
    import numpy as np

    data = np.load(npz_path)
    if "fps" in data.files:
        return data["fps"]
    return data[data.files[0]]


def save_feature_array(npz_path: Path, array) -> None:
    import numpy as np

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, fps=array)


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
