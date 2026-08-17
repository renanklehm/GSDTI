"""Build minimum-size substance/target lookup tables for batch prediction.

Converts flat JSON string arrays (SMILES, protein sequences) into normalized
Parquet tables keyed by a generated sequential integer id. Downstream batch
prediction (main.py batch-predict) reads these ids directly instead of the
raw strings, which is what keeps the cartesian-product prediction output
small: see scripts/batch_predict_utils.py for the chunk writer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_config import REPO_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build substances.parquet / targets.parquet from JSON string arrays.")
    parser.add_argument(
        "--substances-json",
        type=Path,
        default=REPO_ROOT / "substances.json",
        help="JSON array of canonical SMILES strings.",
    )
    parser.add_argument(
        "--targets-json",
        type=Path,
        default=REPO_ROOT / "targets.json",
        help="JSON array of target protein sequences.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "inference",
        help="Directory to write substances.parquet and targets.parquet into.",
    )
    return parser.parse_args(argv)


def read_json_string_array(path: Path, field_name: str) -> list[str]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} JSON must contain a non-empty array of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} JSON must contain only non-empty strings.")
        result.append(item.strip())
    return result


def write_entity_table(values: list[str], id_column: str, value_column: str, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            id_column: pa.array(range(len(values)), type=pa.int32()),
            value_column: pa.array(values, type=pa.string()),
        }
    )
    pq.write_table(table, path, compression="zstd", use_dictionary=[value_column])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    substances = read_json_string_array(args.substances_json, "substances")
    targets = read_json_string_array(args.targets_json, "targets")

    substances_path = args.output_dir / "substances.parquet"
    targets_path = args.output_dir / "targets.parquet"
    write_entity_table(substances, "substance_id", "smiles", substances_path)
    write_entity_table(targets, "target_id", "sequence", targets_path)

    total_pairs = len(substances) * len(targets)
    print(f"Wrote {len(substances):,} substances to {substances_path} ({substances_path.stat().st_size:,} bytes)")
    print(f"Wrote {len(targets):,} targets to {targets_path} ({targets_path.stat().st_size:,} bytes)")
    print(f"Cartesian product for batch-predict: {total_pairs:,} pairs")


if __name__ == "__main__":
    main()
