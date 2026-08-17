"""Chunked, minimum-size Parquet output for cartesian-product batch prediction."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_strings(values) -> str:
    """Content hash of an ordered string sequence.

    batch-predict accepts entity lists from JSON or Parquet, so the resume
    fingerprint hashes the normalized values instead of the source file: the
    same substances rewritten into a different container must resume, and the
    same file with different contents must not.
    """
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def load_progress(path: Path, fingerprint: str, total_pairs: int, resume: bool) -> dict[str, Any]:
    path = Path(path).expanduser()
    if resume and path.exists():
        progress = json.loads(path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint or progress.get("total_pairs") != total_pairs:
            raise ValueError("Progress file belongs to a different input or model; use --no-resume to restart.")
        return progress
    return {
        "fingerprint": fingerprint,
        "total_pairs": total_pairs,
        "next_pair_index": 0,
        "next_chunk_index": 0,
        "started_at": time.time(),
    }


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = time.time()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def pair_stream(
    start_index: int,
    num_substances: int,
    num_targets: int,
    target_major: bool = True,
) -> Iterator[tuple[int, int]]:
    """Stream (substance_position, target_position) pairs in sorted cartesian order.

    Sorted, contiguous order is what lets write_prediction_chunk delta-encode
    the id columns down to a few bytes per row instead of 4+ bytes per row.

    GS-DTI defaults to target-major order (the inner loop walks substances)
    because the protein side of a pair is a serialized PyG graph on disk while
    the drug side is one row of an in-memory matrix. Target-major order loads
    each target graph once and reuses it for every substance; substance-major
    order re-loads every graph for every substance unless the whole graph set
    fits in the cache.
    """
    if target_major:
        start_target, start_substance = divmod(start_index, num_substances)
        for target_position in range(start_target, num_targets):
            substance_start = start_substance if target_position == start_target else 0
            for substance_position in range(substance_start, num_substances):
                yield substance_position, target_position
        return

    start_substance, start_target = divmod(start_index, num_targets)
    for substance_position in range(start_substance, num_substances):
        target_start = start_target if substance_position == start_substance else 0
        for target_position in range(target_start, num_targets):
            yield substance_position, target_position


def chunk_path(output_dir: Path, chunk_index: int, digits: int) -> Path:
    return Path(output_dir) / f"part-{chunk_index:0{digits}d}.parquet"


def write_prediction_chunk(
    path: Path,
    substance_ids: Any,
    target_ids: Any,
    predicted_labels: Any,
    probabilities: Any,
    probability_dtype: str = "float16",
) -> None:
    """Write one chunk of predictions with the smallest schema that still round-trips cleanly.

    Ids are int32 and written with DELTA_BINARY_PACKED (rows arrive in sorted
    cartesian order from pair_stream, so deltas are almost always +1). The
    label is int8 with dictionary encoding (2 distinct values). Probability
    is float16 by default; probability_inactive is never stored since it is
    always 1 - probability_active and adds nothing but bytes.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    prob_np_dtype = np.float32 if probability_dtype == "float32" else np.float16
    table = pa.table(
        {
            "substance_id": pa.array(np.asarray(substance_ids, dtype=np.int32)),
            "target_id": pa.array(np.asarray(target_ids, dtype=np.int32)),
            "predicted_label": pa.array(np.asarray(predicted_labels, dtype=np.int8)),
            "probability_active": pa.array(np.asarray(probabilities, dtype=prob_np_dtype)),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        table,
        temporary_path,
        compression="zstd",
        use_dictionary=["predicted_label"],
        data_page_version="2.0",
        column_encoding={"substance_id": "DELTA_BINARY_PACKED", "target_id": "DELTA_BINARY_PACKED"},
    )
    os.replace(temporary_path, path)


def write_entity_map(path: Path, ids, canonical_ids, values, id_column: str, canonical_column: str, value_column: str) -> None:
    """Persist the input id -> canonical GS-DTI id mapping next to the chunks.

    batch-predict output only stores integer ids, so without this file the
    chunks cannot be joined back to the dataset's Drug_ID / Target_ID rows.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            id_column: pa.array(np.asarray(ids, dtype=np.int32)),
            canonical_column: pa.array([str(value) for value in canonical_ids], type=pa.string()),
            value_column: pa.array([str(value) for value in values], type=pa.string()),
        }
    )
    pq.write_table(table, path, compression="zstd", use_dictionary=[canonical_column, value_column])
