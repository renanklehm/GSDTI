"""Small, streaming Neo4j writer shared by the bulk inference entrypoints."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable


ACTIVITY_PREDICTION_QUERY = """
UNWIND $rows AS row
MATCH (s:Substance {canonical_smiles: row.smiles})
MATCH (t:Target {sequence: row.sequence})
WHERE NOT EXISTS {
    MATCH (e:Evidence_Substance_Target)-[:HAS_SUBSTANCE]->(s)
    MATCH (e)-[:HAS_ACTIVITY]->(t)
}
MERGE (s)-[r:ACTIVITY_PREDICTION {model: row.model}]->(t)
SET r.probability = row.probability
"""


def read_json_string_list(path: str | Path, field_name: str) -> list[str]:
    """Read a JSON array of non-empty strings, retaining first-seen order."""
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} JSON must contain a non-empty array of strings.")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} JSON must contain only non-empty strings.")
        item = item.strip()
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_progress(path: str | Path, fingerprint: str, total_pairs: int, resume: bool) -> dict:
    path = Path(path).expanduser()
    if resume and path.exists():
        progress = json.loads(path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint or progress.get("total_pairs") != total_pairs:
            raise ValueError("Progress file belongs to different input lists or model; use --no-resume to restart.")
        return progress
    return {"fingerprint": fingerprint, "total_pairs": total_pairs, "next_pair_index": 0, "started_at": time.time()}


def save_progress(path: str | Path, progress: dict) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = time.time()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class Neo4jPredictionWriter:
    """Owns one Neo4j driver and writes only bounded UNWIND batches."""

    def __init__(self, config_path: str | Path) -> None:
        config = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
        missing = {"uri", "username", "password"}.difference(config)
        if missing:
            raise ValueError(f"Neo4j config is missing required keys: {', '.join(sorted(missing))}")
        try:
            from neo4j import GraphDatabase
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the neo4j package in this model environment to upload predictions.") from exc
        self.database = config.get("database")
        driver_kwargs = {key: config[key] for key in ("encrypted", "connection_timeout") if key in config}
        self.driver = GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"]), **driver_kwargs)
        self.driver.verify_connectivity()

    def write(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.driver.session(database=self.database) as session:
            session.run(ACTIVITY_PREDICTION_QUERY, rows=rows).consume()

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()
