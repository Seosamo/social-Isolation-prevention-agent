"""Shared artifact I/O and provenance helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from activity_agent.common.config import Config


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_table(config: Config, name: str, frame) -> Path:
    path = config.output / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def load_table(config: Config, name: str):
    path = config.output / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the preceding pipeline stage first.")
    return pd.read_parquet(path)


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:20]


def records(frame: pd.DataFrame) -> list[dict]:
    """JSON-safe records, including numpy values and NaNs."""
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))
