from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from activity_agent.common.config import Config
from activity_agent.common.storage import load_table, save_table
from activity_agent.activity.detection import weak_labels


def connect(config: Config) -> sqlite3.Connection:
    config.output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.output / "reviews.sqlite3")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS weak_labels (
        feature_id TEXT NOT NULL, feature_version TEXT NOT NULL, label_version TEXT NOT NULL,
        label INTEGER NOT NULL CHECK(label IN (-1,0,1)), reason TEXT NOT NULL,
        evidence TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY(feature_id, feature_version, label_version)
      );
      CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT, feature_id TEXT NOT NULL,
        feature_version TEXT NOT NULL, label INTEGER NOT NULL CHECK(label IN (-1,0,1)),
        reviewer TEXT NOT NULL, note TEXT NOT NULL,
        independent INTEGER NOT NULL, created_at TEXT NOT NULL
      );
    """)
    return db


def label(config: Config) -> pd.DataFrame:
    features = load_table(config, "features")
    labels = weak_labels(features, config)
    save_table(config, "weak_labels", labels)
    now = datetime.now(timezone.utc).isoformat()
    evidence_cols = [c for c in ["primary_decline", "previous_decline", "social_consumption_decline", "quality", "coverage"] if c in labels]
    records = []
    for row in labels.to_dict("records"):
        evidence = {c: None if pd.isna(row[c]) else row[c] for c in evidence_cols}
        records.append((row["feature_id"], row["feature_version"], row["label_version"], int(row["label"]),
                        row["label_reason"], json.dumps(evidence, ensure_ascii=False, allow_nan=False), now))
    with connect(config) as db:
        db.executemany("INSERT OR IGNORE INTO weak_labels VALUES (?,?,?,?,?,?,?)", records)
    save_table(config, "combined_labels", combine_labels(labels))
    print(f"Labels: {labels.label_name.value_counts().to_dict()}", flush=True)
    return labels


def combine_labels(labels: pd.DataFrame) -> pd.DataFrame:
    key = ["month", "region", "sex", "age"]
    a = labels.loc[labels.signal.eq("district_mobility"), [*key, "label"]]
    b = labels.loc[labels.signal.eq("consumption"), [*key, "label"]]
    out = a.merge(b, on=key, validate="one_to_one", suffixes=("_mobility", "_consumption"))
    out["label"] = -1
    for value in (0, 1):
        out.loc[out.label_mobility.eq(value) & out.label_consumption.eq(value), "label"] = value
    return out


def add_review(config: Config, feature_id: str, value: int, reviewer: str, note: str, independent: bool = False) -> None:
    if value not in (-1, 0, 1) or not reviewer.strip() or not note.strip():
        raise ValueError("Valid label, reviewer and evidence note are required.")
    rows = load_table(config, "features")
    rows = rows.loc[rows.feature_id.eq(feature_id)]
    if len(rows) != 1:
        raise ValueError("Unknown or ambiguous feature_id.")
    if rows.iloc[0].signal not in ("mobility", "district_mobility", "consumption"):
        raise ValueError("Pattern context cannot be labeled for supervised training.")
    with connect(config) as db:
        db.execute("INSERT INTO reviews(feature_id,feature_version,label,reviewer,note,independent,created_at) VALUES (?,?,?,?,?,?,?)",
                   (feature_id, rows.iloc[0].feature_version, value, reviewer.strip(), note.strip(), int(independent), datetime.now(timezone.utc).isoformat()))


def latest_reviews(config: Config, independent_only: bool = False) -> pd.DataFrame:
    with connect(config) as db:
        # Pick the latest decision first; a newer non-independent review must
        # invalidate an older independent decision for the same feature version.
        rows = pd.read_sql_query("""
          SELECT r.* FROM reviews r JOIN (
            SELECT feature_id,feature_version,MAX(id) AS id FROM reviews
            GROUP BY feature_id,feature_version
          ) last ON r.id=last.id
        """, db)
    if independent_only:
        rows = rows.loc[rows.independent.eq(1)].copy()
    return rows
