"""Versioned weak isolation labels with train-only structural reference fitting."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from activity_agent.common.config import Config
from activity_agent.isolation.config import TASK_ID, destination, load, report_json, save, settings
from activity_agent.common.storage import digest, records

LABEL_NAMES = {-1: "판단 보류", 0: "의심 근거 미약", 1: "고립 의심"}
REFERENCE_KEY = ["geo_level", "structure_profile", "sex", "age_band"]


def activity_state(row, current, previous, quality, previous_quality, cfg):
    if row.get(quality) != "ok" or row.get(previous_quality) != "ok":
        return "abstain", "invalid_or_missing_quality", False
    a, b = row.get(current), row.get(previous)
    if pd.isna(a) or pd.isna(b):
        return "abstain", "missing_comparison", False
    if a >= cfg["suspect_decline"] and b >= cfg["suspect_decline"]:
        return "support", "two_consecutive_declines", True
    if abs(a) <= cfg["stable_change"] and abs(b) <= cfg["stable_change"]:
        return "against", "two_stable_comparisons", True
    return "abstain", "intermediate_or_conflicting_change", True


class IsolationLabelPolicy:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.references = []
        self.fitted = False

    def fit_reference(self, train: pd.DataFrame):
        if self.cfg.get("region_codes"):
            train = train.loc[train.geo_code.isin(self.cfg["region_codes"])]
        if not train.empty and train.month.gt(self.cfg["train_end"]).any():
            raise ValueError("Structural references must be fitted on training months only")
        self.references = []
        clean = train.loc[train.structure_quality.eq("ok") & train.structure_value.notna()].copy()
        key = [*REFERENCE_KEY, "geo_code", "structure_period"]
        if not clean.empty:
            conflicts = clean.groupby(key).structure_value.nunique().gt(1)
            if conflicts.any():
                raise ValueError("A structural observation has conflicting repeated values")
            clean = clean.drop_duplicates(key)
        for group, frame in clean.groupby(REFERENCE_KEY, dropna=False, sort=True):
            count = frame.geo_code.nunique()
            if count < self.cfg["min_reference_regions"] or frame.structure_value.nunique() < 2:
                continue
            self.references.append({**dict(zip(REFERENCE_KEY, group)), "threshold": float(frame.structure_value.quantile(self.cfg["structural_quantile"])),
                                    "regions": int(count), "unique_observations": len(frame)})
        self.fitted = True
        return self

    @property
    def version(self):
        # Only semantic policy settings enter this identifier, never background deaths.
        keys = ["train_end", "geo_level", "time_mode", "suspect_decline", "stable_change", "structural_quantile", "min_reference_regions", "holdout_districts"]
        keys += [k for k in ("region_codes", "mapping_mode") if k in self.cfg]
        return "isolation-rule-v1:" + digest({"task": TASK_ID, "settings": {k: self.cfg[k] for k in keys}, "references": self.references})

    def to_dict(self):
        return {"task_id": TASK_ID, "settings": self.cfg, "references": self.references,
                "fitted": self.fitted, "label_version": self.version}

    @classmethod
    def from_dict(cls, data):
        if data["task_id"] != TASK_ID:
            raise ValueError("Wrong label policy task")
        policy = cls(data["settings"])
        policy.references, policy.fitted = data["references"], data["fitted"]
        if policy.version != data["label_version"]:
            raise ValueError("Label policy version mismatch")
        return policy

    def label(self, cohorts: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Fit a training-only reference before labeling")
        references = {tuple(r[k] for k in REFERENCE_KEY): r for r in self.references}
        decisions = []
        for row in cohorts.to_dict("records"):
            a, a_reason, _ = activity_state(row, "activity_decline", "previous_activity_decline", "quality", "previous_quality", self.cfg)
            ref = references.get(tuple(row[k] for k in REFERENCE_KEY))
            s, s_reason = "abstain", "missing_structure_or_reference"
            if row["structure_quality"] == "ok" and pd.notna(row["structure_value"]) and ref:
                s = "support" if row["structure_value"] >= ref["threshold"] else "against"
                s_reason = "above_train_quantile" if s == "support" else "below_train_quantile"
            b_states, observed = {}, []
            for name, current, previous, quality, past_quality in [
                ("daytime", "daytime_decline", "previous_daytime_decline", "day_quality", "previous_day_quality"),
                ("weekend", "weekend_decline", "previous_weekend_decline", "weekend_quality", "previous_weekend_quality"),
                ("social_consumption", "social_consumption_decline", "previous_social_consumption_decline", "consumption_quality", "previous_consumption_quality")]:
                state, reason, seen = activity_state(row, current, previous, quality, past_quality, self.cfg)
                b_states[name] = {"state": state, "reason": reason, "observed": seen}
                if seen:
                    observed.append(state)
            b = "support" if "support" in observed else "against" if observed and all(v == "against" for v in observed) else "abstain"
            label, reason = -1, "자료 부족·상충 또는 중간 범위"
            if a == s == b == "support":
                label, reason = 1, "집단 유동 위축·구조적 취약성·활동 보조 근거 일치"
            elif a == b == "against" and s != "abstain":
                label, reason = 0, "충분한 자료에서 활동 안정; 고립 부재를 확인한 것은 아님"
            decisions.append({"A_state": a, "A_reason": a_reason, "S_state": s, "S_reason": s_reason,
                              "S_threshold": ref["threshold"] if ref else np.nan, "B_state": b, "B_observed": len(observed),
                              "B_evidence": json.dumps(b_states, ensure_ascii=False), "label": label,
                              "label_name": LABEL_NAMES[label], "label_reason": reason})
        columns = ["A_state", "A_reason", "S_state", "S_reason", "S_threshold", "B_state", "B_observed", "B_evidence", "label", "label_name", "label_reason"]
        out = pd.concat([cohorts.reset_index(drop=True), pd.DataFrame(decisions, columns=columns)], axis=1)
        out["label_version"] = self.version
        out["label_source"] = "weak_isolation_rule"
        return out


def label_database(config: Config):
    config.output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.output / "reviews.sqlite3")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS label_sets (
        task_id TEXT NOT NULL,label_version TEXT NOT NULL,policy TEXT NOT NULL,created_at TEXT NOT NULL,
        PRIMARY KEY(task_id,label_version)
      );
      CREATE TABLE IF NOT EXISTS label_decisions (
        task_id TEXT NOT NULL,cohort_id TEXT NOT NULL,feature_version TEXT NOT NULL,label_version TEXT NOT NULL,
        label INTEGER NOT NULL CHECK(label IN(-1,0,1)),evidence TEXT NOT NULL,created_at TEXT NOT NULL,
        PRIMARY KEY(task_id,cohort_id,feature_version,label_version)
      );
      CREATE TABLE IF NOT EXISTS label_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT NOT NULL,cohort_id TEXT NOT NULL,
        feature_version TEXT NOT NULL,label_version TEXT NOT NULL,label INTEGER NOT NULL CHECK(label IN(-1,0,1)),
        reviewer TEXT NOT NULL,note TEXT NOT NULL,independent INTEGER NOT NULL,created_at TEXT NOT NULL
      );
    """)
    return db


def save_decisions(config: Config, labels: pd.DataFrame, policy: IsolationLabelPolicy):
    now = datetime.now(timezone.utc).isoformat()
    evidence = records(labels[["A_state", "A_reason", "S_state", "S_reason", "S_threshold", "B_state", "B_observed", "B_evidence", "label_reason"]])
    keys = labels[["cohort_id", "feature_version", "label_version", "label"]].itertuples(index=False, name=None)
    db = label_database(config)
    try:
        with db:
            db.execute("INSERT OR IGNORE INTO label_sets VALUES(?,?,?,?)", (TASK_ID, policy.version, json.dumps(policy.to_dict(), ensure_ascii=False), now))
            db.executemany("INSERT OR IGNORE INTO label_decisions VALUES(?,?,?,?,?,?,?)",
                           [(TASK_ID, cid, fv, lv, int(value), json.dumps(ev, ensure_ascii=False, allow_nan=False), now)
                            for (cid, fv, lv, value), ev in zip(keys, evidence)])
    finally:
        db.close()


def label_cohorts(config: Config) -> pd.DataFrame:
    cfg = settings(config)
    cohorts = load(config, "cohorts")
    train = cohorts.loc[cohorts.month.le(cfg["train_end"]) & ~cohorts.district_code.isin(cfg["holdout_districts"])].copy()
    policy = IsolationLabelPolicy(cfg).fit_reference(train)
    labels = policy.label(cohorts)
    save(config, "labels", labels)
    report_json(config, "label_reference", policy.to_dict())
    report_json(config, "label_summary", {"task_id": TASK_ID, "label_version": policy.version,
                "rows": len(labels), "label_counts": {str(k): int(v) for k, v in labels.label.value_counts().items()},
                "label_coverage": float(labels.label.ge(0).mean()) if len(labels) else 0,
                "reference_groups": len(policy.references), "note": "Weak labels, not independently verified isolation outcomes"})
    save_decisions(config, labels, policy)
    print(f"Isolation labels: {len(labels):,} rows; {int(labels.label.ge(0).sum()):,} non-abstaining", flush=True)
    return labels


def review_isolation(config: Config, cohort_id: str, value: int, reviewer: str, note: str, independent=False):
    if value not in (-1, 0, 1) or not reviewer.strip() or not note.strip():
        raise ValueError("A valid label, reviewer and evidence note are required")
    rows = load(config, "labels")
    rows = rows.loc[rows.cohort_id.eq(cohort_id)]
    if len(rows) != 1:
        raise ValueError("Unknown or ambiguous isolation cohort_id")
    row = rows.iloc[0]
    db = label_database(config)
    try:
        with db:
            db.execute("INSERT INTO label_reviews(task_id,cohort_id,feature_version,label_version,label,reviewer,note,independent,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                       (TASK_ID, cohort_id, row.feature_version, row.label_version, value, reviewer.strip(), note.strip(), int(independent), datetime.now(timezone.utc).isoformat()))
    finally:
        db.close()


def isolation_reviews(config: Config, independent_only=False) -> pd.DataFrame:
    db = label_database(config)
    try:
        frame = pd.read_sql_query("""SELECT r.* FROM label_reviews r JOIN (
          SELECT MAX(id) id FROM label_reviews WHERE task_id=?
          GROUP BY cohort_id,feature_version,label_version) latest ON r.id=latest.id""", db, params=[TASK_ID])
    finally:
        db.close()
    return frame.loc[frame.independent.eq(1)].copy() if independent_only else frame
