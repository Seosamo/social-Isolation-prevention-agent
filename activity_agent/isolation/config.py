"""Configuration and storage isolated from the original activity task."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from activity_agent.common.config import Config
from activity_agent.common.storage import write_json

TASK_ID = "regional_isolation_suspicion_v1"
DEFAULTS = {
    "geo_level": "sido", "time_mode": "retrospective",
    "geography_file": "references/geography.csv",
    "block_mapping_file": "references/block_geography.csv",
    "sources_file": "references/structural_sources.csv",
    "train_end": "202510", "validation_month": "202511", "test_month": "202512",
    "suspect_decline": 0.2, "stable_change": 0.1,
    "structural_quantile": 0.75, "min_reference_regions": 5,
    "min_training_regions": 3, "holdout_districts": [],
    "region_codes": [], "mapping_mode": "verified",
    "method": "supervised",
}


def settings(config: Config) -> dict:
    result = DEFAULTS | config.values.get("isolation", {})
    if result["method"] not in ("supervised", "isolation_forest"):
        raise ValueError("isolation.method must be supervised or isolation_forest")
    if result["geo_level"] not in ("sido", "dong", "sigungu"):
        raise ValueError("isolation.geo_level must be sido, sigungu or dong")
    if result["mapping_mode"] not in ("verified", "crs_consensus"):
        raise ValueError("mapping_mode must be verified or crs_consensus")
    if result["mapping_mode"] == "crs_consensus" and result["geo_level"] != "sido":
        raise ValueError("CRS consensus mapping supports sido cohorts only")
    if not isinstance(result["region_codes"], list) or any(
            not isinstance(code, str) or not code.strip() for code in result["region_codes"]):
        raise ValueError("region_codes must be a list of nonempty codes")
    if len(set(result["region_codes"])) != len(result["region_codes"]):
        raise ValueError("region_codes must be unique")
    # Keep the old config name readable for existing district experiments.
    if "holdout_regions" in result:
        if result["holdout_districts"] and result["holdout_regions"] != result["holdout_districts"]:
            raise ValueError("Conflicting holdout_regions and legacy holdout_districts")
        result["holdout_districts"] = result["holdout_regions"]
    result["holdout_regions"] = result["holdout_districts"]
    if not isinstance(result["holdout_regions"], list) or any(
            not isinstance(code, str) or not code.strip() for code in result["holdout_regions"]):
        raise ValueError("holdout_regions must be a list of nonempty region codes")
    if result["time_mode"] not in ("retrospective", "as_of"):
        raise ValueError("isolation.time_mode must be retrospective or as_of")
    if not 0 <= result["stable_change"] < result["suspect_decline"] <= 1:
        raise ValueError("Invalid isolation decline thresholds")
    if not 0 < result["structural_quantile"] < 1:
        raise ValueError("structural_quantile must be between 0 and 1")
    if result["min_reference_regions"] < 2 or result["min_training_regions"] < 2:
        raise ValueError("At least two reference/training regions are required")
    if not result["train_end"] < result["validation_month"] < result["test_month"]:
        raise ValueError("Require isolation train_end < validation_month < test_month")
    for key in ("train_end", "validation_month", "test_month"):
        if not pd.Series([result[key]]).str.fullmatch(r"\d{4}(0[1-9]|1[0-2])").all():
            raise ValueError(f"Invalid isolation {key}")
    return result


def reference_path(config: Config, key: str) -> Path:
    path = Path(settings(config)[key])
    return path if path.is_absolute() else config.root / path


def destination(config: Config, name: str) -> Path:
    return config.output / "isolation" / name


def save(config: Config, name: str, frame: pd.DataFrame) -> Path:
    path = destination(config, name + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def load(config: Config, name: str) -> pd.DataFrame:
    path = destination(config, name + ".parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run isolation {name} prerequisites first")
    return pd.read_parquet(path)


def report_json(config: Config, name: str, value) -> None:
    write_json(destination(config, name + ".json"), value)
