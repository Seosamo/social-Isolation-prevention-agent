"""Construct geography-aligned cohorts, recomputing mobility from common grids."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from activity_agent.common.config import Config
from activity_agent.data.aggregation import compare_skt_window, previous_month
from activity_agent.isolation.config import TASK_ID, destination, load, reference_path, report_json, save, settings
from activity_agent.common.storage import digest
from activity_agent.isolation.structural import province_name, read_geography

COHORT_KEY = ["geo_level", "geo_code", "sex", "age_band", "month"]
BASE_COLUMNS = [*COHORT_KEY, "cohort_id", "feature_version", "task_id", "time_mode", "decision_at",
                "province", "district", "dong", "district_code", "current_flow", "baseline_flow", "coverage",
                "activity_decline", "previous_activity_decline", "quality", "previous_quality",
                "daytime_decline", "previous_daytime_decline", "day_quality", "previous_day_quality",
                "weekend_decline", "previous_weekend_decline", "weekend_quality", "previous_weekend_quality",
                "social_consumption_decline", "previous_social_consumption_decline", "consumption_quality", "previous_consumption_quality",
                "structure_value", "structure_profile", "structure_period", "structure_available_at", "structure_quality",
                "elderly_household_ratio_context", "elderly_poverty_share_context", "registered_households_context"]


def read_block_mapping(config: Config, geography: pd.DataFrame) -> pd.DataFrame:
    columns = ["block_code", "geo_level", "geo_code", "reference_year", "source_url", "verified"]
    path = reference_path(config, "block_mapping_file")
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    if not set(columns).issubset(frame):
        raise ValueError("Invalid block geography mapping columns")
    frame = frame.loc[frame.verified.str.lower().eq("true") & frame.source_url.str.startswith("https://")].copy()
    if frame.duplicated(["block_code", "geo_level", "reference_year"]).any():
        raise ValueError("Each block requires one verified target per level/year")
    check = frame.merge(geography[["geo_level", "geo_code", "reference_year"]], on=["geo_level", "geo_code", "reference_year"],
                        how="left", indicator=True, validate="many_to_one")
    if check._merge.ne("both").any():
        raise ValueError("Block mappings refer to codes absent from the verified geography registry")
    return frame[columns]


def aggregate_activity(config: Config, mappings: pd.DataFrame):
    cfg = settings(config)
    manifest_path = config.output / "clean_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run preprocess before building isolation cohorts")
    files = json.loads(manifest_path.read_text(encoding="utf-8"))["skt_files"]
    outputs, coverage = {}, []
    for family in ("age", "time", "wkdy"):
        cache, parts = {}, []
        for name in sorted(n for n in files if n.startswith(f"clean/skt_{family}_")):
            month = name.rsplit("_", 1)[1].split(".")[0]
            targets = mappings.loc[mappings.reference_year.eq(month[:4]) & mappings.geo_level.eq(cfg["geo_level"]), ["block_code", "geo_code"]]
            raw = pd.read_parquet(config.output / name)
            mapped = raw.merge(targets, on="block_code", validate="many_to_one")
            coverage.append({"family": family, "month": month, "input_grids": len(raw), "mapped_grids": len(mapped),
                             "mapped_fraction": len(mapped) / len(raw) if len(raw) else 0})
            if mapped.empty:
                continue
            mapped["original_block"] = mapped.block_code
            mapped["block_code"] = mapped.geo_code
            cache[month] = mapped
            cache = {m: f for m, f in cache.items() if m in (month, previous_month(month), previous_month(month, 2))}
            part = compare_skt_window(cache, month, family, config.values["minimum_coverage"])
            parts.append(part.rename(columns={"block_code": "geo_code", "age": "age_band"}))
        outputs[family] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return outputs, coverage


def add_previous(frame: pd.DataFrame, columns: list[str], groups: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    prior = frame[[*groups, "month", *columns]].copy()
    prior["month"] = prior.month.map(lambda m: previous_month(m, -1))
    prior = prior.rename(columns={col: "previous_" + col for col in columns})
    return frame.merge(prior, on=[*groups, "month"], how="left", validate="one_to_one")


def attach_consumption(config: Config, cohorts: pd.DataFrame, geography: pd.DataFrame) -> pd.DataFrame:
    path = config.output / "features.parquet"
    if not path.exists() or settings(config)["geo_level"] != "sigungu":
        return cohorts
    card = pd.read_parquet(path)
    card = card.loc[card.signal.eq("consumption")].copy()
    if card.empty:
        return cohorts
    names = card.region.str.split(n=1, expand=True)
    if names.shape[1] != 2:
        return cohorts
    card["province"] = names[0].map(province_name)
    card["district"] = names[1]
    card["reference_year"] = card.month.str[:4]
    registry = geography.loc[geography.geo_level.eq("sigungu"), ["province", "district", "reference_year", "geo_code"]]
    card = card.merge(registry, on=["province", "district", "reference_year"], validate="many_to_one")
    card = card.rename(columns={"age": "age_band", "quality": "consumption_quality", "previous_quality": "previous_consumption_quality"})
    key = ["geo_code", "sex", "age_band", "month"]
    cols = ["social_consumption_decline", "previous_social_consumption_decline", "consumption_quality", "previous_consumption_quality"]
    return cohorts.merge(card[[*key, *cols]], on=key, how="left", validate="one_to_one")


def period_end(value: str) -> pd.Timestamp:
    value = str(value).replace(".", "").replace("-", "")
    if len(value) == 4:
        return pd.Timestamp(f"{value}-12-31 23:59:59.999999999")
    return pd.Period(value[:6], freq="M").end_time


def attach_structure(cohorts: pd.DataFrame, structural: pd.DataFrame, time_mode: str) -> pd.DataFrame:
    index = {}
    for key, part in structural.loc[structural.geo_code.ne("") & structural.source_id.ne("lonely_deaths")].groupby(
            ["geo_level", "geo_code", "sex", "age_band", "metric"], sort=False):
        index[key] = part.sort_values("reference_period", ascending=False).to_dict("records")

    def select(row, sex, age, metric):
        options = index.get((row.geo_level, row.geo_code, sex, age, metric), [])
        decision = pd.Timestamp(row.decision_at)
        for option in options:
            if time_mode == "as_of":
                available = pd.to_datetime(option["available_at"], errors="coerce", utc=True)
                if pd.isna(available) or available.tz_localize(None) > decision or period_end(option["reference_period"]) > decision:
                    continue
            else:
                if option["reference_period"][:4] != row.month[:4]:
                    continue
                if len(option["reference_period"]) > 4 and period_end(option["reference_period"]) > decision:
                    continue
            return option
        return None

    attachments = []
    for row in cohorts.itertuples(index=False):
        structure = select(row, row.sex, row.age_band, "single_household_share")
        output = {"structure_value": np.nan, "structure_profile": "single_household_composition",
                  "structure_period": "", "structure_available_at": "", "structure_quality": "missing_matching_structure"}
        if structure:
            output.update(structure_value=structure["value"], structure_period=structure["reference_period"],
                          structure_available_at=structure["available_at"], structure_quality=structure["quality"])
        for metric, sex, age in [("elderly_household_ratio", "전체", "65+"),
                                 ("elderly_poverty_share", row.sex, "65+"), ("registered_households", "전체", "전체")]:
            # Elderly measures describe 65+, never exact measurements of SKT's 60+.
            item = select(row, sex, age, metric) if metric == "registered_households" or row.age_band == "60+" else None
            output[metric + "_context"] = item["value"] if item and item["quality"] == "ok" else np.nan
        attachments.append(output)
    return pd.concat([cohorts.reset_index(drop=True), pd.DataFrame(attachments)], axis=1)


def build_cohorts(config: Config) -> pd.DataFrame:
    cfg = settings(config)
    structural = load(config, "structural")
    geography = read_geography(config)
    mapping = read_block_mapping(config, geography)
    if mapping.loc[mapping.geo_level.eq(cfg["geo_level"])].empty:
        save(config, "cohorts", pd.DataFrame(columns=BASE_COLUMNS))
        report_json(config, "cohort_manifest", {"rows": 0, "time_mode": cfg["time_mode"], "geo_level": cfg["geo_level"],
                    "mapping_coverage": [], "warnings": ["No verified block mapping for the selected geography level"]})
        print("Cohorts: 0 rows; verified block mapping is required", flush=True)
        return pd.DataFrame(columns=BASE_COLUMNS)
    activity, coverage = aggregate_activity(config, mapping)
    cohorts = activity["age"]
    if cohorts.empty:
        save(config, "cohorts", pd.DataFrame(columns=BASE_COLUMNS))
        report_json(config, "cohort_manifest", {"rows": 0, "geo_level": cfg["geo_level"], "time_mode": cfg["time_mode"],
                    "mapping_coverage": coverage, "warnings": ["No SKT grids matched the verified block mapping"]})
        return pd.DataFrame(columns=BASE_COLUMNS)
    keep = ["geo_code", "sex", "age_band", "month", "current_flow", "baseline_flow", "coverage", "quality", "activity_decline"]
    for col in keep:
        if col not in cohorts:
            cohorts[col] = np.nan
    cohorts = add_previous(cohorts[keep], ["activity_decline", "quality"], ["geo_code", "sex", "age_band"])
    for family, metric, prefix in [("time", "daytime_decline", "day"), ("wkdy", "weekend_decline", "weekend")]:
        frame = activity[family]
        if frame.empty:
            continue
        if metric not in frame:
            frame[metric] = np.nan
        frame = add_previous(frame[["geo_code", "month", metric, "quality"]], [metric, "quality"], ["geo_code"])
        frame = frame.rename(columns={"quality": prefix + "_quality", "previous_quality": "previous_" + prefix + "_quality"})
        cohorts = cohorts.merge(frame, on=["geo_code", "month"], how="left", validate="many_to_one")
    cohorts["geo_level"] = cfg["geo_level"]
    cohorts["reference_year"] = cohorts.month.str[:4]
    meta = geography.loc[geography.geo_level.eq(cfg["geo_level"]), ["geo_level", "geo_code", "reference_year", *["province", "district", "dong"], "parent_code"]]
    cohorts = cohorts.merge(meta, on=["geo_level", "geo_code", "reference_year"], validate="many_to_one")
    cohorts["district_code"] = cohorts.geo_code if cfg["geo_level"] == "sigungu" else cohorts.parent_code
    if cfg["geo_level"] == "dong":
        district_rows = geography.loc[geography.geo_level.eq("sigungu")]
        district_keys = {(r.geo_code, r.reference_year): (r.province, r.district) for r in district_rows.itertuples()}
        if any(district_keys.get((r.parent_code, r.reference_year)) != (r.province, r.district) for r in cohorts.itertuples()):
            raise ValueError("Dong cohorts require verified parent district codes for group evaluation")
    cohorts["time_mode"] = cfg["time_mode"]
    cohorts["decision_at"] = cohorts.month.map(lambda m: pd.Period(m, freq="M").end_time.isoformat())
    cohorts = attach_consumption(config, cohorts, geography)
    cohorts = attach_structure(cohorts, structural, cfg["time_mode"])
    cohorts["task_id"] = TASK_ID
    cohorts["cohort_id"] = [digest(row) for row in cohorts[COHORT_KEY].itertuples(index=False, name=None)]
    # Death-statistic contents are intentionally excluded from model/label provenance.
    source_fields = ["source_id", "reference_period", "available_at", "source_hash", "source_url"]
    structural_signature = structural.loc[structural.source_id.ne("lonely_deaths"), source_fields].drop_duplicates().sort_values(source_fields).to_dict("records")
    manifest_path = config.output / "quality.json"
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    signature = {"sources": structural_signature, "activity_sources": [{k: f.get(k) for k in ("file", "sha256")} for f in raw_manifest.get("files", [])],
                 "geography": geography.to_dict("records"), "block_mapping": mapping.to_dict("records"),
                 "geo_level": cfg["geo_level"], "time_mode": cfg["time_mode"], "algorithm": "cohorts-v1"}
    for col in BASE_COLUMNS:
        if col not in cohorts:
            cohorts[col] = "missing" if "quality" in col else np.nan
    if cohorts.duplicated(COHORT_KEY).any():
        raise ValueError("Isolation cohort keys are not unique")
    cohorts = cohorts[BASE_COLUMNS].sort_values(COHORT_KEY).reset_index(drop=True)
    from activity_agent.common.storage import records
    cohorts["feature_version"] = digest({"provenance": signature,
                                          "data": records(cohorts.drop(columns="feature_version"))})
    save(config, "cohorts", cohorts)
    report_json(config, "cohort_manifest", {"rows": len(cohorts), "feature_version": cohorts.feature_version.unique().tolist(),
                "geo_level": cfg["geo_level"], "time_mode": cfg["time_mode"], "independent_regions": int(cohorts.geo_code.nunique()),
                "independent_districts": int(cohorts.district_code.nunique()), "mapping_coverage": coverage,
                "warnings": ["Retrospective annual statistics are not contemporaneous forecast inputs"] if cfg["time_mode"] == "retrospective" else []})
    print(f"Cohorts: {len(cohorts):,} rows", flush=True)
    return cohorts
