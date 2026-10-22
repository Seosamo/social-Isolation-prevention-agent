"""Derive new features from matched source grids, keeping aggregation levels explicit."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from activity_agent.common.storage import fingerprint
from activity_agent.data.aggregation import previous_month
from activity_agent.data.ingest import AGE_COLUMNS, TIME_COLUMNS, DAY_COLUMNS
from activity_agent.isolation.anomaly import INPUTS
from activity_agent.isolation.config import reference_path, destination, settings

NEW = ["cumulative_decline", "decline_acceleration", "persistent_decline", "relative_decline", "share_change", "hourly_jsd", "weekly_jsd"]
GROUP = [*INPUTS[:2], *NEW[:5]]
REGIONAL = [*INPUTS[2:], "hourly_jsd", "weekly_jsd"]
FEATURE_SETS = {"F0": INPUTS, "F1": [*INPUTS, *NEW[:3]], "F2": [*INPUTS, *NEW[3:5]],
                "F3": [*INPUTS, *NEW[5:]], "F4": [*INPUTS, *NEW], "F5": GROUP, "F6": REGIONAL}
KEY = ["block_code", "sex", "age_band", "month"]
DEFINITIONS = {
    "cumulative_decline": "1 − 동일 관측 격자의 flow(t)/flow(t−2); 감소가 양수",
    "decline_acceleration": "(1−flow(t)/flow(t−1)) − (1−flow(t−1)/flow(t−2)); 양수면 감소 가속",
    "persistent_decline": "동일 격자의 두 연속 월간 감소율이 모두 양수이면 1 (기존 2개월 평균 비교와 구분)",
    "relative_decline": "집단 월간 감소율 − 같은 소지역의 성·연령 유동 합계 월간 감소율; 동일 age 원천에서만 비교",
    "share_change": "group_flow(t)/all_age_flow(t) − group_flow(t−1)/all_age_flow(t−1); 감소 방향은 음수",
    "hourly_jsd": "소지역의 24시간 분포(t)와 분포(t−1)의 base-2 Jensen-Shannon divergence, 0~1",
    "weekly_jsd": "소지역의 7요일 분포(t)와 분포(t−1)의 base-2 Jensen-Shannon divergence, 0~1",
}


def safe_decline(now, before):
    return np.where(np.asarray(before) > 0, 1 - np.asarray(now) / np.where(np.asarray(before) > 0, before, np.nan), np.nan)


def jsd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = (a.sum(axis=1) > 0) & (b.sum(axis=1) > 0) & np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    p = a / np.where(a.sum(axis=1, keepdims=True) > 0, a.sum(axis=1, keepdims=True), np.nan)
    q = b / np.where(b.sum(axis=1, keepdims=True) > 0, b.sum(axis=1, keepdims=True), np.nan)
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        result = .5 * (np.where(p > 0, p * np.log2(p / m), 0).sum(axis=1)
                        + np.where(q > 0, q * np.log2(q / m), 0).sum(axis=1))
    return np.where(valid, np.clip(result, 0, 1), np.nan)


def recompute(frame, baseline=True):
    """Pure function used by both real features and paired source-flow injections."""
    f = frame.copy()
    if baseline:
        f["current_flow"] = f.g0
        f["baseline_flow"] = (f.g1+f.g2)/2
        f["previous_baseline_flow"] = (f.p2+f.p3)/2
        f["activity_decline"] = safe_decline(f.g0, (f.g1 + f.g2) / 2)
        f["previous_activity_decline"] = safe_decline(f.p1, (f.p2 + f.p3) / 2)
        for prefix, metric in (("d", "daytime_decline"), ("w", "weekend_decline")):
            f[metric] = safe_decline(f[prefix + "0"], (f[prefix + "1"] + f[prefix + "2"]) / 2)
            f["previous_" + metric] = safe_decline(f[prefix + "p1"], (f[prefix + "p2"] + f[prefix + "p3"]) / 2)
    current = safe_decline(f.g0, f.g1)
    prior = safe_decline(f.g1, f.g2)
    f["cumulative_decline"] = safe_decline(f.g0, f.g2)
    f["decline_acceleration"] = current - prior
    f["persistent_decline"] = np.where(np.isfinite(current) & np.isfinite(prior), ((current > 0) & (prior > 0)).astype(float), np.nan)
    f["persistent20_mom"] = (current >= .2) & (prior >= .2)
    f["relative_decline"] = current - safe_decline(f.r0, f.r1)
    f["share_change"] = f.g0 / f.r0.where(f.r0 > 0) - f.g1 / f.r1.where(f.r1 > 0)
    for prefix, n, feature in (("h", 24, "hourly_jsd"), ("y", 7, "weekly_jsd")):
        f[feature] = jsd(f[[f"{prefix}0_{i}" for i in range(n)]], f[[f"{prefix}1_{i}" for i in range(n)]])
    return f


def _snapshot(frames, month, family, minimum):
    months = [month, previous_month(month), previous_month(month, 2)]
    if not all(m in frames for m in months):
        return None
    windows = [frames[m] for m in months]
    keys = ["block_code", "x", "y"]
    common = windows[0][keys]
    for other in windows[1:]:
        common = common.merge(other[keys], on=keys, validate="one_to_one")
    count = common.groupby("block_code").size()
    coverage = pd.concat([count / w.groupby("block_code").size() for w in windows], axis=1).min(axis=1)
    cols = {"age": AGE_COLUMNS, "time": TIME_COLUMNS, "wkdy": DAY_COLUMNS}[family]
    aggregates = [w.merge(common, on=keys, validate="one_to_one").groupby("block_code")[cols].sum() for w in windows]
    return aggregates, coverage.ge(minimum)


def build_features(config, run=None, *, base=None, months=None):
    """Read production artifacts only; do not call helpers that write production manifests."""
    if base is None:
        base = pd.read_parquet(destination(config, "anomaly/detections.parquet"))
    months = list(months) if months is not None else ["202510", "202511", "202512"]
    base = base.loc[base.month.isin(months)].copy()
    cfg = settings(config)
    mapping_path = reference_path(config, "block_mapping_file")
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    if cfg["mapping_mode"] == "crs_consensus":
        audit = json.loads(destination(config, "spatial_audit.json").read_text(encoding="utf-8"))
        if fingerprint(mapping_path) != audit.get("candidate_mapping_sha256"):
            raise ValueError("Candidate mapping no longer matches spatial audit")
        mapping = mapping.loc[mapping.assignment_status.eq("candidate_crs_agree")]
    else:
        mapping = mapping.loc[mapping.verified.str.lower().eq("true")]
    mapping = mapping.loc[mapping.geo_level.eq("sido") & mapping.geo_code.isin(cfg["region_codes"])]
    names = base[["geo_code", "province"]].drop_duplicates()
    manifest = json.loads((config.output / "clean_manifest.json").read_text(encoding="utf-8"))
    snapshots = {}
    sources = []
    for family in ("age", "time", "wkdy"):
        frames = {}
        snapshots[family] = {}
        for filename in sorted(n for n in manifest["skt_files"] if n.startswith(f"clean/skt_{family}_")):
            path = config.output / filename
            month = filename.rsplit("_", 1)[1].split(".")[0]
            raw = pd.read_parquet(path)
            targets = mapping.loc[mapping.reference_year.eq(month[:4]), "block_code"]
            frames[month] = raw.loc[raw.block_code.isin(targets)]
            snap = _snapshot(frames, month, family, config.values["minimum_coverage"])
            if snap:
                snapshots[family][month] = snap
            frames = {m: f for m, f in frames.items() if m >= previous_month(month, 2)}
            sources.append({"file": filename, "sha256": fingerprint(path), "rows": len(raw)})
    regional, grouped = [], []
    for month in months:
        prev = previous_month(month)
        if not all(month in snapshots[k] and prev in snapshots[k] for k in snapshots):
            continue
        a, av = snapshots["age"][month]
        ap, apv = snapshots["age"][prev]
        r = pd.DataFrame(index=a[0].index)
        r["new_quality"] = av.reindex(r.index).fillna(False) & apv.reindex(r.index).fillna(False)
        for j in range(3):
            r[f"r{j}"] = a[j].sum(axis=1)
            r[f"rp{j+1}"] = ap[j].sum(axis=1)
        for family, prefix, tag, cols in (("time", "h", "d", TIME_COLUMNS), ("wkdy", "y", "w", DAY_COLUMNS)):
            now, valid = snapshots[family][month]
            past, validp = snapshots[family][prev]
            r["new_quality"] &= valid.reindex(r.index).fillna(False) & validp.reindex(r.index).fillna(False)
            for j in range(2):
                for i, column in enumerate(cols):
                    r[f"{prefix}{j}_{i}"] = now[j][column].reindex(r.index)
            selected = TIME_COLUMNS[9:18] if family == "time" else DAY_COLUMNS[5:]
            for j in range(3):
                r[f"{tag}{j}"] = now[j][selected].sum(axis=1).reindex(r.index) / (1 if family == "time" else 2)
                r[f"{tag}p{j+1}"] = past[j][selected].sum(axis=1).reindex(r.index) / (1 if family == "time" else 2)
        r["month"] = month
        regional.append(r.reset_index())
        for column in AGE_COLUMNS:
            sex = "여성" if column.startswith("WMAN") else "남성"
            age = "60+" if column.endswith("60GU") else column.split("_")[-1].replace("G", "")
            g = pd.DataFrame(index=a[0].index)
            for j in range(3):
                g[f"g{j}"] = a[j][column]
                g[f"p{j+1}"] = ap[j][column].reindex(g.index)
            grouped.append(g.reset_index().assign(sex=sex, age_band=age, month=month))
    if not grouped:
        raise ValueError("No source-flow windows with sufficient history")
    source = pd.concat(grouped, ignore_index=True).merge(pd.concat(regional, ignore_index=True), on=["block_code", "month"], validate="many_to_one")
    source = source.merge(mapping[["block_code", "geo_code"]].drop_duplicates(), on="block_code", validate="many_to_one").merge(names, on="geo_code", validate="many_to_one")
    source = recompute(source)
    # Compare source reconstruction with the operational six features; never silently substitute them.
    matched = base[KEY + INPUTS].merge(source[KEY + INPUTS], on=KEY, suffixes=("_saved", "_raw"), validate="one_to_one")
    errors = {c: float((matched[c + "_saved"] - matched[c + "_raw"]).abs().max()) for c in INPUTS}
    if any(v > 1e-8 for v in errors.values() if np.isfinite(v)):
        raise ValueError(f"Source reconstruction differs from operating features: {errors}")
    quality_cols = [c for c in base if c not in source and c not in ("anomaly_raw", "anomaly_percentile", "candidate", "status", "engine_version")]
    source = source.merge(base[KEY + [c for c in quality_cols if c not in KEY]], on=KEY, how="left", validate="one_to_one")
    source["row_id"] = source[KEY].astype(str).agg("|".join, axis=1)
    if run is not None:
        source.to_parquet(run / "data/features.parquet", index=False)
    return source, {"sources": sources, "reconstruction_max_abs_error": errors,
                    "aggregation": "Age: block×sex×age×month; time/weekday: block×month. Region totals for relative/share use the same AGE family only.",
                    "units": "Provider aggregate floating flow counts, not unique persons. Cross-family absolute volumes are not equated."}


def regional_frame(group):
    r = group.sort_values(KEY).drop_duplicates(["block_code", "month"]).copy()
    r["sex"], r["age_band"] = "전체", "전체"
    for j in range(3):
        r[f"g{j}"] = r[f"r{j}"]
        r[f"p{j+1}"] = r[f"rp{j+1}"]
    r = recompute(r)
    r["baseline_flow"] = (r.r1 + r.r2) / 2
    r["previous_baseline_flow"] = (r.rp2 + r.rp3) / 2
    r["row_id"] = r[KEY].astype(str).agg("|".join, axis=1)
    return r
