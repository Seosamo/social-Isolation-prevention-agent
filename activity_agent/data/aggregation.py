"""Common-grid aggregation shared by both analysis tasks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from activity_agent.data.ingest import AGE_COLUMNS, DAY_COLUMNS, TIME_COLUMNS


GRID = ["block_code", "x", "y"]


GROUP = ["block_code", "sex", "age"]


def previous_month(month: str, offset: int = 1) -> str:
    return str(pd.Period(month, freq="M") - offset).replace("-", "")


def decline(current, baseline):
    """Undefined when baseline <= 0. Preserve negative declines (growth) in evidence."""
    return (baseline - current) / baseline.where(baseline > 0)


def entropy(counts) -> float:
    values = np.asarray(counts, dtype=float)
    total = values.sum()
    if total <= 0:
        return float("nan")
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def aggregate_skt(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    cols = {"age": AGE_COLUMNS, "time": TIME_COLUMNS, "wkdy": DAY_COLUMNS}[family]
    grouped = frame.groupby("block_code")[cols].sum()
    if family == "age":
        result = grouped.reset_index().melt(id_vars="block_code", var_name="band", value_name="flow")
        result["sex"] = np.where(result.band.str.startswith("WMAN"), "여성", "남성")
        result["age"] = result.band.str.extract(r"_(\d+)G", expand=False)
        result.loc[result.band.str.endswith("60GU"), "age"] = "60+"
        return result.drop(columns="band")
    if family == "time":
        out = pd.DataFrame({
            "flow": grouped.sum(axis=1),
            "daytime": grouped[[f"TMST_{h:02d}" for h in range(9, 18)]].sum(axis=1),
            "evening": grouped[[f"TMST_{h:02d}" for h in range(18, 22)]].sum(axis=1),
            "morning": grouped[[f"TMST_{h:02d}" for h in range(6, 9)]].sum(axis=1),
            "night": grouped[[f"TMST_{h:02d}" for h in [22, 23, 0, 1, 2, 3, 4, 5]]].sum(axis=1),
        })
        out["night_ratio"] = out.night / out.flow.where(out.flow > 0)
    else:
        out = pd.DataFrame({"weekday": grouped[DAY_COLUMNS[:5]].mean(axis=1),
                            "weekend": grouped[DAY_COLUMNS[5:]].mean(axis=1)})
        out["flow"] = (out.weekday * 5 + out.weekend * 2) / 7
        out["weekend_ratio"] = out.weekend / out.weekday.where(out.weekday > 0)
    return out.reset_index().assign(sex="전체", age="전체")


def compare_skt_window(frames: dict[str, pd.DataFrame], month: str, family: str, minimum_coverage: float) -> pd.DataFrame:
    current = frames[month]
    full = aggregate_skt(current, family).set_index(GROUP).add_prefix("all_")
    full["x"] = current.groupby("block_code").x.mean().reindex(full.index.get_level_values(0)).to_numpy()
    full["y"] = current.groupby("block_code").y.mean().reindex(full.index.get_level_values(0)).to_numpy()
    past = [previous_month(month, i) for i in (1, 2)]
    full["quality"] = "insufficient_history"
    full["coverage"] = np.nan
    if not all(p in frames for p in past):
        return full.reset_index().assign(month=month)
    windows = [current, *(frames[p] for p in past)]
    grid = GRID + (["original_block"] if "original_block" in current else [])
    common = current[grid]
    for prior in windows[1:]:
        common = common.merge(prior[grid], on=grid, validate="one_to_one")
    restricted = [f.merge(common, on=grid, validate="one_to_one") for f in windows]
    aggregates = [aggregate_skt(f, family).set_index(GROUP) for f in restricted]
    baseline = (aggregates[1] + aggregates[2]) / 2
    common_now = aggregates[0].add_prefix("current_")
    common_base = baseline.add_prefix("baseline_")
    full_base = (aggregate_skt(windows[1], family).set_index(GROUP) + aggregate_skt(windows[2], family).set_index(GROUP)) / 2
    out = full.join(common_now).join(common_base).join(full_base.add_prefix("all_baseline_"))
    counts = common.groupby("block_code").size()
    coverages = pd.concat([counts / f.groupby("block_code").size() for f in windows], axis=1).min(axis=1)
    out["coverage"] = out.index.get_level_values(0).map(coverages).to_numpy()
    out["activity_decline"] = decline(out.current_flow, out.baseline_flow)
    out["all_decline"] = decline(out.all_flow, out.all_baseline_flow)
    if family == "time":
        out["daytime_decline"] = decline(out.current_daytime, out.baseline_daytime)
        out["night_share_change"] = out.current_night_ratio - out.baseline_night_ratio
    if family == "wkdy":
        out["weekend_decline"] = decline(out.current_weekend, out.baseline_weekend)
        out["weekend_ratio_change"] = out.current_weekend_ratio - out.baseline_weekend_ratio
    out["quality"] = np.select(
        [out.coverage.isna() | out.coverage.lt(minimum_coverage), out.baseline_flow.le(0) | out.baseline_flow.isna()],
        ["low_coverage", "zero_baseline"], default="ok")
    return out.reset_index().assign(month=month)
