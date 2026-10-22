"""Explore local contraction inside the selected provinces without changing labels."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from activity_agent.common.config import Config
from activity_agent.common.storage import fingerprint, records
from activity_agent.data.aggregation import compare_skt_window, previous_month
from activity_agent.isolation.cohorts import add_previous, read_block_mapping
from activity_agent.isolation.config import destination, load, report_json, save, settings
from activity_agent.isolation.structural import read_geography
from activity_agent.isolation.labeling import IsolationLabelPolicy

KEY = ["geo_code", "province", "sex", "age_band", "month"]


def local_features(config):
    cfg = settings(config)
    geography = read_geography(config)
    mapping = read_block_mapping(config, geography)
    mapping = mapping.loc[mapping.geo_level.eq("sido")]
    if mapping.empty:
        raise ValueError("No accepted province mapping for exploration")
    manifest = json.loads((config.output / "clean_manifest.json").read_text(encoding="utf-8"))
    families = {}
    for family, metric in (("age", "activity_decline"), ("time", "daytime_decline"), ("wkdy", "weekend_decline")):
        frames, parts = {}, []
        for filename in sorted(n for n in manifest["skt_files"] if n.startswith(f"clean/skt_{family}_")):
            month = filename.rsplit("_", 1)[1].split(".")[0]
            blocks = mapping.loc[mapping.reference_year.eq(month[:4]), "block_code"]
            raw = pd.read_parquet(config.output / filename)
            frames[month] = raw.loc[raw.block_code.isin(blocks)].copy()
            frames = {m: f for m, f in frames.items() if m in (month, previous_month(month), previous_month(month, 2))}
            part = compare_skt_window(frames, month, family, config.values["minimum_coverage"])
            for col in (metric, "current_flow", "baseline_flow"):
                if col not in part:
                    part[col] = np.nan
            parts.append(part[["block_code", "sex", "age", "month", metric, "quality", "coverage", "current_flow", "baseline_flow"]])
        if not parts:
            raise ValueError(f"Missing clean SKT {family} files")
        frame = pd.concat(parts, ignore_index=True).rename(columns={"age": "age_band"})
        frame = add_previous(frame, [metric, "quality", "baseline_flow"], ["block_code", "sex", "age_band"])
        families[family] = frame
    age = families["age"]
    for family, metric, prefix in (("time", "daytime_decline", "day"), ("wkdy", "weekend_decline", "weekend")):
        part = families[family][["block_code", "month", metric, "previous_" + metric, "quality", "previous_quality"]]
        part = part.rename(columns={"quality": prefix + "_quality", "previous_quality": "previous_" + prefix + "_quality"})
        age = age.merge(part, on=["block_code", "month"], how="left", validate="many_to_one")
    age["reference_year"] = age.month.str[:4]
    age = age.merge(mapping[["block_code", "reference_year", "geo_code"]], on=["block_code", "reference_year"], validate="many_to_one")
    age = age.merge(geography.loc[geography.geo_level.eq("sido"), ["geo_code", "reference_year", "province"]],
                    on=["geo_code", "reference_year"], validate="many_to_one")
    return age


def summarize_contraction(frame, threshold=0.2, stable_change=0.1):
    """Count blocks only over valid observations; missing denominators stay missing."""
    data = frame.copy()
    data["single_valid"] = data.quality.eq("ok") & data.activity_decline.notna() & data.baseline_flow.gt(0)
    data["persistent_valid"] = (data.single_valid & data.previous_quality.eq("ok")
                                & data.previous_activity_decline.notna() & data.previous_baseline_flow.gt(0))
    data["single_support"] = data.single_valid & data.activity_decline.ge(threshold)
    data["persistent_support"] = (data.persistent_valid & data.activity_decline.ge(threshold)
                                  & data.previous_activity_decline.ge(threshold))
    data["persistent_stable"] = (data.persistent_valid & data.activity_decline.abs().le(stable_change)
                                 & data.previous_activity_decline.abs().le(stable_change))
    observed, support, stable = [], [], []
    for metric, quality in (("daytime_decline", "day_quality"), ("weekend_decline", "weekend_quality")):
        valid = (data[quality].eq("ok") & data["previous_" + quality].eq("ok")
                 & data[metric].notna() & data["previous_" + metric].notna())
        observed.append(valid)
        support.append(valid & data[metric].ge(threshold) & data["previous_" + metric].ge(threshold))
        stable.append(valid & data[metric].abs().le(stable_change) & data["previous_" + metric].abs().le(stable_change))
    data["B_observed"] = observed[0] | observed[1]
    data["B_support"] = support[0] | support[1]
    data["B_stable"] = data.B_observed & (~observed[0] | stable[0]) & (~observed[1] | stable[1])
    data["joint_valid"] = data.persistent_valid & data.B_observed
    data["joint_support"] = data.persistent_support & data.B_support
    data["joint_stable"] = data.persistent_stable & data.B_stable
    rows = []
    for key, group in data.groupby(KEY, sort=True):
        result = dict(zip(KEY, key)) | {"decline_threshold": threshold, "observed_blocks": len(group)}
        for name, valid, signal in (("single", "single_valid", "single_support"),
                                    ("persistent", "persistent_valid", "persistent_support"),
                                    ("joint", "joint_valid", "joint_support")):
            sample = group.loc[group[valid]]
            positives = sample[signal]
            total_weight = sample.baseline_flow.sum()
            result.update({f"{name}_eligible": len(sample), f"{name}_blocks": int(positives.sum()),
                           f"{name}_share": float(positives.mean()) if len(sample) else None,
                           f"{name}_baseline_flow_share": float(sample.loc[positives, "baseline_flow"].sum() / total_weight)
                           if total_weight > 0 else None})
        result["stable_blocks"] = int(group.persistent_stable.sum())
        result["joint_stable_blocks"] = int(group.joint_stable.sum())
        sample = group.loc[group.persistent_valid]
        for q in (0.1, 0.5, 0.9):
            result[f"decline_p{int(q*100)}"] = float(sample.activity_decline.quantile(q)) if len(sample) else None
        result["baseline_le_1_blocks"] = int((sample.baseline_flow <= 1).sum())
        result["persistent_baseline_le_1_blocks"] = int((sample.persistent_support & sample.baseline_flow.le(1)).sum())
        rows.append(result)
    return data, pd.DataFrame(rows)


def explore_contraction(config: Config):
    cfg = settings(config)
    if cfg["geo_level"] != "sido":
        raise ValueError("Contraction exploration currently requires sido scope")
    policy = IsolationLabelPolicy.from_dict(json.loads(destination(config, "label_reference.json").read_text(encoding="utf-8")))
    semantic_keys = set(cfg) - {"method", "anomaly"}
    if any(policy.cfg.get(key) != cfg[key] for key in semantic_keys):
        raise ValueError("Label policy/settings changed; rebuild isolation labels before exploration")
    print("Computing contraction for mapped blocks across age/time/weekday files...", flush=True)
    features = local_features(config)
    signals, summary = summarize_contraction(features, cfg["suspect_decline"], cfg["stable_change"])
    save(config, "exploration/block_signals", signals)
    save(config, "exploration/contraction_shares", summary)
    out = destination(config, "exploration")
    summary.to_csv(out / "contraction_shares.csv", index=False, encoding="utf-8-sig")
    # Regional S describes province cohorts, never independently observed block outcomes.
    labels = load(config, "labels")
    if labels.duplicated(KEY).any():
        raise ValueError("Duplicate province labels")
    context = signals.merge(labels[[*KEY, "S_state", "structure_quality", "label"]], on=KEY, how="left", validate="many_to_one")
    context["proxy_positive"] = context.joint_support & context.S_state.eq("support")
    context["proxy_negative"] = context.joint_stable & context.S_state.isin(["support", "against"])
    context["proxy_eligible"] = context.joint_valid & context.structure_quality.eq("ok") & context.S_state.isin(["support", "against"])
    proxy_groups = context.groupby(KEY)[["proxy_positive", "proxy_negative", "proxy_eligible"]].sum().reset_index()
    proxy_groups.to_csv(out / "local_proxy_counts.csv", index=False, encoding="utf-8-sig")
    split_reports = {}
    for name, mask in (("train", context.month.le(cfg["train_end"])),
                       ("validation", context.month.eq(cfg["validation_month"])),
                       ("test", context.month.eq(cfg["test_month"]))):
        sample = context.loc[mask]
        split_reports[name] = {
            "local_proxy_positive": int(sample.proxy_positive.sum()), "local_proxy_negative": int(sample.proxy_negative.sum()),
            "eligible_local_rows": int(sample.proxy_eligible.sum()),
            "positive_unique_blocks": int(sample.loc[sample.proxy_positive, "block_code"].nunique()),
            "negative_unique_blocks": int(sample.loc[sample.proxy_negative, "block_code"].nunique()),
            "by_region": records(sample.groupby(["geo_code", "province"])[["proxy_positive", "proxy_negative"]].sum().reset_index()),
        }
    # Assess sensitivity to very low flow using cutoffs learned on training data only.
    train = signals.loc[signals.month.le(cfg["train_end"]) & signals.persistent_valid]
    sensitivity = []
    for (region, sex, age), part in train.groupby(["geo_code", "sex", "age_band"]):
        for quantile in (0.1, 0.25):
            cutoff = float(part.baseline_flow.quantile(quantile))
            sample = signals.loc[signals.geo_code.eq(region) & signals.sex.eq(sex) & signals.age_band.eq(age)
                                 & signals.persistent_valid & signals.baseline_flow.ge(cutoff)]
            for month, group in sample.groupby("month"):
                sensitivity.append({"geo_code": region, "sex": sex, "age_band": age, "month": month,
                                    "train_baseline_quantile": quantile, "baseline_cutoff": cutoff,
                                    "eligible": len(group), "persistent_blocks": int(group.persistent_support.sum()),
                                    "persistent_share": float(group.persistent_support.mean())})
    pd.DataFrame(sensitivity).to_csv(out / "baseline_sensitivity.csv", index=False, encoding="utf-8-sig")
    total = summary.loc[summary.month.ge(cfg["train_end"])]
    training_summary = summary.loc[summary.month.le(cfg["train_end"]) & summary.persistent_eligible.gt(0)]
    has_local_classes = all(s["local_proxy_positive"] > 0 and s["local_proxy_negative"] > 0 for s in split_reports.values())
    report = {
        "region_codes": cfg["region_codes"], "mapping_mode": cfg["mapping_mode"],
        "decline_threshold": cfg["suspect_decline"], "stable_change": cfg["stable_change"],
        "unique_blocks": int(features.block_code.nunique()), "province_cohort_rows": len(summary),
        "mapped_blocks": int(read_block_mapping(config, read_geography(config)).block_code.nunique()),
        "training_province_rows_with_history": len(training_summary),
        "training_months_with_persistent_history": sorted(training_summary.month.unique()),
        "local_proxy_split_counts": split_reports,
        "local_proxy_has_both_classes_in_each_split": has_local_classes,
        "province_classifier_ready": False,
        "reason": "Local contraction shares are exploratory features, not province-level positive labels. No share cutoff was selected or fit to validation/test data. Effective training history remains one month with two regions.",
        "notes": ["Every share uses quality-valid blocks as denominator; missing history is not zero contraction.",
                  "Persistent means the same block passes two consecutive comparisons; each comparison uses its own 3-month common-grid window.",
                  "Count shares and baseline-flow-weighted shares are both reported; tiny flows can inflate count shares.",
                  "Local proxy counts inherit province-level structural context; they are neither independent block structural measurements nor verified isolation labels.",
                  "No original labels, trained models, or thresholds were changed."],
        "provenance": {"clean_manifest_sha256": fingerprint(config.output / "clean_manifest.json"),
                       "label_reference_sha256": fingerprint(destination(config, "label_reference.json"))},
    }
    report_json(config, "exploration/feasibility", report)
    write_plot(out, total, cfg["suspect_decline"])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def write_plot(out, summary, threshold):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    chart = summary.copy()
    chart["group"] = chart.province + " / " + chart.sex + " / " + chart.age_band
    figure = make_subplots(rows=1, cols=2, shared_yaxes=True,
                          subplot_titles=["소지역 개수 기준 (%)", "기준 유동량 가중 (%)"])
    for col, metric in enumerate(("persistent_share", "persistent_baseline_flow_share"), start=1):
        matrix = chart.pivot(index="group", columns="month", values=metric).sort_index() * 100
        figure.add_trace(go.Heatmap(x=matrix.columns, y=matrix.index, z=matrix.to_numpy(),
                                   colorscale="YlOrRd", zmin=0, coloraxis="coloraxis",
                                   text=matrix.round(2).to_numpy(), texttemplate="%{text:.2f}",
                                   hovertemplate="%{y}<br>%{x}<br>%{z:.3f}%<extra></extra>"), row=1, col=col)
    figure.update_layout(title=f"서울·강원: 두 연속 비교에서 {threshold:.0%} 이상 위축된 소지역 비중",
                         height=900, width=1200, coloraxis_colorscale="YlOrRd", coloraxis_cmin=0)
    figure.update_xaxes(type="category")
    figure.write_html(out / "contraction_shares.html", include_plotlyjs=True)
