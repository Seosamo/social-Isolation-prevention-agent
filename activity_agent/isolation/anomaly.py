"""Label-free local activity anomaly ranking, restricted to the selected provinces."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from activity_agent.common.storage import digest, fingerprint, records, write_json
from activity_agent.isolation.config import destination, load, report_json, save, settings
from activity_agent.isolation.exploration import local_features

INPUTS = ["activity_decline", "previous_activity_decline", "daytime_decline", "previous_daytime_decline",
          "weekend_decline", "previous_weekend_decline"]
PROFILE = ["geo_code", "sex", "age_band"]
MODEL_FILE = "activity_agent/model/isolation_forest.pkl"


def selected_model_id(config):
    identity = config.values.get("isolation", {}).get("model_id", "IF_BASELINE")
    if identity not in ("IF_BASELINE", "IF_F2"):
        raise ValueError("isolation.model_id must be IF_BASELINE or IF_F2")
    return identity


def model_path(config):
    """Resolve the version-controlled operating model independently of artifacts."""
    if selected_model_id(config) == "IF_F2":
        from pathlib import Path
        name = config.values["isolation"].get("model_path")
        if not name:
            raise ValueError("IF_F2 requires isolation.model_path")
        path = Path(name)
        return path if path.is_absolute() else config.root / path
    return config.root / MODEL_FILE


def relocated_model_path(path):
    """Resolve old experiment hash records without rewriting their provenance."""
    from pathlib import Path
    path = Path(path)
    if path.parts[-4:-1] == ("isolation", "anomaly", "models") and path.name in (
            "isolation_forest.pkl", "isolation_forest.json"):
        for parent in path.parents:
            target = parent / "activity_agent/model" / path.name
            if target.is_file():
                return target
    return path


def anomaly_settings(config):
    values = {"n_estimators": 200, "reference_quantile": .95,
              "baseline_quantile": .1, "min_baseline_flow": 1., "minimum_decline": .1,
              "stability_seeds": [42, 43, 44]} | config.values.get("isolation", {}).get("anomaly", {})
    if not .5 < values["reference_quantile"] < 1 or not 0 <= values["baseline_quantile"] < 1:
        raise ValueError("Invalid anomaly reference/baseline quantile")
    if values["min_baseline_flow"] < 0 or not 0 <= values["minimum_decline"] <= 1:
        raise ValueError("Invalid anomaly baseline/decline setting")
    if values["n_estimators"] < 10 or not values["stability_seeds"]:
        raise ValueError("Anomaly analysis requires estimators and stability seeds")
    return values


def quality_mask(frame):
    qualities = ["quality", "previous_quality", "day_quality", "previous_day_quality", "weekend_quality", "previous_weekend_quality"]
    return (frame[qualities].eq("ok").all(axis=1) & np.isfinite(frame[INPUTS]).all(axis=1)
            & frame.baseline_flow.gt(0) & frame.previous_baseline_flow.gt(0))


def fit_profiles(training, cfg):
    profiles = []
    for key, group in training.groupby(PROFILE, sort=True):
        cutoff = max(cfg["min_baseline_flow"], float(group.baseline_flow.quantile(cfg["baseline_quantile"])))
        kept = group.loc[group.baseline_flow.ge(cutoff)]
        if len(kept) < 8:
            continue
        median = kept[INPUTS].median()
        scale = (kept[INPUTS].quantile(.75) - kept[INPUTS].quantile(.25)).clip(lower=.001)
        profiles.append(dict(zip(PROFILE, key)) | {"baseline_cutoff": cutoff,
                        "median": median.to_dict(), "scale": scale.to_dict(), "training_rows": len(kept)})
    return profiles


def transform(frame, profiles):
    valid = quality_mask(frame)
    matrix = pd.DataFrame(np.nan, index=frame.index, columns=INPUTS)
    cutoff = pd.Series(np.nan, index=frame.index, dtype=float)
    for profile in profiles:
        same = pd.Series(True, index=frame.index)
        for key in PROFILE:
            same &= frame[key].eq(profile[key])
        cutoff.loc[same] = profile["baseline_cutoff"]
        rows = same & valid & frame.baseline_flow.ge(profile["baseline_cutoff"])
        matrix.loc[rows] = ((frame.loc[rows, INPUTS] - pd.Series(profile["median"])) / pd.Series(profile["scale"])).clip(-20, 20)
    eligible = valid & matrix.notna().all(axis=1)
    return matrix, eligible, cutoff


def forest(cfg, seed):
    return IsolationForest(n_estimators=cfg["n_estimators"], max_samples="auto", contamination="auto",
                           random_state=seed, n_jobs=1)


def score_frame(model, frame):
    if model.get("model_id") == "IF_F2":
        from .f2 import score_frame as score_f2
        return score_f2(model, frame)
    result = frame.copy()
    matrix, eligible, cutoff = transform(frame, model["profiles"])
    eligible &= frame.geo_code.isin(model["region_codes"])
    result["baseline_cutoff"] = cutoff
    result["anomaly_raw"] = np.nan
    result["anomaly_percentile"] = np.nan
    publish = eligible & frame.month.gt(model["train_end"])
    if publish.any():
        scores = -model["forest"].score_samples(matrix.loc[publish, INPUTS])
        result.loc[publish, "anomaly_raw"] = scores
        result.loc[publish, "anomaly_percentile"] = 100 * np.searchsorted(model["reference_scores"], scores, side="right") / len(model["reference_scores"])
    result["quality_eligible"] = eligible
    result["direction_support"] = (frame.activity_decline.ge(model["anomaly_settings"]["minimum_decline"])
                                    & (frame.daytime_decline.gt(0) | frame.weekend_decline.gt(0)))
    result["candidate"] = publish & result.anomaly_raw.ge(model["score_cutoff"]) & result.direction_support
    result["persistent_20pct"] = frame.activity_decline.ge(.2) & frame.previous_activity_decline.ge(.2)
    result["status"] = "자료 이력·품질 부족 또는 학습 범위 밖"
    result.loc[quality_mask(frame) & cutoff.notna() & ~eligible, "status"] = "기준 유동량 부족 또는 학습 범위 밖"
    result.loc[eligible & ~publish, "status"] = "학습 기간: 예측 순위 미제공"
    result.loc[publish, "status"] = "활동 이상 점수 산출"
    result.loc[result.candidate, "status"] = "활동 위축 이상 후보"
    result["engine_version"] = model["model_version"]
    result["score_kind"] = "학습 분포 대비 활동 이상 백분위; 고립 확률 아님"
    return result


def attach_context(config, frame):
    """Province composition is display context, never an anomaly model input."""
    if not destination(config, "structural.parquet").exists():
        return frame.assign(province_structure_context=np.nan)
    structural = load(config, "structural")
    part = structural.loc[structural.metric.eq("single_household_share") & structural.geo_level.eq("sido")
                          & structural.geo_code.ne("") & structural.quality.eq("ok")].copy()
    part["year"] = part.reference_period.str[:4]
    part = part[["geo_code", "sex", "age_band", "year", "value", "available_at", "reference_period"]].rename(
        columns={"value": "province_structure_context", "available_at": "context_available_at", "reference_period": "context_period"})
    if part.duplicated([*PROFILE, "year"]).any():
        raise ValueError("Ambiguous structural context")
    result = frame.assign(year=frame.month.str[:4]).merge(part, on=[*PROFILE, "year"], how="left", validate="many_to_one")
    if settings(config)["time_mode"] == "as_of":
        from activity_agent.isolation.cohorts import period_end
        available = pd.to_datetime(result.context_available_at, errors="coerce", utc=True)
        decision = pd.to_datetime(result.month.map(lambda m: pd.Period(m, freq="M").end_time), utc=True)
        ended = pd.to_datetime(result.context_period.map(lambda p: period_end(p) if pd.notna(p) else pd.NaT), utc=True)
        result.loc[~(available.notna() & available.le(decision) & ended.le(decision)), "province_structure_context"] = np.nan
    return result.drop(columns="year")


def train_anomaly(config):
    if selected_model_id(config) == "IF_F2":
        raise ValueError("IF_F2 uses its saved experiment model; run detect --task isolation (no retraining)")
    cfg, params = settings(config), anomaly_settings(config)
    if cfg["geo_level"] != "sido" or cfg["holdout_regions"]:
        raise ValueError("Anomaly experiment uses selected sido regions with temporal evaluation; no region holdout")
    print("Building label-free local activity features...", flush=True)
    frame = local_features(config).sort_values(["month", "block_code", "sex", "age_band"]).reset_index(drop=True)
    training = frame.loc[frame.month.le(cfg["train_end"]) & quality_mask(frame)]
    profiles = fit_profiles(training, params)
    matrix, eligible, _ = transform(frame, profiles)
    fit_rows = eligible & frame.month.le(cfg["train_end"])
    if fit_rows.sum() < 32 or frame.loc[fit_rows, "geo_code"].nunique() < 2:
        raise ValueError("Need at least 32 quality-valid local rows across two selected regions")
    version = "local-isolation-forest-v1:" + digest({"rows": records(frame.loc[fit_rows, ["block_code", "month", *PROFILE, *INPUTS, "baseline_flow"]]),
                                                   "profiles": profiles, "params": params, "seed": config.values["seed"],
                                                   "split": {k: cfg[k] for k in ("train_end", "validation_month", "test_month", "time_mode", "mapping_mode")}})
    estimator = forest(params, config.values["seed"])
    estimator.fit(matrix.loc[fit_rows, INPUTS])
    reference = np.sort(-estimator.score_samples(matrix.loc[fit_rows, INPUTS]))
    model = {"task_id": "local_activity_anomaly_v1", "forest": estimator, "profiles": profiles,
             "inputs": INPUTS, "model_version": version, "region_codes": sorted(frame.loc[fit_rows, "geo_code"].unique()),
             "train_end": cfg["train_end"], "validation_month": cfg["validation_month"], "test_month": cfg["test_month"],
             "reference_scores": reference, "score_cutoff": float(np.quantile(reference, params["reference_quantile"])),
             "anomaly_settings": params, "mapping_mode": cfg["mapping_mode"],
             "seed": config.values["seed"], "time_mode": cfg["time_mode"],
             "training_rows": int(fit_rows.sum()), "training_blocks": int(frame.loc[fit_rows, "block_code"].nunique()),
             "training_months": sorted(frame.loc[fit_rows, "month"].unique()),
             "training_snapshot_sha": digest(records(frame.loc[fit_rows, ["block_code", "month", *PROFILE, *INPUTS]])),
             "clean_manifest_sha256": fingerprint(config.output / "clean_manifest.json")}
    scored = score_frame(model, frame)
    report = diagnostics(config, model, frame, matrix, fit_rows, scored)
    scored = attach_context(config, scored)
    path = model_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(model, stream)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), {k: v for k, v in model.items() if k not in ("forest", "reference_scores")})
    save(config, "anomaly/detections", scored)
    scored.loc[scored.candidate].sort_values(["month", "anomaly_raw", "activity_decline"], ascending=[True, False, False]).to_csv(
        destination(config, "anomaly/candidates.csv"), index=False, encoding="utf-8-sig")
    report_json(config, "anomaly/evaluation", report)
    report_json(config, "anomaly/readiness", anomaly_readiness(config))
    write_chart(config, scored, model)
    print(json.dumps({k: report[k] for k in ("model_version", "training_rows", "training_blocks", "training_months", "monthly")}, ensure_ascii=False, indent=2), flush=True)
    return report


def diagnostics(config, model, frame, matrix, fit_rows, scored):
    monthly = []
    seed_sets = {}
    for seed in model["anomaly_settings"]["stability_seeds"]:
        alternate = forest(model["anomaly_settings"], seed)
        alternate.fit(matrix.loc[fit_rows, INPUTS])
        reference = np.sort(-alternate.score_samples(matrix.loc[fit_rows, INPUTS]))
        clone = model | {"forest": alternate, "reference_scores": reference,
                         "score_cutoff": float(np.quantile(reference, model["anomaly_settings"]["reference_quantile"]))}
        alt = score_frame(clone, frame)
        seed_sets[seed] = alt
    future = scored.loc[scored.month.gt(model["train_end"])]
    for (month, region), group in future.groupby(["month", "geo_code"]):
        top = group.loc[group.candidate].sort_values("anomaly_raw", ascending=False).drop_duplicates("block_code").head(20)
        top_set = set(top.block_code)
        stability = []
        for seed, alt in seed_sets.items():
            other = alt.loc[alt.month.eq(month) & alt.geo_code.eq(region) & alt.candidate]
            other_set = set(other.sort_values("anomaly_raw", ascending=False).drop_duplicates("block_code").head(20).block_code)
            union = top_set | other_set
            stability.append({"seed": seed, "top20_block_jaccard": len(top_set & other_set) / len(union) if union else None})
        baseline = group.loc[group.quality_eligible & group.direction_support].sort_values("activity_decline", ascending=False).drop_duplicates("block_code").head(20)
        baseline_set = set(baseline.block_code)
        union = top_set | baseline_set
        threshold_sensitivity = []
        for quantile in (.90, .95, .975):
            candidate = (group.anomaly_raw.ge(float(np.quantile(model["reference_scores"], quantile))) & group.direction_support)
            threshold_sensitivity.append({"training_quantile": quantile, "candidate_rows": int(candidate.sum()),
                                          "candidate_blocks": int(group.loc[candidate, "block_code"].nunique())})
        monthly.append({"month": month, "geo_code": region, "province": group.province.iloc[0],
                        "eligible_rows": int(group.anomaly_raw.notna().sum()), "candidate_rows": int(group.candidate.sum()),
                        "candidate_blocks": int(group.loc[group.candidate, "block_code"].nunique()),
                        "persistent_20pct_rows": int((group.candidate & group.persistent_20pct).sum()),
                        "threshold_sensitivity": threshold_sensitivity,
                        "percentile_100_rows": int(group.anomaly_percentile.eq(100).sum()),
                        "seed_stability": stability,
                        "decline_baseline_top20_block_jaccard": len(top_set & baseline_set) / len(union) if union else None})
    return {"model_version": model["model_version"], "training_rows": model["training_rows"],
            "training_blocks": model["training_blocks"], "training_months": model["training_months"],
            "monthly": monthly, "accuracy_evaluated": False,
            "notes": ["No isolation labels, structural values, region codes, sex or age identifiers are model inputs.",
                      "Median/IQR normalization and small-flow cutoffs are fit per region/sex/age on training data only.",
                      "Only one effective training month is available: cross-sectional activity anomaly ranking, not learned normal seasonality.",
                      "95th training-score quantile is an operating threshold, not an assumed true anomaly prevalence.",
                      "Candidate direction requires at least 10% current group decline and positive daytime or weekend decline.",
                      "Repeated age/sex rows are not independent regions or people. Structural statistics are province-level context only.",
                      "Seed stability and baseline overlap are diagnostics, not isolation detection accuracy."]}


def read_model(config):
    if selected_model_id(config) == "IF_F2":
        from .f2 import read_model as read_f2
        return read_f2(config)
    with model_path(config).open("rb") as stream:
        model = pickle.load(stream)
    if model.get("task_id") != "local_activity_anomaly_v1" or model.get("inputs") != INPUTS:
        raise ValueError("Anomaly model task/input mismatch")
    cfg = settings(config)
    if sorted(cfg["region_codes"]) != model["region_codes"]:
        raise ValueError("Anomaly region scope changed; retrain the model")
    if (any(cfg[key] != model[key] for key in ("train_end", "validation_month", "test_month", "mapping_mode"))
            or anomaly_settings(config) != model["anomaly_settings"]
            or model.get("seed", config.values["seed"]) != config.values["seed"]
            or model.get("time_mode", "retrospective") != cfg["time_mode"]):
        raise ValueError("Anomaly settings changed; retrain the model")
    return model


def detect_anomaly(config):
    model = read_model(config)
    if selected_model_id(config) == "IF_F2":
        from .f2 import prepare_features
        frame = prepare_features(config, model)
    else:
        frame = local_features(config).reset_index(drop=True)
    scored = attach_context(config, score_frame(model, frame))
    save(config, "anomaly/detections", scored)
    scored.loc[scored.candidate].sort_values(["month", "anomaly_raw", "activity_decline"], ascending=[True, False, False]).to_csv(
        destination(config, "anomaly/candidates.csv"), index=False, encoding="utf-8-sig")
    write_chart(config, scored, model)
    if model.get("model_id") == "IF_F2":
        from .f2 import evaluation
        report = evaluation(model, scored)
        report_json(config, "anomaly/evaluation", report)
        report_json(config, "anomaly/deployment", report)
        report_json(config, "anomaly/readiness", anomaly_readiness(config))
    return scored


def load_candidates(config, month=None, sex=None, age=None, region=None, limit=20):
    model = read_model(config)
    frame = load(config, "anomaly/detections")
    if set(frame.engine_version) != {model["model_version"]}:
        raise ValueError("Anomaly predictions do not match the model; rerun detect --task isolation")
    frame = frame.loc[frame.month.eq(month or frame.month.max()) & frame.candidate].copy()
    for key, value in (("sex", sex), ("age_band", age)):
        if value:
            frame = frame.loc[frame[key].eq(value)]
    if region:
        from activity_agent.isolation.structural import province_name
        frame = frame.loc[frame.geo_code.eq(region) | frame.province.eq(province_name(region))]
    return frame.sort_values(["anomaly_raw", "activity_decline", "block_code"], ascending=[False, False, True]).head(limit)


def anomaly_readiness(config):
    try:
        model = read_model(config)
        predictions = load(config, "anomaly/detections")
        if set(predictions.engine_version) != {model["model_version"]}:
            raise ValueError("Anomaly predictions do not match the model")
    except (FileNotFoundError, ValueError) as error:
        return {"method": "isolation_forest", "training_ready": False, "blockers": [str(error)]}
    return {"method": "isolation_forest", "training_ready": True, "blockers": [],
            "model_version": model["model_version"], "training_rows": model["training_rows"],
            "training_blocks": model["training_blocks"], "training_months": model["training_months"],
            "region_codes": model["region_codes"], "accuracy_evaluated": False}


def write_chart(config, scored, model):
    import plotly.express as px
    import plotly.graph_objects as go

    chart = scored.loc[scored.candidate].sort_values("anomaly_raw", ascending=False).drop_duplicates(["month", "block_code"]).copy()
    chart["유동 감소율 (%)"] = chart.activity_decline * 100
    if chart.empty:
        figure = go.Figure().add_annotation(text="활동 위축 이상 후보 없음", showarrow=False)
        figure.write_html(destination(config, "anomaly/candidates.html"), include_plotlyjs=True)
        return
    figure = px.scatter(chart, x="유동 감소율 (%)", y="anomaly_raw", color="province", facet_col="month",
                        hover_data=["block_code", "sex", "age_band", "anomaly_percentile", "persistent_20pct"],
                        title="서울·강원 활동 위축 이상 후보: 소지역별 가장 높은 이상 점수 관측")
    figure.update_yaxes(title="활동 이상 원점수 (높을수록 학습 분포에서 이례적)")
    figure.write_html(destination(config, "anomaly/candidates.html"), include_plotlyjs=True)
