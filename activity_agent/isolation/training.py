"""Readiness, training and evaluation for the separate isolation-suspicion task."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from activity_agent.common.config import Config
from activity_agent.isolation.config import TASK_ID, destination, load, report_json, save, settings
from activity_agent.common.storage import digest, records
from activity_agent.isolation.labeling import IsolationLabelPolicy, isolation_reviews
from activity_agent.common.evaluation import metrics, temporal_split

NUMERIC_INPUTS = ["activity_decline", "previous_activity_decline", "current_flow", "baseline_flow", "coverage",
                  "daytime_decline", "previous_daytime_decline", "weekend_decline", "previous_weekend_decline",
                  "social_consumption_decline", "previous_social_consumption_decline", "structure_value"]
CATEGORICAL_INPUTS = ["sex", "age_band", "structure_profile"]
MODEL_INPUTS = [*NUMERIC_INPUTS, *CATEGORICAL_INPUTS]
LABEL_KEY = ["cohort_id", "feature_version", "label_version"]


def prediction_eligible(cohorts):
    return (cohorts.quality.eq("ok") & cohorts.previous_quality.eq("ok") & cohorts.structure_quality.eq("ok")
            & cohorts[["activity_decline", "previous_activity_decline", "current_flow", "baseline_flow", "coverage", "structure_value"]].notna().all(axis=1))


def supported_by_model(model, cohorts):
    profiles = {tuple(r[col] for col in CATEGORICAL_INPUTS) for r in model["supported_profiles"]}
    supported = pd.Series([tuple(row) in profiles for row in cohorts[CATEGORICAL_INPUTS].itertuples(index=False, name=None)],
                          index=cohorts.index, dtype=bool)
    region_scope = cohorts.geo_code.isin(model["region_codes"]) if model.get("region_codes") else True
    return prediction_eligible(cohorts) & supported & region_scope


def labeled_dataset(config: Config, source="weak"):
    cohorts = load(config, "cohorts")
    labels = load(config, "labels").merge(cohorts[["cohort_id", "feature_version"]],
                                         on=["cohort_id", "feature_version"], validate="one_to_one")
    decision = labels[[*LABEL_KEY, "label"]].copy()
    if source == "reviewed":
        review = isolation_reviews(config)
        decision = review[[*LABEL_KEY, "label"]]
    elif source != "weak":
        raise ValueError("source must be weak or reviewed")
    # Every review remains tied to the exact current feature AND label-policy version.
    keys = labels[[*LABEL_KEY]]
    current = keys.merge(decision, on=LABEL_KEY, validate="one_to_one")
    joined = cohorts.merge(current, on=["cohort_id", "feature_version"], validate="one_to_one")
    joined = joined.loc[joined.label.ge(0) & prediction_eligible(joined)].copy()
    return cohorts, labels, joined


def validate_training_readiness(config: Config, source="weak") -> dict:
    cfg = settings(config)
    cohorts, labels, usable = labeled_dataset(config, source)
    training, validation, test = temporal_split(usable, cfg["train_end"], cfg["validation_month"], cfg["test_month"])
    training = training.loc[~training.district_code.isin(cfg["holdout_districts"])]
    validation = validation.loc[~validation.district_code.isin(cfg["holdout_districts"])]
    blockers = []
    region_level = "sido" if cfg["geo_level"] == "sido" else "sigungu"
    if cohorts.empty:
        blockers.append("No mapped cohorts: supply verified geography and block mappings")
    elif not cohorts.structure_quality.eq("ok").any():
        blockers.append("No exact geography/sex/age structural evidence available under the selected time mode")
    if len(cohorts) and (set(cohorts.geo_level) != {cfg["geo_level"]} or set(cohorts.time_mode) != {cfg["time_mode"]}):
        blockers.append("Cohorts do not match selected geography/time mode; rebuild them")
    if cfg["region_codes"] and not set(cohorts.geo_code).issubset(cfg["region_codes"]):
        blockers.append("Cohorts contain regions outside the selected scope; rebuild them")
    if training.label.nunique() != 2:
        blockers.append("Training requires both positive and negative non-abstaining labels")
    if training.district_code.nunique() < cfg["min_training_regions"]:
        blockers.append(f"Need at least {cfg['min_training_regions']} independent training regions ({region_level}); repeated months/groups do not add regions")
    if validation.empty:
        blockers.append("No eligible validation labels")
    if test.empty:
        blockers.append("No eligible test labels")
    if cfg["holdout_districts"] and not cohorts.district_code.isin(cfg["holdout_districts"]).any():
        blockers.append("Configured holdout regions are absent")
    policy_path = destination(config, "label_reference.json")
    if not policy_path.exists():
        blockers.append("Missing training-only label policy; rerun label --task isolation")
    else:
        policy = IsolationLabelPolicy.from_dict(json.loads(policy_path.read_text(encoding="utf-8")))
        if policy.cfg != cfg or (len(labels) and set(labels.label_version) != {policy.version}):
            blockers.append("Labels do not match current policy/settings; rebuild isolation labels")
    if len(labels) != len(cohorts):
        blockers.append("Labels do not cover the current feature snapshot; rebuild isolation labels")
    breakdown = records(labels.groupby(["geo_level", "sex", "age_band", "label"], dropna=False).size().reset_index(name="rows"))
    result = {"task_id": TASK_ID, "time_mode": cfg["time_mode"], "geo_level": cfg["geo_level"], "label_source": source,
              "region_codes": cfg["region_codes"], "mapping_mode": cfg["mapping_mode"],
              "evaluation_scope": "Later months within the selected regions; not unseen-region generalization",
              "warnings": (["Only two regions: structural quantiles are relative comparisons between those two regions"]
                           if cohorts.geo_code.nunique() == 2 else [])
                          + (["Spatial mapping uses agreement between two UTM-K candidates; source datum remains undocumented"]
                             if cfg["mapping_mode"] == "crs_consensus" else []),
              "training_ready": not blockers, "blockers": blockers, "cohort_rows": len(cohorts),
              "independent_regions": int(cohorts.geo_code.nunique()), "independent_districts": int(cohorts.district_code.nunique()),
              "evaluation_region_level": region_level, "independent_evaluation_regions": int(cohorts.district_code.nunique()),
              "structure_matching_rows": int(cohorts.structure_quality.eq("ok").sum()),
              "structure_quality": cohorts.structure_quality.value_counts().to_dict(),
              "label_coverage": float(labels.label.ge(0).mean()) if len(labels) else 0,
              "split_rows": {"train": len(training), "validation": len(validation), "test": len(test)},
              "split_label_counts": {name: {str(k): int(v) for k, v in data.label.value_counts().items()}
                                     for name, data in (("train", training), ("validation", validation), ("test", test))},
              "by_region": records(labels.groupby(["geo_code", "province", "month", "label"], dropna=False)
                                   .size().reset_index(name="rows")),
              "label_breakdown": breakdown,
              "note": "2024 deaths are background only; labels are not independently established isolation outcomes"}
    for name in ("structural_quality", "cohort_manifest"):
        path = destination(config, name + ".json")
        if path.exists():
            result[name] = json.loads(path.read_text(encoding="utf-8"))
    report_json(config, "readiness", result)
    return result


def train_model(config: Config, source="weak") -> dict:
    cfg = settings(config)
    readiness = validate_training_readiness(config, source)
    if not readiness["training_ready"]:
        raise ValueError("Isolation training blocked: " + "; ".join(readiness["blockers"]))
    cohorts, labels, usable = labeled_dataset(config, source)
    policy = IsolationLabelPolicy.from_dict(json.loads(destination(config, "label_reference.json").read_text(encoding="utf-8")))
    if policy.cfg != cfg or set(labels.label_version) != {policy.version}:
        raise ValueError("Label policy/settings changed; rerun label --task isolation")
    training, validation, test = temporal_split(usable, cfg["train_end"], cfg["validation_month"], cfg["test_month"])
    training = training.loc[~training.district_code.isin(cfg["holdout_districts"])].sort_values("cohort_id")
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler())
    transform = ColumnTransformer([("numeric", numeric, NUMERIC_INPUTS),
                                   ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_INPUTS)])
    pipeline = make_pipeline(transform, LogisticRegression(class_weight="balanced", max_iter=3000, random_state=config.values["seed"]))
    pipeline.fit(training[MODEL_INPUTS], training.label)
    model_version = "isolation-logistic-v1:" + digest({"data": records(training[["cohort_id", "feature_version", *MODEL_INPUTS, "label"]]),
                                                     "policy": policy.to_dict(), "source": source, "seed": config.values["seed"]})
    artifact = {"task_id": TASK_ID, "pipeline": pipeline, "model_version": model_version, "source": source,
                "inputs": MODEL_INPUTS, "geo_level": cfg["geo_level"], "time_mode": cfg["time_mode"],
                "region_codes": cfg["region_codes"], "mapping_mode": cfg["mapping_mode"],
                "train_end": cfg["train_end"], "validation_month": cfg["validation_month"], "test_month": cfg["test_month"],
                "holdout_districts": cfg["holdout_districts"], "policy": policy.to_dict(),
                "training_ids": training.cohort_id.tolist(), "training_districts": sorted(training.district_code.unique()),
                "evaluation_region_level": "sido" if cfg["geo_level"] == "sido" else "sigungu",
                "training_regions": sorted(training.district_code.unique()), "holdout_regions": cfg["holdout_regions"],
                "supported_profiles": records(training[CATEGORICAL_INPUTS].drop_duplicates().sort_values(CATEGORICAL_INPUTS)),
                "feature_versions": sorted(training.feature_version.unique())}
    path = destination(config, "models/logistic.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(artifact, stream)
    temporary.replace(path)
    report_json(config, "models/logistic", {k: v for k, v in artifact.items() if k not in ("pipeline", "training_ids")}
                | {"training_rows": len(training), "activated": False, "score_meaning": "Weak/reviewed isolation-suspicion classification score, not verified isolation probability"})
    result = evaluate_model(config)
    print(f"Trained {model_version}; isolation model saved separately and not automatically activated", flush=True)
    return result


def read_isolation_model(config: Config):
    # Load only locally generated, trusted pickle artifacts.
    with destination(config, "models/logistic.pkl").open("rb") as stream:
        artifact = pickle.load(stream)
    if artifact.get("task_id") != TASK_ID or artifact.get("inputs") != MODEL_INPUTS:
        raise ValueError("Isolation model schema/task mismatch")
    return artifact


def evaluate_model(config: Config):
    model = read_isolation_model(config)
    cohorts = load(config, "cohorts")
    if not cohorts.empty and (set(cohorts.geo_level) != {model["geo_level"]} or set(cohorts.time_mode) != {model["time_mode"]}):
        raise ValueError("Model/cohort geography or time mode mismatch")
    policy = IsolationLabelPolicy.from_dict(model["policy"])
    rule = policy.label(cohorts)
    independent = isolation_reviews(config, independent_only=True)
    result = {"task_id": TASK_ID, "model_version": model["model_version"], "time_mode": model["time_mode"],
              "region_codes": model.get("region_codes", []), "mapping_mode": model.get("mapping_mode", "verified"),
              "label_source": model["source"], "splits": {},
              "note": "Weak-label agreement is not validated isolation detection performance; retrospective data are not historical real-time forecasts"}
    for name in ("validation", "test"):
        sample = rule.loc[rule.month.eq(model[name + "_month"]) & supported_by_model(model, rule)].copy()
        if name == "validation":
            sample = sample.loc[~sample.district_code.isin(model["holdout_districts"])]
        if set(sample.cohort_id) & set(model["training_ids"]):
            raise ValueError("Training/evaluation cohorts overlap")
        split_result = {"eligible_rows": len(sample), "independent_districts": int(sample.district_code.nunique()),
                        "evaluation_region_level": "sido" if model["geo_level"] == "sido" else "sigungu",
                        "independent_evaluation_regions": int(sample.district_code.nunique()),
                        "weak_label_coverage": float(sample.label.ge(0).mean()) if len(sample) else 0}
        for label_source in ("weak_label_proxy", "independent_review"):
            if label_source == "weak_label_proxy":
                data = sample.loc[sample.label.ge(0)].copy()
                data["truth"] = data.label
            else:
                truth = independent[[*LABEL_KEY, "label"]].rename(columns={"label": "truth"})
                data = sample.merge(truth, on=LABEL_KEY, validate="one_to_one")
                data = data.loc[data.truth.ge(0)].copy()
            if data.empty:
                split_result[label_source] = {"rows": 0, "reason": "No eligible labels for this evaluation source"}
                continue
            scores = model["pipeline"].predict_proba(data[MODEL_INPUTS])[:, 1]
            covered = data.label.ge(0)
            rule_metrics = metrics(data.loc[covered, "truth"], data.loc[covered, "label"].astype(float))
            details = {"model": metrics(data.truth, scores), "rule_baseline": rule_metrics,
                       "rule_coverage": float(covered.mean()),
                       "by_region": [{"geo_code": code, "province": part.province.iloc[0],
                                      **metrics(part.truth, scores[data.index.get_indexer(part.index)])}
                                     for code, part in data.groupby("geo_code")],
                       "by_group": [{"sex": sex, "age_band": age, **metrics(part.truth, scores[data.index.get_indexer(part.index)])}
                                    for (sex, age), part in data.groupby(["sex", "age_band"])]}
            if name == "test" and model["holdout_districts"]:
                held = data.district_code.isin(model["holdout_districts"])
                details["heldout_districts"] = metrics(data.loc[held, "truth"], scores[held])
                details["heldout_district_count"] = int(data.loc[held, "district_code"].nunique())
                details["heldout_regions"] = details["heldout_districts"]
                details["heldout_region_count"] = details["heldout_district_count"]
            split_result[label_source] = details
        result["splits"][name] = split_result
    report_json(config, "evaluation", result)
    return result


def predict_isolation(config: Config) -> pd.DataFrame:
    model = read_isolation_model(config)
    cohorts = load(config, "cohorts")
    if not cohorts.empty and (set(cohorts.geo_level) != {model["geo_level"]} or set(cohorts.time_mode) != {model["time_mode"]}):
        raise ValueError("Model/cohort geography or time mode mismatch")
    out = cohorts.copy()
    out["score"] = np.nan
    out["status"] = "판단 보류: 자료 부족 또는 학습 종료 이전 시점"
    mask = supported_by_model(model, out) & out.month.gt(model["train_end"])
    if mask.any():
        out.loc[mask, "score"] = model["pipeline"].predict_proba(out.loc[mask, MODEL_INPUTS])[:, 1] * 100
        out.loc[mask, "status"] = "고립 의심 라벨 분류 점수 산출"
    out["engine_version"] = model["model_version"]
    out["label_version"] = model["policy"]["label_version"]
    out["score_kind"] = "고립 의심 라벨 분류 점수; 검증된 고립 확률 아님"
    save(config, "detections", out)
    return out
