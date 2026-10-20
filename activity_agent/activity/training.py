from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from activity_agent.common.config import Config
from activity_agent.common.evaluation import metrics, temporal_split
from activity_agent.common.storage import load_table, write_json
from activity_agent.activity.detection import RuleDetector
from activity_agent.activity.labels import latest_reviews

# Explicit allowlist: never infer predictors from numeric columns, labels or review text.
PREDICTORS = {
    "mobility": ["activity_decline", "previous_decline", "current_flow", "baseline_flow", "coverage"],
    "district_mobility": ["activity_decline", "previous_decline", "current_flow", "baseline_flow", "coverage"],
    "consumption": ["consumption_decline", "social_consumption_decline", "diversity_decline", "previous_decline",
                    "previous_social_consumption_decline", "current_count", "baseline_count", "current_social_ratio"],
}


def dataset(config: Config, signal: str, source: str) -> pd.DataFrame:
    frame = load_table(config, "features")
    frame = frame.loc[frame.signal.eq(signal) & frame.quality.eq("ok")].copy()
    if source == "weak":
        labels = load_table(config, "weak_labels")[["feature_id", "feature_version", "label"]]
    elif source == "reviewed":
        labels = latest_reviews(config)[["feature_id", "feature_version", "label"]]
    else:
        raise ValueError("source must be weak or reviewed")
    return frame.merge(labels, on=["feature_id", "feature_version"], validate="one_to_one").query("label >= 0")


def train(config: Config, signal: str = "mobility", source: str = "weak", train_end: str = "202510",
          validation_month: str = "202511", test_month: str = "202512", holdout_region: str | None = None):
    if signal not in PREDICTORS:
        raise ValueError("Unsupported training signal.")
    frame = dataset(config, signal, source)
    training, validation, test = temporal_split(frame, train_end, validation_month, test_month)
    if holdout_region:
        if not frame.region.eq(holdout_region).any():
            raise ValueError("Holdout region absent; verified regional mapping may be required.")
        training = training.loc[training.region.ne(holdout_region)]
        validation = validation.loc[validation.region.ne(holdout_region)]
    if training.label.nunique() != 2:
        raise ValueError("Training requires positive and negative labels before train_end. Keep abstentions unlabeled; collect more labels instead.")
    if validation.empty or test.empty:
        raise ValueError("Validation and test months need labeled rows; choose valid splits or collect labels.")
    predictors = PREDICTORS[signal]
    if training[predictors].isna().all().any():
        raise ValueError("A required predictor is entirely missing in training.")
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000, random_state=config.values["seed"]))
    pipeline.fit(training[predictors], training.label)
    signature = "|".join(sorted(training.feature_id + ":" + training.label.astype(str)))
    model_id = hashlib.sha256((signature + config.version + source + signal).encode()).hexdigest()[:16]
    artifact = {"pipeline": pipeline, "predictors": predictors, "signal": signal, "source": source,
                "model_version": f"logistic-v1:{model_id}", "train_end": train_end,
                "validation_month": validation_month, "test_month": test_month, "holdout_region": holdout_region,
                "training_feature_ids": sorted(training.feature_id), "config_version": config.version}
    path = config.path("model_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(artifact, stream)
    write_json(path.with_suffix(".json"), {k: v for k, v in artifact.items() if k not in ("pipeline", "training_feature_ids")}
               | {"training_rows": len(training), "note": "Not activated automatically; weak labels do not establish isolation validity."})
    report = evaluate(config)
    print(f"Trained {artifact['model_version']} on {len(training):,} rows; model remains inactive", flush=True)
    return report


def read_model(path: Path) -> dict:
    # Only load locally generated, trusted artifacts; pickle is not an interchange format.
    with path.open("rb") as stream:
        model = pickle.load(stream)
    if model["signal"] not in PREDICTORS or model["predictors"] != PREDICTORS[model["signal"]]:
        raise ValueError("Model schema is not supported.")
    return model


def evaluate(config: Config) -> dict:
    model = read_model(config.path("model_path"))
    features = load_table(config, "features")
    features = features.loc[features.signal.eq(model["signal"]) & features.quality.eq("ok")].copy()
    report = {"model_version": model["model_version"], "training_label_source": model["source"],
              "note": "Weak-label agreement is a proxy metric, not validated isolation detection performance.", "splits": {}}
    weak = load_table(config, "weak_labels")[["feature_id", "feature_version", "label"]]
    independent = latest_reviews(config, independent_only=True)[["feature_id", "feature_version", "label"]]
    for name in ("validation", "test"):
        sample = features.loc[features.month.eq(model[f"{name}_month"])].copy()
        if name == "validation" and model["holdout_region"]:
            sample = sample.loc[sample.region.ne(model["holdout_region"])]
        if set(sample.feature_id) & set(model["training_feature_ids"]):
            raise ValueError("Training/evaluation feature IDs overlap.")
        report["splits"][name] = {}
        for label_name, labels in [("weak_label_proxy", weak), ("independent_review", independent)]:
            data = sample.merge(labels, on=["feature_id", "feature_version"], validate="one_to_one").query("label >= 0")
            if data.empty:
                result = {"rows": 0, "reason": "No eligible labels for this evaluation source."}
            else:
                probabilities = model["pipeline"].predict_proba(data[model["predictors"]])[:, 1]
                rule_score = RuleDetector().predict(data).score.fillna(0).to_numpy() / 100
                # Fixed 20% threshold for the rule baseline; report separately from model threshold.
                baseline = metrics(data.label, rule_score / 0.4)
                baseline["threshold_in_rule_score"] = 20
                result = {"model": metrics(data.label, probabilities), "rule_baseline": baseline}
                if name == "test" and model["holdout_region"]:
                    held = data.region.eq(model["holdout_region"])
                    result["heldout_region"] = metrics(data.loc[held, "label"], probabilities[held])
            report["splits"][name][label_name] = result
    write_json(config.output / "model_evaluation.json", report)
    return report


class SupervisedDetector:
    def __init__(self, path: Path):
        self.model = read_model(path)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # Unsupported signal types retain their explicit rule engine metadata.
        out = RuleDetector().predict(features)
        # Retrospective dashboards must not use a model trained on later months.
        mask = out.signal.eq(self.model["signal"]) & out.quality.eq("ok") & out.month.gt(self.model["train_end"])
        out.loc[mask, "score"] = self.model["pipeline"].predict_proba(out.loc[mask, self.model["predictors"]])[:, 1] * 100 if mask.any() else np.nan
        out.loc[mask, "engine"] = "supervised"
        out.loc[mask, "engine_version"] = self.model["model_version"]
        out.loc[mask, "status"] = "학습 모델 산출"
        out.loc[mask, "evidence_metrics"] = ",".join(self.model["predictors"])
        out.loc[mask, "score_kind"] = f"{self.model['source']} 라벨 분류 점수; 고립 확률 아님"
        return out
