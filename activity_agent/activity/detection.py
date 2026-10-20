from __future__ import annotations

import numpy as np
import pandas as pd

from activity_agent.common.config import Config
from activity_agent.common.storage import load_table, save_table, write_json

RULE_VERSION = "rule-v1"
METRICS = {
    "mobility": ["activity_decline"],
    "district_mobility": ["activity_decline"],
    "time_pattern": ["daytime_decline"],
    "weekday_pattern": ["weekend_decline"],
    "consumption": ["consumption_decline", "social_consumption_decline", "diversity_decline"],
}


class RuleDetector:
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        out["score"] = np.nan
        out["status"] = "판단 보류"
        out["evidence_metrics"] = ""
        for signal, metrics in METRICS.items():
            mask = out.signal.eq(signal)
            out.loc[mask, "evidence_metrics"] = ",".join(metrics)
            valid = mask & out.quality.eq("ok") & out[metrics].notna().all(axis=1)
            out.loc[valid, "score"] = out.loc[valid, metrics].clip(0, 1).mean(axis=1) * 100
            out.loc[valid, "status"] = "위축 신호 산출"
        out["engine"] = "rule"
        out["engine_version"] = RULE_VERSION
        out["score_kind"] = "활동 위축 점수; 고립 확률 아님"
        return out


def weak_labels(features: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = features.copy()
    out["label"] = -1
    out["label_reason"] = "자료 부족·상충 또는 임계값 사이"
    eligible = out.signal.isin(["mobility", "district_mobility", "consumption"])
    valid = eligible & out.quality.eq("ok") & out.previous_quality.eq("ok")
    suspect = valid & out.primary_decline.ge(config.values["suspect_decline"]) & out.previous_decline.ge(config.values["suspect_decline"])
    stable = valid & out.primary_decline.abs().le(config.values["stable_change"]) & out.previous_decline.abs().le(config.values["stable_change"])
    # Consumption suspicion needs both total and social-count contractions;
    # missing/zero social baselines never become a negative label.
    consumption = out.signal.eq("consumption")
    social_now = out.get("social_consumption_decline", pd.Series(np.nan, index=out.index))
    social_past = out.get("previous_social_consumption_decline", pd.Series(np.nan, index=out.index))
    suspect &= ~consumption | (social_now.ge(config.values["suspect_decline"]) & social_past.ge(config.values["suspect_decline"]))
    stable &= ~consumption | (social_now.abs().le(config.values["stable_change"]) & social_past.abs().le(config.values["stable_change"]))
    out.loc[suspect, ["label", "label_reason"]] = [1, "두 연속 비교에서 감소 임계값 충족"]
    out.loc[stable, ["label", "label_reason"]] = [0, "두 연속 비교에서 변화가 안정 범위 이내; 고립 부재를 뜻하지 않음"]
    out.loc[~eligible, "label_reason"] = "시간대·요일은 설명 자료이며 독립 라벨 제외"
    out["label_name"] = out.label.map({-1: "판단 보류", 0: "위축 근거 미약", 1: "위축 의심"})
    out["label_source"] = "weak_rule"
    out["label_version"] = RULE_VERSION + ":" + config.version
    return out


def combined_signals(results: pd.DataFrame) -> pd.DataFrame:
    key = ["month", "region", "sex", "age"]
    a = results.loc[results.signal.eq("district_mobility"), [*key, "score", "primary_decline", "quality"]]
    b = results.loc[results.signal.eq("consumption"), [*key, "score", "primary_decline", "quality"]]
    combined = a.merge(b, on=key, suffixes=("_mobility", "_consumption"), validate="one_to_one")
    combined["both_declining"] = (combined.quality_mobility.eq("ok") & combined.quality_consumption.eq("ok")
                                  & combined.primary_decline_mobility.gt(0) & combined.primary_decline_consumption.gt(0))
    return combined


def detect(config: Config) -> pd.DataFrame:
    features = load_table(config, "features")
    if config.values["detector"] == "rule":
        detector = RuleDetector()
    elif config.values["detector"] == "supervised":
        from activity_agent.activity.training import SupervisedDetector
        detector = SupervisedDetector(config.path("model_path"))
    else:
        raise ValueError("detector must be rule or supervised")
    results = detector.predict(features)
    save_table(config, "detections", results)
    save_table(config, "combined_signals", combined_signals(results))
    write_json(config.output / "detection_manifest.json", {
        "config_version": config.version, "engine": config.values["detector"], "rows": len(results),
        "scored_rows": int(results.score.notna().sum()), "score_definition": "Activity decline; never an individual isolation probability."})
    print(f"Detected: {int(results.score.notna().sum()):,} scored rows", flush=True)
    return results
