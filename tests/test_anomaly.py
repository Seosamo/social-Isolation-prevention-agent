import json

import numpy as np
import pandas as pd
import pytest

from activity_agent.isolation import anomaly
from activity_agent.isolation.config import destination
from activity_agent.isolation.queries import IsolationEvidenceAgent


def test_agent_evidence_contract_and_backend_handoff(trained_anomaly):
    from activity_agent.isolation.handoff import candidate_evidence
    from activity_agent.llm.vllm import VLLMBackend, VLLMSettings
    cfg, _, _ = trained_anomaly
    payload = candidate_evidence(cfg, month="202512", region="11", limit=1)
    assert payload["status"] == "OK"
    assert payload["summary"]["returned_rows"] == 1
    assert payload["summary"]["candidate_rows"] == 1
    assert not payload["summary"]["truncated"]
    item = payload["candidates"][0]
    assert item["observation"]["geo_code"] == "11"
    assert all(item["decision"][k] for k in ("candidate", "score_threshold_met", "group_decline_met", "regional_decline_met"))
    json.dumps(payload, allow_nan=False)
    assert candidate_evidence(cfg, month="202601")["status"] == "NO_DATA"
    assert candidate_evidence(cfg, month="202510")["status"] == "NOT_EVALUABLE"
    assert candidate_evidence(cfg, month="202511", region="11")["status"] == "NO_CANDIDATES"
    with pytest.raises(ValueError): candidate_evidence(cfg, limit=-1)
    with pytest.raises(ValueError): candidate_evidence(cfg, month="202513")

    class Client:
        settings = VLLMSettings()
        def __init__(self): self.messages = []
        def chat(self, system, user, **kwargs):
            self.messages.append(json.loads(user))
            return json.dumps({"month": "202512", "region": "11"}) if len(self.messages) == 1 else "관측된 활동 위축 후보입니다."
    client = Client()
    agent = IsolationEvidenceAgent(cfg, VLLMBackend(cfg, client))
    agent.answer("활동 위축 후보")
    assert client.messages[1]["structured_evidence"]["schema_version"] == "activity-anomaly-evidence-v1"
    assert client.messages[1]["structured_evidence"]["filters"]["month"] == "202512"


def sample_features():
    rng = np.random.default_rng(12)
    rows = []
    for month in ("202510", "202511", "202512"):
        for code, province in (("11", "서울특별시"), ("32", "강원특별자치도")):
            for sex in ("남성", "여성"):
                for age in ("20", "50"):
                    for block in range(40):
                        values = rng.normal(.01, .05, 6)
                        if month != "202510" and block < 4:
                            values = np.array([.7, .5, .6, .4, .7, .5])
                        if month != "202510" and block == 4:
                            values = np.array([-.8, -.6, -.7, -.6, -.8, -.7])
                        rows.append(dict(zip(anomaly.INPUTS, values)) | {
                            "block_code": f"{code}-b{block}", "geo_code": code, "province": province,
                            "sex": sex, "age_band": age, "month": month,
                            "baseline_flow": float(100 + block), "previous_baseline_flow": float(100 + block),
                            "quality": "ok", "previous_quality": "ok", "day_quality": "ok",
                            "previous_day_quality": "ok", "weekend_quality": "ok", "previous_weekend_quality": "ok"})
    return pd.DataFrame(rows)


@pytest.fixture
def trained_anomaly(config, monkeypatch):
    config.values["isolation"]["anomaly"].update(n_estimators=30, stability_seeds=[42, 43])
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / "clean_manifest.json").write_text('{"skt_files": []}', encoding="utf-8")
    frame = sample_features()
    monkeypatch.setattr(anomaly, "local_features", lambda cfg: frame.copy())
    report = anomaly.train_anomaly(config)
    return config, frame, report


def test_label_free_training_scores_only_future_and_filters_growth(trained_anomaly):
    cfg, frame, report = trained_anomaly
    model = anomaly.read_model(cfg)
    assert anomaly.model_path(cfg) == cfg.root / "activity_agent/model/isolation_forest.pkl"
    assert anomaly.model_path(cfg).with_suffix(".json").is_file()
    assert not destination(cfg, "anomaly/models/isolation_forest.pkl").exists()
    assert anomaly.relocated_model_path(destination(cfg, "anomaly/models/isolation_forest.pkl")) == anomaly.model_path(cfg)
    assert not destination(cfg, "labels.parquet").exists()
    assert not destination(cfg, "structural.parquet").exists()
    assert report["accuracy_evaluated"] is False
    assert anomaly.anomaly_readiness(cfg)["training_ready"] is True
    assert report["training_months"] == ["202510"]
    assert not set(anomaly.INPUTS) & {"geo_code", "sex", "age_band", "label", "structure_value"}
    scored = anomaly.score_frame(model, frame)
    assert scored.loc[scored.month.eq("202510"), "anomaly_percentile"].isna().all()
    assert not scored.loc[scored.month.eq("202510"), "candidate"].any()
    assert scored.loc[scored.month.eq("202512"), "candidate"].any()
    assert not scored.loc[scored.activity_decline.lt(0), "candidate"].any()
    assert scored.anomaly_percentile.dropna().between(0, 100).all()
    assert model["score_cutoff"] == np.quantile(model["reference_scores"], .95)
    candidates = anomaly.load_candidates(cfg, region="서울", month="202512")
    assert not candidates.empty and set(candidates.geo_code) == {"11"}
    answer = IsolationEvidenceAgent(cfg).answer("2025년 12월 서울 활동 이상")
    assert "백분위" in answer and "확률이 아닙니다" in answer


def test_future_data_does_not_change_fit_and_small_flow_is_not_scored(trained_anomaly, monkeypatch):
    cfg, frame, _ = trained_anomaly
    first = anomaly.read_model(cfg)
    changed = frame.copy()
    changed.loc[changed.month.gt("202510"), anomaly.INPUTS] = .99
    changed.loc[changed.month.gt("202510"), "baseline_flow"] = 10000.
    monkeypatch.setattr(anomaly, "local_features", lambda cfg: changed.copy())
    anomaly.train_anomaly(cfg)
    second = anomaly.read_model(cfg)
    assert first["model_version"] == second["model_version"]
    assert first["profiles"] == second["profiles"]
    np.testing.assert_array_equal(first["reference_scores"], second["reference_scores"])
    tiny = frame.loc[frame.month.eq("202512")].head(1).copy()
    tiny["baseline_flow"] = .5
    result = anomaly.score_frame(second, tiny)
    assert result.anomaly_percentile.isna().all() and not result.candidate.any()
    outsider = frame.loc[frame.month.eq("202512")].head(1).copy()
    outsider["geo_code"] = "31"
    result = anomaly.score_frame(second, outsider)
    assert result.anomaly_percentile.isna().all() and not result.candidate.any()
    cfg.values["isolation"]["anomaly"]["minimum_decline"] = .2
    with pytest.raises(ValueError, match="settings changed"):
        anomaly.read_model(cfg)


def test_anomaly_dashboard_without_classification_labels(trained_anomaly):
    from streamlit.testing.v1 import AppTest

    cfg, _, _ = trained_anomaly
    path = cfg.root / "config/default.json"
    path.write_text(json.dumps(cfg.values), encoding="utf-8")
    app = AppTest.from_string(f"from app import main\nmain({str(path)!r})").run(timeout=30)
    app.radio(key="analysis_task").set_value("고립 의심").run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="anomaly_month").value == "202512"
    next(w for w in app.text_input if w.label == "활동 위축 이상 질문").set_value("2025년 12월 서울 활동 이상")
    next(w for w in app.button if w.label == "활동 이상 근거 조회").click().run(timeout=30)
    assert not app.exception
