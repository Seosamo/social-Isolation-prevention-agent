import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from activity_agent.common.storage import load_table, save_table
from activity_agent.common.config import Config
from activity_agent.activity.detection import RuleDetector, detect, weak_labels
from activity_agent.data.aggregation import aggregate_skt, compare_skt_window, entropy
from activity_agent.activity.features import build_features
from activity_agent.data.ingest import AGE_COLUMNS, DAY_COLUMNS, TIME_COLUMNS, preprocess, read_skt
from activity_agent.activity.labels import add_review, label, latest_reviews
from activity_agent.activity.training import SupervisedDetector, train
from activity_agent.common.evaluation import metrics, temporal_split
from activity_agent.activity.queries import EvidenceAgent, QueryTools


def grid(month, values):
    return pd.DataFrame([[month, "b", x, 0, *([v] * len(AGE_COLUMNS))] for x, v in values],
                        columns=["month", "block_code", "x", "y", *AGE_COLUMNS])


def test_common_grid_prevents_missing_coordinate_false_decline():
    frames = {"202507": grid("202507", [(1, 100), (2, 100)]),
              "202508": grid("202508", [(1, 100), (2, 100)]),
              "202509": grid("202509", [(1, 100)])}
    result = compare_skt_window(frames, "202509", "age", 0.8)
    assert result.activity_decline.eq(0).all()
    assert result.all_decline.eq(0.5).all()
    assert result.coverage.eq(0.5).all()
    assert result.quality.eq("low_coverage").all()


def test_missing_calendar_month_and_zero_baseline():
    frames = {"202507": grid("202507", [(1, 100)]), "202509": grid("202509", [(1, 10)])}
    assert compare_skt_window(frames, "202509", "age", 0.8).quality.eq("insufficient_history").all()
    frames["202508"] = grid("202508", [(1, 0)])
    frames["202507"] = grid("202507", [(1, 0)])
    result = compare_skt_window(frames, "202509", "age", 0.8)
    assert result.activity_decline.isna().all()
    assert result.quality.eq("zero_baseline").all()


def test_entropy_and_time_partitions():
    assert entropy([5, 5]) == pytest.approx(math.log(2))
    assert entropy([10, 0]) == 0
    assert np.isnan(entropy([0, 0]))
    frame = pd.DataFrame([["b", *([1] * 24)]], columns=["block_code", *TIME_COLUMNS])
    result = aggregate_skt(frame, "time").iloc[0]
    assert result.daytime == 9 and result.evening == 4 and result.morning == 3 and result.night == 8
    assert result.night_ratio == pytest.approx(8 / 24)
    days = pd.DataFrame([["b", *([10] * 5), 20, 20]], columns=["block_code", *DAY_COLUMNS])
    assert aggregate_skt(days, "wkdy").iloc[0].weekend_ratio == 2


def test_invalid_duplicate_and_conflicting_rows(tmp_path):
    raw = grid("202507", [(1, 10), (1, 10), (2, 20), (2, 25), (3, -1)])
    raw.columns = ["STD_YM", "BLOCK_CD", "X_COORD", "Y_COORD", *AGE_COLUMNS]
    path = tmp_path / "flow_age_pop_202507.csv"
    raw.to_csv(path, sep="|", index=False)
    clean, bad, profile = read_skt(path, "age")
    assert len(clean) == 1 and len(bad) == 3
    assert profile["exact_duplicates_removed"] == 1
    assert profile["rows_read"] == profile["rows_clean"] + profile["quarantined"] + profile["exact_duplicates_removed"]


def test_pipeline_scopes_card_conservation_and_labels(complete_pipeline):
    cfg = complete_pipeline
    features = load_table(cfg, "features")
    assert not features.feature_id.duplicated().any()
    assert features.loc[features.signal.str.endswith("pattern"), "sex"].eq("전체").all()
    assert features.loc[features.month.isin(["202507", "202508"]), "quality"].eq("insufficient_history").all()
    assert features.region.fillna("").eq("").sum() > 0
    card = load_table(cfg, "card_monthly")
    sample = card.loc[card.month.eq("202507") & card.region.eq("서울 강남구") & card.age.eq("60+")].iloc[0]
    assert sample["count"] == 400  # 60s + 70s, 2 industries; no card2 addition or corporate leakage.
    assert sample.diversity == pytest.approx(math.log(2))
    labels = load_table(cfg, "weak_labels")
    pos = labels.loc[labels.signal.eq("mobility") & labels.area_id.eq("b1") & labels.month.eq("202511")]
    assert pos.label.eq(1).all()
    stable = labels.loc[labels.signal.eq("mobility") & labels.area_id.eq("b2") & labels.month.eq("202510")]
    assert stable.label.eq(0).all()
    assert labels.loc[labels.signal.str.endswith("pattern"), "label"].eq(-1).all()
    assert load_table(cfg, "combined_signals").empty


def test_reviews_survive_rerun_and_latest_review_supersedes(complete_pipeline):
    cfg = complete_pipeline
    fid = load_table(cfg, "features").query("signal == 'mobility'").iloc[0].feature_id
    add_review(cfg, fid, 1, "reviewer", "external survey evidence", True)
    add_review(cfg, fid, -1, "reviewer2", "insufficient evidence", False)
    label(cfg)
    label(cfg)
    assert latest_reviews(cfg).iloc[0].label == -1
    assert latest_reviews(cfg, True).empty
    with sqlite3.connect(cfg.output / "reviews.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM weak_labels").fetchone()[0] == len(load_table(cfg, "features"))


def test_temporal_leakage_and_model_swap(complete_pipeline):
    cfg = complete_pipeline
    result = train(cfg)
    assert result["splits"]["test"]["independent_review"]["rows"] == 0
    assert result["splits"]["test"]["weak_label_proxy"]["model"]["rows"] > 0
    features = load_table(cfg, "features")
    predicted = SupervisedDetector(cfg.path("model_path")).predict(features)
    assert predicted.loc[predicted.signal.str.endswith("pattern"), "engine"].eq("rule").all()
    assert predicted.loc[predicted.signal.eq("mobility") & predicted.quality.eq("ok") & predicted.month.gt("202510"), "engine"].eq("supervised").all()
    assert predicted.loc[predicted.month.le("202510"), "engine"].eq("rule").all()
    with pytest.raises(ValueError, match="train_end"):
        temporal_split(features, "202512", "202511", "202510")
    assert metrics([], [])["rows"] == 0


def test_future_observations_do_not_change_past_features(complete_pipeline):
    cfg = complete_pipeline
    before = load_table(cfg, "features")
    path = cfg.output / "clean/skt_age_202512.parquet"
    frame = pd.read_parquet(path)
    frame[AGE_COLUMNS] *= 100
    frame.to_parquet(path, index=False)
    after = build_features(cfg)
    cols = ["feature_id", "activity_decline", "previous_decline", "quality"]
    pd.testing.assert_frame_equal(before.loc[before.month.lt("202512"), cols].reset_index(drop=True),
                                  after.loc[after.month.lt("202512"), cols].reset_index(drop=True))


def test_verified_mapping_district_aggregation(complete_pipeline):
    cfg = complete_pipeline
    cfg.path("mapping_file").write_text("block_code,region,source_url,verified\nb1,서울 강남구,https://www.gangnam.go.kr/,true\nb2,강원 춘천시,https://www.chuncheon.go.kr/,true\n", encoding="utf-8")
    features = build_features(cfg)
    district = features.loc[features.signal.eq("district_mobility")]
    assert not district.empty
    sample = district.loc[district.month.eq("202511") & district.region.eq("서울 강남구")].iloc[0]
    assert sample.activity_decline == pytest.approx(1 - 10 / 37.5)
    detect(cfg)
    label(cfg)
    assert not load_table(cfg, "combined_labels").empty


def test_agent_does_not_invent_missing_context_or_resources(complete_pipeline):
    tools = QueryTools(complete_pipeline)
    answer = EvidenceAgent(tools).answer("이번 달 50대 남성 활동 위축과 예방 프로그램 추천")
    assert "202512" in answer and "고립 확률이 아닙니다" in answer
    assert "정책 조회" in answer and "시도 또는 시군구를 지정" not in answer
    assert "계산 가능한 결과가 없습니다" in EvidenceAgent(tools).answer("2026-09 위험 지역")
    assert "통합 탐지의 연령 구간" in EvidenceAgent(tools).answer("70대 위험 지역")
    assert tools.find_prevention_resources().empty


def test_missing_social_evidence_abstains(complete_pipeline):
    features = load_table(complete_pipeline, "features")
    features.loc[features.signal.eq("consumption"), "social_consumption_decline"] = np.nan
    labels = weak_labels(features, complete_pipeline)
    assert labels.loc[labels.signal.eq("consumption"), "label"].eq(-1).all()
    assert RuleDetector().predict(features).loc[features.signal.eq("consumption"), "score"].isna().all()


def test_report_does_not_create_duplicate_export(complete_pipeline):
    from activity_agent.activity.reporting import report
    report(complete_pipeline)
    assert (complete_pipeline.output / "detections.parquet").exists()
    assert not (complete_pipeline.output / "candidate_export.parquet").exists()


def test_resources_check_host_age_and_location(complete_pipeline):
    cfg = complete_pipeline
    cols = ["resource_id", "title", "region", "age_min", "age_max", "url", "published_at", "checked_at", "description"]
    rows = [
        ["a", "synthetic fixture", "서울 강남구", "40", "59", "https://www.gangnam.go.kr/test", "2025-01-01", "2025-01-02", "test only"],
        ["b", "spoof", "전국", "10", "99", "https://gangnam.go.kr.evil.example/test", "2025-01-01", "2025-01-02", "test only"],
        ["c", "future", "전국", "10", "99", "https://www.gov.kr/test", "2025-01-01", "2099-01-02", "test only"],
    ]
    pd.DataFrame(rows, columns=cols).to_csv(cfg.path("resources_file"), index=False)
    tools = QueryTools(cfg)
    assert tools.find_prevention_resources("서울 강남구", "50").resource_id.tolist() == ["a"]
    assert tools.find_prevention_resources("강원 춘천시", "50").empty
    assert tools.find_prevention_resources("서울 강남구", "10").empty
    assert tools.find_prevention_resources(None, "50").empty


def test_dashboard_load_and_query(complete_pipeline):
    from streamlit.testing.v1 import AppTest
    path = complete_pipeline.root / "config/default.json"
    script = f"from app import main\nmain({str(path)!r})\n"
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    next(w for w in app.text_input if w.label == "질문").set_value("이번 달 예방 프로그램 추천")
    next(w for w in app.button if w.label == "근거 조회").click()
    app.run(timeout=30)
    assert not app.exception


@pytest.mark.skipif(not (Config.load(Path(__file__).resolve().parents[1] / 'config/default.json').output / 'detections.parquet').exists(),
                    reason="Run-all artifacts are optional for the real-data UI smoke test")
def test_dashboard_current_artifacts():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=60)
    assert not app.exception
    assert app.metric[0].value != "0"
    next(w for w in app.selectbox if w.label == "신호").set_value("consumption")
    app.run(timeout=30)
    assert not app.exception
