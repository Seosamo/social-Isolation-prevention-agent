import csv
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from activity_agent.isolation.cohorts import attach_structure, build_cohorts
from activity_agent.common.storage import load_table
from activity_agent.activity.features import build_features
from activity_agent.data.ingest import AGE_COLUMNS, DAY_COLUMNS, TIME_COLUMNS, preprocess
from activity_agent.isolation.queries import IsolationEvidenceAgent, IsolationQueryTools
from activity_agent.isolation.config import destination, load, settings
from activity_agent.isolation.labeling import IsolationLabelPolicy, isolation_reviews, label_cohorts, review_isolation
from activity_agent.activity.labels import label as activity_label
from activity_agent.isolation.structural import location, normalize_source, load_structural_sources
from activity_agent.isolation.training import (MODEL_INPUTS, evaluate_model, predict_isolation, read_isolation_model,
                                     train_model, validate_training_readiness)


def write_csv(path, rows, encoding="utf-8-sig", delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding=encoding) as stream:
        csv.writer(stream, delimiter=delimiter).writerows(rows)


@pytest.fixture
def isolation_pipeline(source_data):
    cfg = source_data
    cfg.values["isolation"]["geo_level"] = "sigungu"
    cfg.values["isolation"]["method"] = "supervised"
    cfg.values["isolation"].update(mapping_mode="verified", region_codes=[],
                                     block_mapping_file="references/block_geography.csv",
                                     min_reference_regions=5, min_training_regions=3)
    cfg.values["isolation"].pop("holdout_regions", None)
    data = cfg.path("data_dir")
    districts = [f"테스트{i}구" for i in range(6)]
    geography = [["geo_level", "geo_code", "province", "district", "dong", "parent_code", "reference_year", "source_url", "verified"]]
    mapping = [["block_code", "geo_level", "geo_code", "reference_year", "source_url", "verified"]]
    for i, district in enumerate(districts):
        geography.append(["sigungu", f"s{i}", "서울특별시", district, "", "11", "2025", "https://test.go.kr/synthetic", "true"])
        mapping.append([f"b{i}", "sigungu", f"s{i}", "2025", "https://test.go.kr/synthetic", "true"])
    write_csv(cfg.root / "references/geography.csv", geography)
    write_csv(cfg.root / "references/block_geography.csv", mapping)
    for offset in range(6):
        month = f"2025{7+offset:02d}"
        for family, columns in [("age", AGE_COLUMNS), ("time", TIME_COLUMNS), ("wkdy", DAY_COLUMNS)]:
            rows = [["STD_YM", "BLOCK_CD", "X_COORD", "Y_COORD", *columns]]
            for i in range(6):
                value = [100, 100, 50, 25, 10, 5][offset] if i < 3 else 100
                for x in (100, 200):
                    rows.append([month, f"b{i}", x, 500, *([value] * len(columns))])
            write_csv(data / f"flow_{family}_pop_{month}.csv", rows, delimiter="|")
    single = [["행정구역별(시군구)", "연령별", "2025", "2025", "2025"], ["행정구역별(시군구)", "연령별", "1인가구", "남자", "여자"]]
    # Complete explicit region names, synthetic values only; all age and sex totals reconcile.
    for district, count in zip(districts, [300, 280, 260, 60, 40, 20]):
        name = "서울특별시 " + district
        single.append([name, "합계", 1000, 500, 500])
        for age in ("20~24세", "25~29세"):
            single.append([name, age, (500-count)//2+150, (500-count)//2, 150])
        for age in ("50~54세", "55~59세"):
            single.append([name, age, count//2+100, count//2, 100])
    write_csv(data / "single.csv", single, "cp949")
    death = [["시도별", "2024", "2024", "2024"], ["시도별", "고독사", "인구10만명당", "사망자비중"], ["서울특별시", 784, 8.4, 1.49]]
    write_csv(data / "deaths.csv", death, "cp949")
    ratio = [["행정구역별(1)", "행정구역별(2)", "행정구역별(3)", "2025", "2025", "2025"], ["시도", "구", "동", "비율", "65세이상", "일반가구"]]
    ratio.extend([["서울특별시", d, "소계", 10, 100, 1000] for d in districts])
    write_csv(data / "ratio.csv", ratio, "cp949")
    sources = [["source_id", "path", "reference_period", "available_at", "url"],
               ["single_households", "data/single.csv", "2025", "", "https://test.go.kr/synthetic"],
               ["lonely_deaths", "data/deaths.csv", "2024", "2025-11-27", "https://test.go.kr/synthetic"],
               ["elderly_households", "data/ratio.csv", "2025", "", "https://test.go.kr/synthetic"]]
    write_csv(cfg.root / "references/structural_sources.csv", sources)
    preprocess(cfg)
    build_features(cfg)
    activity_label(cfg)
    load_structural_sources(cfg)
    build_cohorts(cfg)
    label_cohorts(cfg)
    validate_training_readiness(cfg)
    (cfg.root / "config/default.json").write_text(json.dumps(cfg.values), encoding="utf-8")
    return cfg


def test_normalizer_detects_actual_geography_and_preserves_missing():
    meta = dict(source_id="single_households", reference_period="2025", available_at="", url="", source_hash="fixture")
    rows = [["region", "age", "2025", "2025", "2025"], ["region", "age", "all", "male", "female"],
            ["서울특별시", "합계", "100", "50", "50"], ["서울특별시", "20~24세", "40", "20", "20"],
            ["서울특별시", "25~29세", "60", "30", "30"]]
    frame = normalize_source(meta, rows)
    assert set(frame.geo_level) == {"sido"}
    share = frame.loc[frame.sex.eq("남성") & frame.metric.eq("single_household_share")].iloc[0]
    assert share.value == 0.5  # Composition among all single households, not population prevalence.
    rows[-1][-1] = "-"
    invalid = normalize_source(meta, rows)
    assert invalid.loc[invalid.sex.eq("여성"), "quality"].ne("ok").all()
    rows = [["region", "age", "2025", "2025", "2025"], ["region", "age", "all", "male", "female"],
            ["서울특별시", "합계", "100", "50", "50"], ["서울특별시", "20~34세", "100", "50", "50"]]
    assert normalize_source(meta, rows).quality.ne("ok").all()  # A coarse band cannot be silently split into 20s.
    nested = location(["전남광주통합특별시", "광주광역시", "동구"])
    assert nested["geo_level"] == "unresolved" and nested["dong"] == "동구"
    ward = location(["경기도", "수원시", "장안구"])
    assert ward["geo_level"] == "sigungu" and ward["district"] == "수원시 장안구"


def test_sido_pipeline_trains_and_holds_out_whole_provinces(isolation_pipeline):
    from activity_agent.isolation.config import save
    from activity_agent.isolation.structural import PROVINCES

    cfg = isolation_pipeline
    cfg.values["isolation"].update(geo_level="sido", holdout_regions=["s5"])
    registry_path = cfg.root / "references/geography.csv"
    registry = pd.read_csv(registry_path, dtype=str).fillna("")
    provinces = dict(zip(registry.geo_code, PROVINCES[:6]))
    registry["geo_level"] = "sido"
    registry["province"] = registry.geo_code.map(provinces)
    registry[["district", "dong", "parent_code"]] = ""
    registry.to_csv(registry_path, index=False)
    mapping_path = cfg.root / "references/block_geography.csv"
    mapping = pd.read_csv(mapping_path, dtype=str)
    mapping["geo_level"] = "sido"
    mapping.to_csv(mapping_path, index=False)
    structural = load(cfg, "structural")
    selected = structural.geo_code.isin(provinces)
    structural.loc[selected, "province"] = structural.loc[selected, "geo_code"].map(provinces)
    structural.loc[selected, "geo_level"] = "sido"
    structural.loc[selected, ["district", "dong"]] = ""
    save(cfg, "structural", structural)

    cohorts = build_cohorts(cfg)
    assert set(cohorts.geo_level) == {"sido"}
    assert (cohorts.district_code == cohorts.geo_code).all()
    assert cohorts.social_consumption_decline.isna().all()  # District cards cannot represent a province.
    assert cohorts.structure_quality.eq("ok").any()
    label_cohorts(cfg)
    readiness = validate_training_readiness(cfg)
    assert readiness["training_ready"]
    assert readiness["evaluation_region_level"] == "sido"
    assert readiness["independent_evaluation_regions"] == 6
    evaluation = train_model(cfg)
    model = read_isolation_model(cfg)
    assert "s5" not in model["training_regions"]
    assert model["holdout_regions"] == ["s5"]
    assert evaluation["splits"]["test"]["weak_label_proxy"]["heldout_region_count"] == 1
    predictions = predict_isolation(cfg)
    assert predictions.loc[predictions.month.le("202510"), "score"].isna().all()
    assert predictions.score.notna().any()
    tools = IsolationQueryTools(cfg)
    candidates = tools.candidates(region="서울", month="202512")
    assert not candidates.empty
    assert set(candidates.province) == {"서울특별시"}


def test_scoped_consensus_mapping_excludes_unselected_and_unresolved_blocks(isolation_pipeline):
    from activity_agent.common.storage import fingerprint, write_json
    from activity_agent.isolation.training import supported_by_model

    cfg = isolation_pipeline
    path = cfg.root / "references/block_geography.csv"
    mapping = pd.read_csv(path, dtype=str)
    mapping["verified"] = "false"
    mapping["geo_level"] = "sido"
    mapping["assignment_status"] = "candidate_crs_agree"
    mapping.loc[mapping.geo_code.eq("s1"), "assignment_status"] = "unresolved_block"
    unresolved_in_scope = mapping.geo_code.eq("s2")
    mapping.loc[unresolved_in_scope, "geo_code"] = "s0"
    mapping.loc[unresolved_in_scope, "assignment_status"] = "unresolved_block"
    mapping.to_csv(path, index=False)
    registry_path = cfg.root / "references/geography.csv"
    registry = pd.read_csv(registry_path, dtype=str).fillna("")
    registry["geo_level"] = "sido"
    registry["province"] = [f"합성시도{i}" for i in range(6)]
    registry[["district", "dong", "parent_code"]] = ""
    registry.to_csv(registry_path, index=False)
    structural = load(cfg, "structural")
    matching = structural.geo_code.str.startswith("s")
    structural.loc[matching, "geo_level"] = "sido"
    from activity_agent.isolation.config import save
    save(cfg, "structural", structural)
    cfg.values["isolation"].update(geo_level="sido", region_codes=["s0", "s3"],
                                   mapping_mode="crs_consensus", min_reference_regions=2,
                                   min_training_regions=2)
    write_json(destination(cfg, "spatial_audit.json"), {
        "candidate_mapping_sha256": fingerprint(path),
        "input_manifest_sha256": fingerprint(cfg.output / "clean_manifest.json")})
    cohorts = build_cohorts(cfg)
    assert set(cohorts.geo_code) == {"s0", "s3"}
    label_cohorts(cfg)
    report = train_model(cfg)
    model = read_isolation_model(cfg)
    assert model["region_codes"] == ["s0", "s3"]
    assert model["mapping_mode"] == "crs_consensus"
    assert all(r["regions"] == 2 for r in model["policy"]["references"])
    assert len(report["splits"]["test"]["weak_label_proxy"]["by_region"]) == 2
    outsider = cohorts.loc[cohorts.month.eq("202512")].copy()
    outsider["geo_code"] = "s2"
    assert not supported_by_model(model, outsider).any()
    selection = json.loads(destination(cfg, "spatial_selection.json").read_text(encoding="utf-8"))
    assert selection["selected_block_year_rows"] == 2
    assert selection["excluded_block_year_rows"] == 4
    # A revised candidate file must go through the spatial audit again.
    mapping.to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Spatial candidate mapping changed"):
        build_cohorts(cfg)


def test_quantile_fit_deduplicates_months_and_ignores_future(isolation_pipeline):
    cfg = isolation_pipeline
    cohorts = load(cfg, "cohorts")
    train = cohorts.loc[cohorts.month.le("202510")]
    first = IsolationLabelPolicy(settings(cfg)).fit_reference(train)
    duplicated = pd.concat([train, train, train])
    second = IsolationLabelPolicy(settings(cfg)).fit_reference(duplicated)
    assert first.to_dict() == second.to_dict()
    assert all(r["unique_observations"] == r["regions"] for r in first.references)
    future = cohorts.loc[cohorts.month.eq("202512")].copy()
    future["structure_value"] = 999
    with pytest.raises(ValueError, match="training months"):
        IsolationLabelPolicy(settings(cfg)).fit_reference(pd.concat([train, future]))


def test_cohort_scope_and_label_abstentions(isolation_pipeline):
    cfg = isolation_pipeline
    cohorts, labels = load(cfg, "cohorts"), load(cfg, "labels")
    assert not cohorts.duplicated(["geo_level", "geo_code", "sex", "age_band", "month"]).any()
    assert set(labels.label) == {-1, 0, 1}
    assert labels.loc[labels.age_band.eq("10"), "label"].eq(-1).all()
    assert labels.loc[labels.age_band.eq("60+"), "label"].eq(-1).all()  # 65+ ratio is context only.
    assert cohorts.loc[cohorts.age_band.eq("60+"), "elderly_household_ratio_context"].eq(0.1).all()
    assert cohorts.social_consumption_decline.isna().all()  # Card absence is not zero.
    references = IsolationLabelPolicy.from_dict(json.loads(destination(cfg, "label_reference.json").read_text(encoding="utf-8")))
    stable = cohorts.loc[cohorts.month.eq("202512") & cohorts.geo_code.eq("s5") & cohorts.sex.eq("남성") & cohorts.age_band.eq("50")].copy()
    stable[["day_quality", "previous_day_quality", "weekend_quality", "previous_weekend_quality"]] = "missing"
    assert references.label(stable).label.eq(-1).all()


def test_train_evaluate_review_and_preserve_original_task(isolation_pipeline):
    cfg = isolation_pipeline
    original_labels = load_table(cfg, "weak_labels")
    legacy_path = cfg.path("model_path")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_bytes(b"synthetic-legacy-model")
    report = train_model(cfg)
    assert report["splits"]["test"]["independent_review"]["rows"] == 0
    assert report["splits"]["test"]["weak_label_proxy"]["model"]["rows"] > 0
    model = read_isolation_model(cfg)
    assert not any("death" in name or "label" in name or "state" in name for name in MODEL_INPUTS)
    assert model["task_id"] == "regional_isolation_suspicion_v1"
    assert legacy_path.read_bytes() == b"synthetic-legacy-model"
    pd.testing.assert_frame_equal(original_labels, load_table(cfg, "weak_labels"))
    prediction = predict_isolation(cfg)
    assert prediction.loc[prediction.month.le("202510"), "score"].isna().all()
    assert prediction.loc[prediction.month.gt("202510"), "score"].notna().any()
    assert prediction.loc[prediction.sex.eq("여성"), "score"].isna().all()  # No eligible female training profile in this fixture.
    row = load(cfg, "labels").query("month == '202512' and label >= 0").iloc[0]
    review_isolation(cfg, row.cohort_id, int(row.label), "tester", "independent synthetic evidence", True)
    report = evaluate_model(cfg)
    assert report["splits"]["test"]["independent_review"]["model"]["rows"] == 1
    review_isolation(cfg, row.cohort_id, -1, "tester2", "retract synthetic evidence", False)
    assert isolation_reviews(cfg, True).empty
    label_cohorts(cfg)
    with sqlite3.connect(cfg.output / "reviews.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM label_reviews").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0


def test_deaths_change_only_background_not_cohorts_or_labels(isolation_pipeline):
    cfg = isolation_pipeline
    before, labels_before = load(cfg, "cohorts"), load(cfg, "labels")
    rows = [["시도별", "2024", "2024", "2024"], ["시도별", "고독사", "인구10만명당", "사망자비중"], ["서울특별시", 9999, 999, 99]]
    write_csv(cfg.root / "data/deaths.csv", rows, "cp949")
    load_structural_sources(cfg)
    after = build_cohorts(cfg)
    labels_after = label_cohorts(cfg)
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(labels_before, labels_after)
    row = labels_after.query("month == '202512' and label == 1").iloc[0]
    background = IsolationQueryTools(cfg).background(row.cohort_id)
    assert background.loc[background.metric.eq("lonely_deaths_per_100k"), "value"].iloc[0] == 999
    answer = IsolationEvidenceAgent(cfg).answer("2025년 12월 50대 남성")
    assert "라벨·모델 입력에는 사용하지 않았습니다" in answer and "999.0" in answer


def test_as_of_rejects_unknown_or_future_publication_dates(isolation_pipeline):
    cfg = isolation_pipeline
    cohorts, structural = load(cfg, "cohorts"), load(cfg, "structural")
    base = cohorts.drop(columns=[c for c in cohorts if c.startswith("structure_") or c.endswith("_context")])
    assert attach_structure(base, structural, "as_of").structure_value.isna().all()
    structural["available_at"] = "2026-01-01"
    assert attach_structure(base, structural, "as_of").structure_value.isna().all()
    # An annual 2025 statistic is not assumed to describe July before year end.
    structural["available_at"] = "2025-01-01"
    joined = attach_structure(base, structural, "as_of")
    assert joined.loc[joined.month.lt("202512"), "structure_value"].isna().all()


def test_holdout_is_excluded_from_fit_and_group_counts_are_real(isolation_pipeline):
    cfg = isolation_pipeline
    cfg.values["isolation"]["holdout_districts"] = ["s5"]
    label_cohorts(cfg)
    train_model(cfg)
    model = read_isolation_model(cfg)
    assert "s5" not in model["training_districts"]
    assert all(r["regions"] == 5 for r in model["policy"]["references"])
    report = evaluate_model(cfg)
    assert report["splits"]["test"]["weak_label_proxy"]["heldout_district_count"] == 1
    cfg.values["isolation"]["min_training_regions"] = 10
    readiness = validate_training_readiness(cfg)
    assert not readiness["training_ready"]
    assert readiness["independent_districts"] == 6


def test_isolation_dashboard_loads_and_queries(isolation_pipeline):
    from streamlit.testing.v1 import AppTest
    path = str(isolation_pipeline.root / "config/default.json")
    app = AppTest.from_string(f"from app import main\nmain({path!r})").run(timeout=30)
    app.radio(key="analysis_task").set_value("고립 의심").run(timeout=30)
    assert not app.exception
    next(w for w in app.text_input if w.label == "고립 의심 질문").set_value("2025년 12월 고립 의심")
    next(w for w in app.button if w.label == "고립 근거 조회").click().run(timeout=30)
    assert not app.exception


def test_dong_training_uses_parent_district_holdout(isolation_pipeline):
    cfg = isolation_pipeline
    registry_path = cfg.root / "references/geography.csv"
    blocks_path = cfg.root / "references/block_geography.csv"
    registry, blocks = pd.read_csv(registry_path, dtype=str).fillna(""), pd.read_csv(blocks_path, dtype=str).fillna("")
    dongs = registry.copy()
    dongs["parent_code"] = dongs.geo_code
    dongs["geo_code"] = [f"d{i}" for i in range(6)]
    dongs["geo_level"] = "dong"
    dongs["dong"] = [f"테스트{i}동" for i in range(6)]
    pd.concat([registry, dongs]).to_csv(registry_path, index=False)
    dong_blocks = blocks.copy()
    dong_blocks["geo_level"] = "dong"
    dong_blocks["geo_code"] = [f"d{i}" for i in range(6)]
    pd.concat([blocks, dong_blocks]).to_csv(blocks_path, index=False)
    raw = pd.read_csv(cfg.root / "data/single.csv", header=None, encoding="cp949", dtype=str).fillna("")
    for i in range(6):
        mask = raw[0].eq(f"서울특별시 테스트{i}구")
        raw.loc[mask, 0] += f" 테스트{i}동"
    raw.to_csv(cfg.root / "data/single.csv", header=False, index=False, encoding="cp949")
    cfg.values["isolation"].update(geo_level="dong", holdout_districts=["s5"])
    load_structural_sources(cfg)
    cohorts = build_cohorts(cfg)
    assert set(cohorts.geo_level) == {"dong"}
    assert set(cohorts.district_code) == {f"s{i}" for i in range(6)}
    assert cohorts.social_consumption_decline.isna().all()
    label_cohorts(cfg)
    train_model(cfg)
    model = read_isolation_model(cfg)
    assert "s5" not in model["training_districts"]
    assert model["geo_level"] == "dong"


def test_no_mapping_produces_readiness_report_without_fake_labels(isolation_pipeline):
    cfg = isolation_pipeline
    mapping = cfg.root / "references/block_geography.csv"
    mapping.write_text("block_code,geo_level,geo_code,reference_year,source_url,verified\n", encoding="utf-8")
    assert build_cohorts(cfg).empty
    assert label_cohorts(cfg).empty
    report = validate_training_readiness(cfg)
    assert not report["training_ready"] and report["cohort_rows"] == 0
    with pytest.raises(ValueError, match="training blocked"):
        train_model(cfg)


def test_detector_selection_preserves_label_policy_settings(config):
    # Contraction exploration compares these settings to the saved label policy.
    before = settings(config)
    config.values['isolation'].update(model_id='IF_F2', model_path='models/IF_F2.pkl',
                                     flow_cutoffs_path='models/flow_cutoffs.csv')
    assert settings(config) == before
