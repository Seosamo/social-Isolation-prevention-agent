import numpy as np
import pandas as pd

from activity_agent.isolation.exploration import local_features, summarize_contraction


def test_shares_exclude_invalid_history_and_report_flow_weights():
    rows = []
    for block, decline, past, quality, past_quality, weight in [
            ("a", .3, .3, "ok", "ok", 1),
            ("b", 0, 0, "ok", "ok", 99),
            ("c", .8, .8, "low_coverage", "ok", 50),
            ("d", .4, np.nan, "ok", "insufficient_history", 100)]:
        rows.append(dict(block_code=block, geo_code="11", province="서울특별시", sex="남성", age_band="20",
                         month="202510", activity_decline=decline, previous_activity_decline=past,
                         quality=quality, previous_quality=past_quality,
                         baseline_flow=weight, previous_baseline_flow=weight,
                         daytime_decline=decline, previous_daytime_decline=past,
                         day_quality=quality, previous_day_quality=past_quality,
                         weekend_decline=np.nan, previous_weekend_decline=np.nan,
                         weekend_quality="missing", previous_weekend_quality="missing"))
    signals, summary = summarize_contraction(pd.DataFrame(rows))
    row = summary.iloc[0]
    assert row.observed_blocks == 4
    assert row.single_eligible == 3 and row.single_blocks == 2
    assert row.persistent_eligible == 2 and row.persistent_blocks == 1
    assert row.persistent_share == .5
    assert row.persistent_baseline_flow_share == .01
    assert row.joint_share == .5
    assert row.joint_stable_blocks == 1
    assert not signals.loc[signals.block_code.isin(["c", "d"]), "persistent_support"].any()


def test_no_history_produces_missing_share_not_zero():
    frame = pd.DataFrame([dict(geo_code="11", province="서울특별시", sex="남성", age_band="20", month="202507",
        activity_decline=np.nan, previous_activity_decline=np.nan, quality="insufficient_history", previous_quality=None,
        baseline_flow=np.nan, previous_baseline_flow=np.nan,
        daytime_decline=np.nan, previous_daytime_decline=np.nan, day_quality=None, previous_day_quality=None,
        weekend_decline=np.nan, previous_weekend_decline=np.nan, weekend_quality=None, previous_weekend_quality=None)])
    _, summary = summarize_contraction(frame)
    assert summary.iloc[0].persistent_eligible == 0
    assert pd.isna(summary.iloc[0].persistent_share)
    assert pd.isna(summary.iloc[0].persistent_baseline_flow_share)


def test_local_features_preserve_blocks_scope_and_two_window_history(source_data):
    from activity_agent.data.ingest import preprocess

    cfg = source_data
    cfg.values["isolation"].update(mapping_mode="verified", region_codes=["11"],
                                   block_mapping_file="references/block_geography.csv")
    (cfg.root / "references/geography.csv").write_text(
        "geo_level,geo_code,province,district,dong,parent_code,reference_year,source_url,verified\n"
        "sido,11,서울특별시,,,,2025,https://test.go.kr/synthetic,true\n"
        "sido,32,강원특별자치도,,,,2025,https://test.go.kr/synthetic,true\n", encoding="utf-8")
    (cfg.root / "references/block_geography.csv").write_text(
        "block_code,geo_level,geo_code,reference_year,source_url,verified\n"
        "b1,sido,11,2025,https://test.go.kr/synthetic,true\n"
        "b2,sido,32,2025,https://test.go.kr/synthetic,true\n", encoding="utf-8")
    preprocess(cfg)
    features = local_features(cfg)
    assert set(features.block_code) == {"b1"}
    assert set(features.geo_code) == {"11"}
    signals, summary = summarize_contraction(features)
    assert not signals.loc[signals.month.le("202509"), "persistent_valid"].any()
    assert signals.loc[signals.month.eq("202510"), "joint_support"].all()
    assert summary.loc[summary.month.eq("202510"), "persistent_share"].eq(1).all()
    # The region is not used as a substitute for the block identifier.
    assert not features.duplicated(["block_code", "sex", "age_band", "month"]).any()
