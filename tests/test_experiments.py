import json
import pickle
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from activity_agent.isolation.experiments.features import recompute, jsd, _snapshot, FEATURE_SETS, INPUTS
from activity_agent.isolation.experiments.models import ExperimentModel, Normalizer, split_blocks, direction
from activity_agent.isolation.experiments.injection import inject
from activity_agent.isolation.experiments.runner import _temporal, consolidate
from activity_agent.isolation.experiments.paths import model_directory, migrate_models


def test_model_migration_preserves_files_and_relative_locator(tmp_path):
    run = tmp_path / "artifacts/isolation/experiments/test_run"
    source = run / "models"
    source.mkdir(parents=True)
    (source / "IF_F0.pkl").write_bytes(b"unchanged-model")
    (source / "operating_original.pkl").write_bytes(b"baseline")
    (source / "IF_F0.json").write_text('{}', encoding="utf-8")
    target = migrate_models(run, tmp_path)
    assert target == tmp_path / "activity_agent/model/experiments/test_run"
    assert model_directory(run) == target
    assert (target / "IF_F0.pkl").read_bytes() == b"unchanged-model"
    assert (target / "IF_BASELINE_ORIGINAL.pkl").read_bytes() == b"baseline"
    assert not list(source.glob("*"))
    assert migrate_models(run, tmp_path) == target


def source_rows(n=80, seed=4):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"g0": rng.uniform(40, 130, n), "g1": rng.uniform(70, 130, n), "g2": 100.,
                      "p1": rng.uniform(70, 130, n), "p2": 100., "p3": 100.,
                      "r0": 1000., "r1": 1100., "r2": 1200., "rp1": 1100., "rp2": 1200., "rp3": 1300.,
                      "d0": rng.uniform(80,120,n), "d1":100., "d2":100., "dp1":100., "dp2":100., "dp3":100.,
                      "w0":rng.uniform(80,120,n), "w1":100., "w2":100., "wp1":100., "wp2":100., "wp3":100.})
    for prefix, length in (("h",24),("y",7)):
        for offset in (0,1):
            for i in range(length):
                f[f"{prefix}{offset}_{i}"] = float(i+1)
    f = recompute(f)
    f["block_code"] = [f"b{i}" for i in range(n)]
    f["geo_code"] = ["11" if i%2==0 else "32" for i in range(n)]
    f["province"] = np.where(f.geo_code.eq("11"), "서울특별시", "강원특별자치도")
    f["sex"], f["age_band"], f["month"] = "남성", "20", "202510"
    f["row_id"] = f.block_code + "|남성|20|" + f.month
    f["baseline_flow"] = (f.g1+f.g2)/2
    f["new_quality"], f["common_eligible"] = True, True
    return f


def test_feature_formulas_and_jsd_units():
    f = source_rows(1)
    f[["g0","g1","g2","r0","r1"]] = [60.,80.,100.,900.,1000.]
    out = recompute(f)
    assert out.cumulative_decline.iloc[0] == pytest.approx(.4)
    assert out.decline_acceleration.iloc[0] == pytest.approx(.05)
    assert out.persistent_decline.iloc[0] == 1
    assert out.relative_decline.iloc[0] == pytest.approx(.15)
    assert out.share_change.iloc[0] == pytest.approx(60/900-80/1000)
    assert jsd([[1,0]], [[0,1]])[0] == pytest.approx(1)
    assert jsd([[1,2]], [[10,20]])[0] == pytest.approx(0)
    assert np.isnan(jsd([[0,0]], [[1,2]])[0])
    f["g2"] = 0
    assert recompute(f).cumulative_decline.isna().all()


def test_injection_is_paired_and_preserves_distribution_controls():
    f = source_rows(4)
    original = f.copy(deep=True)
    change = inject(f, "S1", .3)
    assert np.allclose(change.g0, f.g0*.7)
    assert np.allclose(change.r0, f.r0-f.g0*.3)
    pd.testing.assert_frame_equal(f, original)
    scaled = inject(f, "NC2", .3)
    assert np.allclose(scaled.hourly_jsd, f.hourly_jsd)
    assert np.allclose(scaled.weekly_jsd, f.weekly_jsd)
    shifted = inject(f, "S8", .5)
    assert np.allclose(shifted[[f"h0_{i}" for i in range(24)]].sum(axis=1), f[[f"h0_{i}" for i in range(24)]].sum(axis=1))
    assert (shifted.hourly_jsd > f.hourly_jsd).all()
    persistent=inject(f,"S4",.2)
    assert np.allclose(persistent.g0/persistent.g1,.8)
    assert np.allclose(persistent.g1/persistent.g2,.8)
    sudden=inject(f,"S5",.4)
    assert np.allclose(sudden.activity_decline,.4)
    assert np.allclose(sudden.previous_activity_decline,0.)
    growth = f.copy()
    growth["g0"] = 120.
    growth = inject(growth,"S6",.3)
    assert not direction(growth).any()


def test_common_shock_and_selective_group_intervention_are_distinct():
    f = source_rows(1)
    f[["g0","g1","r0","r1"]] = [80.,100.,800.,1000.]
    f = recompute(f)
    assert inject(f,"S7",.3).relative_decline.iloc[0] == pytest.approx(0)
    assert inject(f,"S3",.35).relative_decline.iloc[0] > 0


def test_october_block_split_and_train_only_normalization():
    f = source_rows()
    repeated = pd.concat([f, f.assign(sex="여성"), f.assign(month="202512",g0=1e9)], ignore_index=True)
    train, cal = split_blocks(repeated)
    assert not train & cal and train | cal == set(f.block_code)
    assert split_blocks(f) == (train,cal)
    fit = f.loc[f.block_code.isin(train)]
    scaler = Normalizer(INPUTS).fit(fit)
    first = scaler.transform(fit)
    repeated.loc[repeated.month.eq("202512"),INPUTS] = 1e6
    np.testing.assert_array_equal(scaler.transform(fit),first)
    assert all(d["training_rows"] <= len(train) for d in scaler.diagnostics)


def test_source_windows_ignore_future_and_require_history():
    from activity_agent.data.ingest import AGE_COLUMNS
    frames={}
    for month,value in (("202508",100.),("202509",200.),("202510",300.)):
        frames[month]=pd.DataFrame({"block_code":["b","b"],"x":[1.,2.],"y":[0.,0.],
                                    **{c:[value,value] for c in AGE_COLUMNS}})
    original=_snapshot(frames,"202510","age",.8)
    frames["202511"]=frames["202510"].assign(**{c:99999. for c in AGE_COLUMNS})
    after=_snapshot(frames,"202510","age",.8)
    for left,right in zip(original[0],after[0]):
        pd.testing.assert_frame_equal(left,right)
    assert _snapshot({"202510":frames["202510"]},"202510","age",.8) is None


@pytest.mark.parametrize("kind",["IF","LOF","KNN","ROBUST_Z"])
def test_calibration_threshold_and_score_direction(kind):
    fit, cal = source_rows(140,1), source_rows(50,2)
    model = ExperimentModel(kind,"F0").fit(fit,cal)
    expected = model.raw_score(cal)
    assert model.threshold == pytest.approx(np.quantile(expected,.95))
    extreme = cal.head(2).copy()
    extreme[INPUTS] = 20.
    assert model.raw_score(extreme).mean() > model.raw_score(cal).mean()
    if kind == "LOF":
        assert model.estimators["all"].novelty
    again = ExperimentModel(kind,"F0").fit(fit,cal)
    np.testing.assert_allclose(again.raw_score(cal),expected)


def test_temporal_followup_is_not_a_long_term_label(tmp_path):
    (tmp_path/"data").mkdir()
    f = pd.concat([source_rows(4).assign(month="202511",candidate=True,raw_score=.5,percentile=99),
                   source_rows(4).assign(month="202512",candidate=False,raw_score=.4,percentile=80)], ignore_index=True)
    _temporal(tmp_path,"test",f)
    result = pd.read_csv(tmp_path/"data/temporal_test.csv")
    assert result.rows.sum()==4


def test_temporal_comparison_uses_december_matched_lags(tmp_path):
    (tmp_path/"data").mkdir()
    november=source_rows(4).assign(month="202511",candidate=True,g0=9999.)
    december=source_rows(4).assign(month="202512",candidate=False,g0=[50.,95.,85.,100.],g1=[80.,55.,70.,100.],g2=100.)
    _temporal(tmp_path,"same_grid",pd.concat([november,december],ignore_index=True))
    result=pd.read_csv(tmp_path/"data/temporal_same_grid.csv")
    assert set(result.type)=={"지속 감소","회복","반등","정체·부분 회복"}


def test_offline_report_pages_and_embedded_candidate_values(tmp_path):
    from activity_agent.isolation.experiments.report import render_report, PAGES
    from activity_agent.common.storage import write_json

    for name in ("data","models","config"):
        (tmp_path/name).mkdir()
    fit, cal = source_rows(60,3),source_rows(20,4)
    model = ExperimentModel("IF","F0").fit(fit,cal)
    with (tmp_path/"models/IF_F0.pkl").open("wb") as stream:
        pickle.dump(model,stream)
    f = pd.concat([fit,cal.assign(month="202511")],ignore_index=True)
    f.to_parquet(tmp_path/"data/features.parquet",index=False)
    fit[FEATURE_SETS["F4"]].corr().to_csv(tmp_path/"data/feature_correlations.csv")
    pd.DataFrame([dict(month="202510",geo_code="11",sex="남성",age_band="20",observations=60,common_valid=60)]).to_csv(tmp_path/"data/dataset_counts.csv",index=False)
    predicted = model.predict(cal.assign(month="202511")).assign(model_id="IF_F0",unit="집단")
    predicted.to_parquet(tmp_path/"data/predictions_IF_F0.parquet",index=False)
    write_json(tmp_path/"data/metrics_IF_F0.json",dict(model_id="IF_F0",fit_seconds=1.,inference_seconds=.1,operating_row_jaccard=.5))
    pd.DataFrame([dict(model_id="IF_F0",geo_code="11",province="서울특별시",month="202511",candidate_rate=.2,candidate_rows=4,candidate_blocks=4,eligible_rows=20)]).to_csv(tmp_path/"data/regional_IF_F0.csv",index=False)
    write_json(tmp_path/"data/source_diagnostics.json",{"source":"synthetic unit-test fixture"})
    write_json(tmp_path/"data/status.json",{"IF_F0":{"experiment":"IF_F0","status":"COMPLETE"}})
    write_json(tmp_path/"run_manifest.json",{"run_id":"synthetic-test"})
    write_json(tmp_path/"config/experiment_config.json",{"seed":42})
    consolidate(tmp_path)
    migrate_models(tmp_path, tmp_path)
    render_report(tmp_path)
    for page in PAGES:
        content=(tmp_path/f"{page}.html").read_text(encoding="utf-8")
        assert '<html lang="ko">' in content
        assert '<script src=' not in content
        assert 'plotly.js' in content
    explorer=(tmp_path/"candidate_explorer.html").read_text(encoding="utf-8")
    assert 'DecompressionStream' in explorer and 'candidateSearch' in explorer
    assert (tmp_path/"data/candidate_comparison.csv").exists()
    # Consolidation must not recursively ingest its previous output on resume.
    before=pd.read_csv(tmp_path/"data/regional_metrics.csv")
    consolidate(tmp_path)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/"data/regional_metrics.csv"),before)
