"""F2 deployment must preserve saved scores and reject incompatible inputs."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from activity_agent.isolation import anomaly
from activity_agent.isolation.experiments.features import FEATURE_SETS
from activity_agent.isolation.experiments.models import ExperimentModel, eligible, low_flow_mask, split_blocks


def source_features():
    rng = np.random.default_rng(7)
    rows = []
    for month in ('202510', '202511', '202512'):
        for code, province in [('11', '서울특별시'), ('32', '강원특별자치도')]:
            for block in range(50):
                values = rng.normal(.02, .05, len(FEATURE_SETS['F4']))
                if month != '202510' and 10 <= block < 14:
                    values[:] = .8
                rows.append(dict(zip(FEATURE_SETS['F4'], values)) | {
                    'block_code': f'{code}-{block:03d}', 'geo_code': code, 'province': province,
                    'sex': '남성', 'age_band': '50', 'month': month,
                    'baseline_flow': float(100 + block), 'previous_baseline_flow': float(100 + block),
                    'current_flow': 90., 'coverage': 1., 'new_quality': True,
                    **{c: 'ok' for c in ['quality', 'previous_quality', 'day_quality',
                                        'previous_day_quality', 'weekend_quality', 'previous_weekend_quality']}})
    return pd.DataFrame(rows)


@pytest.fixture
def deployed_f2(config):
    # Never use the project's absolute output path or mutate the shipped model.
    config.values['output_dir'] = 'artifacts'
    config.values['isolation'].update(model_id='IF_F2', model_path='models/IF_F2.pkl',
        flow_cutoffs_path='models/flow_cutoffs.csv')
    frame = source_features()
    train_blocks, calibration_blocks = split_blocks(frame)
    allowed, cutoffs = low_flow_mask(frame, train_blocks)
    valid = allowed & eligible(frame, FEATURE_SETS['F4'])
    training = frame.loc[valid & frame.month.eq('202510') & frame.block_code.isin(train_blocks)]
    calibration = frame.loc[valid & frame.month.eq('202510') & frame.block_code.isin(calibration_blocks)]
    estimator = ExperimentModel('IF', 'F2').fit(training, calibration)
    path = config.root / 'models/IF_F2.pkl'
    path.parent.mkdir(parents=True)
    path.write_bytes(pickle.dumps(estimator))
    cutoffs.to_csv(config.root / 'models/flow_cutoffs.csv', index=False)
    path.with_suffix('.json').write_text(json.dumps({'input_features': estimator.columns,
        'normalization': 'N0', 'threshold': estimator.threshold}), encoding='utf-8')
    return config, frame, estimator


def test_f2_detection_uses_saved_estimator_and_evidence_contract(deployed_f2, monkeypatch):
    from activity_agent.isolation import f2
    from activity_agent.isolation.handoff import candidate_evidence
    from activity_agent.isolation.config import destination
    config, source, estimator = deployed_f2
    monkeypatch.setattr(f2, 'source_features', lambda cfg: source.copy())
    model_bytes = anomaly.model_path(config).read_bytes()
    result = anomaly.detect_anomaly(config)
    assert anomaly.model_path(config).read_bytes() == model_bytes
    expected = estimator.predict(source.loc[source.month.gt('202510')].copy())
    actual = result.loc[result.month.gt('202510') & result.anomaly_raw.notna()]
    expected = expected.loc[expected.index.isin(actual.index)]
    np.testing.assert_allclose(actual.anomaly_raw, expected.raw_score)
    np.testing.assert_allclose(actual.anomaly_percentile, expected.percentile)
    np.testing.assert_array_equal(actual.candidate, expected.candidate)
    assert not result.loc[result.month.eq('202510'), 'anomaly_raw'].notna().any()
    assert result.candidate.any()
    payload = candidate_evidence(config, month='202512')
    assert payload['model']['id'] == 'IF_F2'
    assert payload['model']['score_reference'] == 'calibration'
    assert payload['provenance']['model'] == 'models/IF_F2.pkl'
    assert 'calibration_percentile' in payload['candidates'][0]['scores']
    assert destination(config, 'anomaly/evaluation.json').exists()
    assert anomaly.anomaly_readiness(config)['training_ready']
    from activity_agent.isolation.queries import IsolationEvidenceAgent
    assert '보정 표본' in IsolationEvidenceAgent(config).answer('2025년 12월 활동 이상')


def test_f2_rejects_changed_training_reference_and_does_not_publish(deployed_f2, monkeypatch):
    from activity_agent.isolation import f2
    from activity_agent.isolation.config import destination
    config, source, _ = deployed_f2
    source.loc[source.month.eq('202510'), 'relative_decline'] += .25
    monkeypatch.setattr(f2, 'source_features', lambda cfg: source)
    with pytest.raises(ValueError, match='reference'):
        anomaly.detect_anomaly(config)
    assert not destination(config, 'anomaly/detections.parquet').exists()


def test_f2_missing_features_and_small_flows_abstain(deployed_f2, monkeypatch):
    from activity_agent.isolation import f2
    config, source, _ = deployed_f2
    altered = source.copy()
    future = altered.index[altered.month.eq('202512') & altered.baseline_flow.ge(110)][:3]
    altered.loc[future[0], 'relative_decline'] = np.nan
    altered.loc[future[1], 'baseline_flow'] = .5
    altered.loc[future[2], 'geo_code'] = '31'
    monkeypatch.setattr(f2, 'source_features', lambda cfg: altered)
    scored = anomaly.detect_anomaly(config)
    assert scored.loc[future, 'anomaly_raw'].isna().all()
    assert not scored.loc[future, 'candidate'].any()


def test_f2_training_command_cannot_overwrite_experiment_model(deployed_f2):
    config, _, _ = deployed_f2
    path = config.root / 'models/IF_F2.pkl'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='detect'):
        anomaly.train_anomaly(config)
    assert path.read_bytes() == before


def test_f2_dashboard_rejects_predictions_from_another_model(deployed_f2, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from activity_agent.isolation import f2
    from activity_agent.isolation.config import destination
    config, source, _ = deployed_f2
    monkeypatch.setattr(f2, 'source_features', lambda cfg: source.copy())
    anomaly.detect_anomaly(config)
    path = destination(config, 'anomaly/detections.parquet')
    predictions = pd.read_parquet(path)
    predictions['engine_version'] = 'old-baseline-version'
    predictions.to_parquet(path, index=False)
    cfg_path = config.root / 'config/default.json'
    cfg_path.write_text(json.dumps(config.values))
    app = AppTest.from_string(f'from activity_agent.ui.isolation import render_anomaly_dashboard\n'
                             f'from activity_agent.common.config import Config\n'
                             f'render_anomaly_dashboard(Config.load({str(cfg_path)!r}))').run(timeout=30)
    assert not app.exception
    assert any('model' in error.value.lower() for error in app.error)
    assert not app.dataframe


def test_f2_builds_relative_features_from_raw_without_existing_predictions(source_data):
    from activity_agent.data.ingest import preprocess
    from activity_agent.isolation import f2
    from activity_agent.isolation.config import destination
    config = source_data
    config.values['isolation'].update(mapping_mode='verified',
        geography_file='references/geography.csv', block_mapping_file='references/block_geography.csv')
    geography = pd.DataFrame([
        {'geo_level': 'sido', 'geo_code': code, 'province': province, 'district': '', 'dong': '',
         'parent_code': '', 'reference_year': '2025', 'source_url': 'https://example.go.kr/geo', 'verified': 'true'}
        for code, province in [('11', '서울특별시'), ('32', '강원특별자치도')]])
    geography.to_csv(config.root / 'references/geography.csv', index=False)
    pd.DataFrame([
        {'block_code': block, 'geo_level': 'sido', 'geo_code': code, 'reference_year': '2025',
         'source_url': 'https://example.go.kr/geo', 'verified': 'true'}
        for block, code in [('b1', '11'), ('b2', '32')]]).to_csv(config.root / 'references/block_geography.csv', index=False)
    preprocess(config)
    result = f2.source_features(config)
    assert set(result.month) == {'202510', '202511', '202512'}
    assert set(FEATURE_SETS['F2']).issubset(result.columns)
    # All twelve age/sex groups change by the same amount in this raw fixture.
    np.testing.assert_allclose(result.relative_decline, 0., atol=1e-12)
    np.testing.assert_allclose(result.share_change, 0., atol=1e-12)
    assert result.new_quality.all()
    assert not destination(config, 'anomaly/detections.parquet').exists()


def test_f2_rejects_rescaled_absolute_flow_reference(deployed_f2, monkeypatch):
    from activity_agent.isolation import f2
    config, source, _ = deployed_f2
    source.loc[source.month.eq('202510'), 'baseline_flow'] *= 10
    monkeypatch.setattr(f2, 'source_features', lambda cfg: source)
    with pytest.raises(ValueError, match='flow.*reference'):
        anomaly.detect_anomaly(config)


def test_f2_requires_original_flow_cutoff_artifact(deployed_f2):
    config, _, _ = deployed_f2
    (config.root / 'models/flow_cutoffs.csv').unlink()
    with pytest.raises(FileNotFoundError, match='flow cutoff reference'):
        anomaly.detect_anomaly(config)
