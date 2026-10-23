"""Deploy the saved IF_F2 experiment without refitting its estimator or calibration."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from activity_agent.common.storage import fingerprint, digest
from .config import destination, settings


def flow_cutoffs_path(config):
    from pathlib import Path
    name = config.values.get("isolation", {}).get("flow_cutoffs_path")
    if not name:
        from .anomaly import model_path
        return model_path(config).with_name('flow_cutoffs.csv')
    path = Path(name)
    return path if path.is_absolute() else config.root / path


def read_model(config):
    from .anomaly import anomaly_settings, model_path
    from .experiments.features import FEATURE_SETS
    from .experiments.models import ExperimentModel

    path = model_path(config)
    with path.open('rb') as stream:
        estimator = pickle.load(stream)
    if (not isinstance(estimator, ExperimentModel) or estimator.kind != 'IF'
            or estimator.feature_set != 'F2' or estimator.normalization != 'N0'
            or estimator.regional or estimator.columns != FEATURE_SETS['F2']):
        raise ValueError('Configured IF_F2 model is not a fitted group-level IF/F2/N0 model')
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if (metadata.get('input_features') != estimator.columns
            or metadata.get('normalization') != 'N0'
            or metadata.get('threshold') != estimator.threshold):
        raise ValueError('IF_F2 model and JSON metadata do not match')
    cfg, params = settings(config), anomaly_settings(config)
    regions = sorted({str(key[0]) for key in estimator.scaler.profiles})
    if sorted(cfg['region_codes']) != regions:
        raise ValueError('IF_F2 region scope does not match the saved model')
    if (cfg['train_end'] != '202510' or cfg['validation_month'] != '202511'
            or cfg['test_month'] != '202512' or cfg['time_mode'] != 'retrospective'
            or cfg['geo_level'] != 'sido' or cfg['mapping_mode'] != 'crs_consensus'
            or cfg['holdout_regions']):
        raise ValueError('IF_F2 requires its original October 2025 training and Seoul/Gangwon CRS-consensus experiment settings')
    if (config.values['seed'] != estimator.seed or params['reference_quantile'] != .95
            or params['baseline_quantile'] != .1 or params['min_baseline_flow'] != 1.
            or params['minimum_decline'] != .1 or config.values['minimum_coverage'] != .8
            or any(tree.n_estimators != params['n_estimators'] for tree in estimator.estimators.values())):
        raise ValueError('IF_F2 settings changed; use the saved experiment settings')
    reference = np.asarray(estimator.reference, dtype=float)
    if (reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all()
            or not np.all(reference[:-1] <= reference[1:])
            or not np.isclose(np.quantile(reference, .95), estimator.threshold, rtol=0, atol=1e-12)):
        raise ValueError('Invalid IF_F2 calibration reference')
    cutoff_path = flow_cutoffs_path(config)
    cutoff_sha = fingerprint(cutoff_path) if cutoff_path.exists() else None
    expected_sha = metadata.get('flow_cutoffs_sha256')
    if expected_sha and cutoff_sha is not None and cutoff_sha != expected_sha:
        raise ValueError('IF_F2 flow cutoff reference does not match the packaged model metadata')
    version = 'IF_F2:' + digest({'model_sha256': fingerprint(path), 'flow_cutoffs_sha256': cutoff_sha})
    blocks = None
    deployment = destination(config, 'anomaly/deployment.json')
    if deployment.exists():
        saved = json.loads(deployment.read_text(encoding='utf-8'))
        if saved.get('model_version') == version:
            blocks = saved.get('training_blocks')
    return {'task_id': 'local_activity_anomaly_f2_v1', 'model_id': 'IF_F2',
            'estimator': estimator, 'inputs': list(estimator.columns), 'model_version': version,
            'region_codes': regions, 'train_end': cfg['train_end'],
            'validation_month': cfg['validation_month'], 'test_month': cfg['test_month'],
            'reference_scores': reference, 'score_cutoff': float(estimator.threshold),
            'score_reference': 'calibration', 'anomaly_settings': params,
            'flow_cutoffs_sha256': cutoff_sha,
            'mapping_mode': cfg['mapping_mode'], 'time_mode': cfg['time_mode'],
            'training_rows': estimator.training_rows, 'calibration_rows': estimator.calibration_rows,
            'training_blocks': blocks, 'training_months': ['202510']}


def source_features(config):
    """Reuse the experiment's exact common-grid formulas, without prior predictions."""
    from .exploration import local_features
    from .experiments.features import build_features

    base = local_features(config)
    months = sorted(base.loc[base.month.ge('202510'), 'month'].unique())
    source, _ = build_features(config, base=base, months=months)
    return source


def prepare_features(config, model):
    """Verify original cutoffs and dimensionless references before publishing.

    The historical pickle omitted absolute-flow cutoffs, so require the saved
    experiment CSV rather than accepting a newly inferred eligibility boundary.
    """
    from .experiments.features import FEATURE_SETS
    from .experiments.models import Normalizer, eligible, low_flow_mask, split_blocks

    cutoff_path = flow_cutoffs_path(config)
    if not cutoff_path.exists():
        raise FileNotFoundError(f"Missing original IF_F2 flow cutoff reference: {cutoff_path}; restore flow_cutoffs.csv from the experiment run")
    original_cutoffs = pd.read_csv(cutoff_path, dtype={k: str for k in ('geo_code', 'sex', 'age_band')})
    keys = ['geo_code', 'sex', 'age_band']
    if not set([*keys, 'flow_cutoff']).issubset(original_cutoffs) or original_cutoffs.duplicated(keys).any():
        raise ValueError('Invalid IF_F2 flow cutoff reference schema')
    original_cutoffs['flow_cutoff'] = pd.to_numeric(original_cutoffs.flow_cutoff, errors='coerce')
    if not np.isfinite(original_cutoffs.flow_cutoff).all() or original_cutoffs.flow_cutoff.lt(1.).any():
        raise ValueError('Invalid IF_F2 flow cutoff reference values')
    if fingerprint(cutoff_path) != model['flow_cutoffs_sha256']:
        raise ValueError('IF_F2 flow cutoff reference changed while loading')
    frame = source_features(config).reset_index(drop=True)
    if frame.empty or not frame.month.eq('202510').any():
        raise ValueError('IF_F2 requires original October training/calibration reference data')
    train_blocks, calibration_blocks = split_blocks(frame, model['estimator'].seed)
    allowed, cutoffs = low_flow_mask(frame, train_blocks)
    compared = cutoffs.merge(original_cutoffs, on=keys, suffixes=('_rebuilt', '_saved'),
                             how='outer', validate='one_to_one')
    if not np.allclose(compared.flow_cutoff_rebuilt, compared.flow_cutoff_saved, rtol=1e-10, atol=1e-12):
        raise ValueError('IF_F2 flow cutoff reference changed; restore original absolute-flow training data')
    common = allowed & eligible(frame, FEATURE_SETS['F4'])
    training = frame.loc[common & frame.month.eq('202510') & frame.block_code.isin(train_blocks)]
    calibration = frame.loc[common & frame.month.eq('202510') & frame.block_code.isin(calibration_blocks)]
    estimator = model['estimator']
    if len(training) != estimator.training_rows or len(calibration) != estimator.calibration_rows:
        raise ValueError('IF_F2 reference sample counts changed; restore original training data and mapping')
    rebuilt = Normalizer(estimator.columns, 'N0').fit(training)
    if rebuilt.profiles.keys() != estimator.scaler.profiles.keys():
        raise ValueError('IF_F2 reference normalization profiles changed')
    for key, (median, scale) in rebuilt.profiles.items():
        old_median, old_scale = estimator.scaler.profiles[key]
        if not (np.allclose(median, old_median, rtol=1e-10, atol=1e-12)
                and np.allclose(scale, old_scale, rtol=1e-10, atol=1e-12)):
            raise ValueError('IF_F2 reference normalization changed; restore original training data')
    scores = np.sort(estimator.raw_score(calibration))
    if not np.allclose(scores, model['reference_scores'], rtol=1e-10, atol=1e-12):
        raise ValueError('IF_F2 calibration reference changed; restore original training data')
    frame['experiment_flow_cutoff'] = frame[['geo_code', 'sex', 'age_band']].merge(
        original_cutoffs[keys + ['flow_cutoff']], on=keys, how='left', validate='many_to_one').flow_cutoff.to_numpy()
    model['training_blocks'] = int(training.block_code.nunique())
    return frame


def score_frame(model, frame):
    from .anomaly import quality_mask
    from .experiments.features import FEATURE_SETS

    result = frame.copy()
    # IF_F2 was compared on the F4 common-valid population, not merely eight nonmissing columns.
    valid = (quality_mask(frame) & frame.new_quality.fillna(False)
             & np.isfinite(frame[FEATURE_SETS['F4']]).all(axis=1)
             & frame.baseline_flow.ge(frame.experiment_flow_cutoff)
             & frame.geo_code.isin(model['region_codes']))
    supported = pd.MultiIndex.from_frame(frame[['geo_code', 'sex', 'age_band']]).isin(model['estimator'].scaler.profiles)
    valid &= supported
    publish = valid & frame.month.gt(model['train_end'])
    result['anomaly_raw'] = np.nan
    result['anomaly_percentile'] = np.nan
    if publish.any():
        scores = model['estimator'].raw_score(frame.loc[publish])
        result.loc[publish, 'anomaly_raw'] = scores
        result.loc[publish, 'anomaly_percentile'] = 100 * np.searchsorted(model['reference_scores'], scores, side='right') / len(model['reference_scores'])
    result['baseline_cutoff'] = frame.experiment_flow_cutoff
    result['quality_eligible'] = valid
    result['direction_support'] = frame.activity_decline.ge(.1) & (frame.daytime_decline.gt(0) | frame.weekend_decline.gt(0))
    result['candidate'] = publish & result.anomaly_raw.ge(model['score_cutoff']) & result.direction_support
    result['persistent_20pct'] = frame.activity_decline.ge(.2) & frame.previous_activity_decline.ge(.2)
    result['status'] = '자료 이력·품질·특징 부족, 소량 유동 또는 학습 범위 밖'
    result.loc[valid & ~publish, 'status'] = '학습 기간: 예측 순위 미제공'
    result.loc[publish, 'status'] = '활동 이상 점수 산출'
    result.loc[result.candidate, 'status'] = '활동 위축 이상 후보'
    result['model_id'] = 'IF_F2'
    result['engine_version'] = model['model_version']
    result['score_kind'] = '보정 표본 분포 대비 활동 이상 백분위; 고립 확률 아님'
    return result


def evaluation(model, scored):
    monthly = []
    for (month, code), group in scored.loc[scored.month.gt(model['train_end'])].groupby(['month', 'geo_code']):
        monthly.append({'month': month, 'geo_code': code,
                        'eligible_rows': int(group.anomaly_raw.notna().sum()),
                        'candidate_rows': int(group.candidate.sum()),
                        'candidate_blocks': int(group.loc[group.candidate, 'block_code'].nunique())})
    return {'model_id': 'IF_F2', 'model_version': model['model_version'],
            'training_rows': model['training_rows'], 'training_blocks': model['training_blocks'],
            'calibration_rows': model['calibration_rows'], 'training_months': model['training_months'],
            'score_reference': 'calibration', 'score_cutoff': model['score_cutoff'],
            'monthly': monthly, 'accuracy_evaluated': False,
            'notes': ['Saved IF_F2 estimator, normalization and October calibration threshold were not refit.',
                      'Original training/calibration reference was verified before scoring.',
                      'Scores are activity anomalies, not isolation probabilities.']}
