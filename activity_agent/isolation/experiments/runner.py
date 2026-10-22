"""Checkpointed experiments. Production artifacts are read-only and hash-guarded."""
from __future__ import annotations

import importlib.metadata
import json
import pickle
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from activity_agent.common.storage import fingerprint, write_json, records
from activity_agent.isolation import anomaly
from .features import build_features, regional_frame, recompute, FEATURE_SETS, DEFINITIONS, INPUTS, KEY
from .models import ExperimentModel, DualModel, split_blocks, low_flow_mask, eligible, direction
from .injection import inject, SCENARIOS
from .paths import model_directory, configure_model_directory

VERSION = "activity-experiments-v1"
PAGE_FILES = ["index", "dataset_diagnostics", "model_comparison", "injection_experiments", "ablation_study", "regional_temporal", "candidate_explorer", "methodology"]


def experiment_root(config):
    return config.output / "isolation/experiments"


def _guard_files(config):
    root = config.output / "isolation"
    files = [p for p in root.rglob("*") if p.is_file() and "experiments" not in p.relative_to(root).parts]
    files += [config.root / "config/default.json", config.output / "clean_manifest.json",
              anomaly.model_path(config), anomaly.model_path(config).with_suffix(".json")]
    return {str(p.resolve()): fingerprint(p) for p in files}


def _status(run, identity, status, **details):
    path = run / "data/status.json"
    states = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    states[identity] = {"experiment": identity, "status": status, **details}
    write_json(path, states)
    print(f"{identity}: {status}" + (f" — {details['reason']}" if details.get("reason") else ""), flush=True)


def _save_predictions(run, identity, predicted):
    columns = ["row_id", *KEY, "geo_code", "province", *FEATURE_SETS["F4"], "baseline_flow", "g0", "g1", "g2",
               "raw_score", "percentile", "candidate"]
    result = predicted[[c for c in columns if c in predicted]].copy()
    result["model_id"] = identity
    result["unit"] = "소지역×월" if result.sex.eq("전체").all() else "소지역×성×연령×월"
    result.to_parquet(run / f"data/predictions_{identity}.parquet", index=False)
    return result


class OperatingBaseline:
    kind, feature_set, normalization, regional = "IF", "F0", "OPERATIONAL", False
    columns = INPUTS

    def __init__(self, model):
        self.model = model
        self.reference, self.threshold = model["reference_scores"], model["score_cutoff"]
        self.fit_seconds = 0.
        self.training_rows, self.calibration_rows = model["training_rows"], 0

    def predict(self, frame):
        scored = anomaly.score_frame(self.model, frame)
        result = frame.copy()
        result["raw_score"], result["percentile"], result["candidate"] = scored.anomaly_raw, scored.anomaly_percentile, scored.candidate
        return result


def _evaluate(run, identity, model, evaluation, reference=None):
    start = time.perf_counter()
    predicted = model.predict(evaluation)
    elapsed = time.perf_counter() - start
    predicted = _save_predictions(run, identity, predicted)
    rows = []
    for (month, code), part in predicted.groupby(["month", "geo_code"]):
        valid = part.raw_score.notna()
        chosen = part.loc[part.candidate]
        rows.append({"model_id": identity, "month": month, "geo_code": code, "province": part.province.iloc[0],
                     "eligible_rows": int(valid.sum()), "candidate_rows": len(chosen),
                     "candidate_blocks": int(chosen.block_code.nunique()),
                     "candidate_rate": float(part.loc[valid, "candidate"].mean()) if valid.any() else None,
                     "unit": part.unit.iloc[0]})
    pd.DataFrame(rows).to_csv(run / f"data/regional_{identity}.csv", index=False)
    size = (model_directory(run) / f"{identity}.pkl").stat().st_size
    jaccard = None
    if reference is not None:
        left = set(predicted.loc[predicted.candidate, "row_id"])
        right = set(reference.loc[reference.candidate, "row_id"])
        jaccard = len(left & right) / len(left | right) if left | right else None
    metrics = {"model_id": identity, "model": model.kind, "features": model.feature_set,
               "normalization": model.normalization, "unit": "지역" if model.regional else "집단",
               "training_rows": model.training_rows, "calibration_rows": model.calibration_rows,
               "evaluation_rows": len(evaluation), "candidate_rows": int(predicted.candidate.sum()),
               "candidate_blocks": int(predicted.loc[predicted.candidate, "block_code"].nunique()),
               "fit_seconds": model.fit_seconds, "inference_seconds": elapsed, "model_bytes": size,
               "operating_row_jaccard": jaccard if not model.regional else None,
               "threshold": model.threshold, "status": "COMPLETE"}
    write_json(run / f"data/metrics_{identity}.json", metrics)
    return predicted


def _injections(run, identity, model, evaluation, seed=42, only_scenarios=None):
    rng = np.random.default_rng(seed)
    picks = []
    for _, part in evaluation.groupby(["geo_code", "month"]):
        blocks = np.array(sorted(part.block_code.unique()))
        selected = rng.choice(blocks, min(24, len(blocks)), replace=False)
        # Stable one-group anchor per selected block, independent of scores/candidates.
        for block in selected:
            choices = part.loc[part.block_code.eq(block)].sort_values(["sex", "age_band"])
            picks.append(choices.iloc[int(rng.integers(len(choices)))])
    anchors = pd.DataFrame(picks).reset_index(drop=True)
    if only_scenarios and (run/f"data/injection_anchors_{identity}.parquet").exists():
        anchors=pd.read_parquet(run/f"data/injection_anchors_{identity}.parquet").reset_index(drop=True)
    if anchors.empty:
        _status(run, f"INJECTION_{identity}", "SKIPPED", reason="No eligible evaluation anchors")
        return
    original = model.predict(anchors)
    anchors.to_parquet(run / f"data/injection_anchors_{identity}.parquet", index=False)
    anchor_meta = anchors[KEY + ["province", "row_id", "geo_code"]].to_dict("records")
    original_values = anchors[model.columns].to_numpy(dtype=float)
    original_raw = original.raw_score.to_numpy()
    original_pct = original.percentile.to_numpy()
    original_flags = original.candidate.to_numpy()
    original_feature_names = ["original_" + column for column in model.columns]
    injected_feature_names = ["injected_" + column for column in model.columns]
    metrics, examples, pairs = [], [], []
    for scenario in SCENARIOS:
        if only_scenarios and scenario not in only_scenarios:
            continue
        if model.regional and scenario in ("S1", "S3", "S4", "S5", "S6"):
            metrics.append({"model_id": identity, "scenario": scenario, "status": "SKIPPED",
                            "reason": "집단 전용 원천 주입은 지역 시간·요일 모델의 입력 수준과 다름"})
            continue
        nominal = {"S1": .3, "S2": .4, "S3": .35, "S4": .2, "S5": .4, "S6": .3, "S7": .3, "S8": .4, "NC2": .3}[scenario]
        levels = sorted(set([.1, .2, .3, .4, .5, nominal]))
        prior_scores = None
        for strength in levels:
            changed = inject(anchors, scenario, strength)
            changed_values = changed[model.columns].to_numpy(dtype=float)
            valid = changed.injection_feasible & np.isfinite(changed[model.columns]).all(axis=1)
            # Frozen low-flow cutoffs apply after injection. They are not refit.
            if "experiment_flow_cutoff" in changed:
                valid &= ((changed.g1 + changed.g2) / 2).ge(changed.experiment_flow_cutoff)
            scores = np.full(len(changed), np.nan)
            candidate = np.zeros(len(changed), bool)
            if valid.any():
                evaluated = model.predict(changed.loc[valid])
                scores[valid] = evaluated.raw_score
                candidate[valid] = evaluated.candidate
            comparable = np.isfinite(scores) & np.isfinite(original.raw_score)
            newly = ~original.candidate.to_numpy()
            increasing = changed.activity_decline.le(0).to_numpy() & comparable
            monotonic = None
            if prior_scores is not None:
                pair = np.isfinite(prior_scores) & comparable
                monotonic = float(np.mean(scores[pair] >= prior_scores[pair] - 1e-12)) if pair.any() else None
            record = {"model_id": identity, "scenario": scenario, "scenario_name": SCENARIOS[scenario],
                      "strength": strength, "nominal": strength == nominal, "pairs": len(anchors),
                      "scored_pairs": int(comparable.sum()), "withheld_pairs": int((~comparable).sum()),
                      "detection_rate": float(candidate.mean()),
                      "raw_anomaly_rate": float(np.mean(scores >= model.threshold)),
                      "score_delta": float(np.mean(scores[comparable] - original.raw_score.to_numpy()[comparable])) if comparable.any() else None,
                      "percentile_delta": float(np.mean(100*np.searchsorted(model.reference, scores[comparable], side="right") / len(model.reference) - original.percentile.to_numpy()[comparable])) if comparable.any() else None,
                      "candidate_transition_rate": float(candidate[newly].mean()) if newly.any() else None,
                      "severity_nondecreasing_fraction": monotonic,
                      "actually_increasing_pairs": int(increasing.sum()),
                      "negative_direction_rejection": float(1-candidate[increasing].mean()) if increasing.any() else None,
                      "status": "COMPLETE" if comparable.any() else "SKIPPED",
                      "reason": "" if comparable.any() else "주입 후 유효 특징·유동량 조건을 충족하는 쌍 없음"}
            record.update(geo_code="ALL", month="ALL")
            metrics.append(record)
            for (code, month), group in anchors.groupby(["geo_code", "month"]):
                idx = group.index.to_numpy()
                cmp = comparable[idx]
                fresh = newly[idx]
                inc = increasing[idx]
                sub = record | {"geo_code": code, "month": month, "pairs": len(idx),
                    "scored_pairs": int(cmp.sum()), "withheld_pairs": int((~cmp).sum()),
                    "detection_rate": float(candidate[idx].mean()), "raw_anomaly_rate": float(np.mean(scores[idx] >= model.threshold)),
                    "score_delta": float(np.mean(scores[idx][cmp] - original.raw_score.to_numpy()[idx][cmp])) if cmp.any() else None,
                    "percentile_delta": float(np.mean(100*np.searchsorted(model.reference, scores[idx][cmp], side="right") / len(model.reference) - original.percentile.to_numpy()[idx][cmp])) if cmp.any() else None,
                    "candidate_transition_rate": float(candidate[idx][fresh].mean()) if fresh.any() else None,
                    "actually_increasing_pairs": int(inc.sum()),
                    "negative_direction_rejection": float(1-candidate[idx][inc].mean()) if inc.any() else None}
                metrics.append(sub)
            for j in range(len(anchors)):
                pairs.append({"model_id": identity, "scenario": scenario, "strength": strength,
                    **anchor_meta[j],
                    "original_score": float(original_raw[j]), "injected_score": float(scores[j]),
                    "original_percentile": float(original_pct[j]),
                    "injected_percentile": float(100*np.searchsorted(model.reference, scores[j], side="right")/len(model.reference)) if np.isfinite(scores[j]) else np.nan,
                    "original_candidate": bool(original_flags[j]), "injected_candidate": bool(candidate[j]),
                    **dict(zip(original_feature_names, original_values[j])),
                    **dict(zip(injected_feature_names, changed_values[j]))})
            for j in np.flatnonzero(~candidate & comparable)[:5]:
                examples.append({"model_id": identity, "scenario": scenario, "strength": strength,
                    **{k: anchors.iloc[j][k] for k in KEY + ["province"]},
                    "original_raw_score": float(original.raw_score.iloc[j]), "injected_raw_score": float(scores[j]),
                    "original_features": json.dumps({c: float(anchors.iloc[j][c]) for c in model.columns}),
                    "injected_features": json.dumps({c: float(changed.iloc[j][c]) for c in model.columns}), "candidate": False})
            prior_scores = scores.copy()
    outputs=[(run/f"data/injection_{identity}.csv",pd.DataFrame(metrics)),
             (run/f"data/failures_{identity}.csv",pd.DataFrame(examples)),
             (run/f"data/injection_pairs_{identity}.parquet",pd.DataFrame(pairs))]
    for path,data in outputs:
        if only_scenarios and path.exists():
            archive=run/"data/archive/pre_scenario_v2";archive.mkdir(parents=True,exist_ok=True)
            if not (archive/path.name).exists():shutil.copy2(path,archive/path.name)
            try:
                previous=pd.read_parquet(path) if path.suffix==".parquet" else pd.read_csv(path)
                data=pd.concat([previous.loc[~previous.scenario.isin(only_scenarios)],data],ignore_index=True)
            except pd.errors.EmptyDataError:pass
        if path.suffix==".parquet":data.to_parquet(path,index=False)
        else:data.to_csv(path,index=False)
    _status(run, f"INJECTION_{identity}", "COMPLETE", pairs=len(anchors), reason="실제 정상 라벨을 가정하지 않는 원천 집계량 쌍 실험")


def _temporal(run, identity, predicted):
    nov = predicted.loc[predicted.month.eq("202511") & predicted.candidate].copy()
    dec = predicted.loc[predicted.month.eq("202512")]
    follow = nov.merge(dec[["block_code", "sex", "age_band", "g0", "g1", "g2", "candidate"]],
                       on=["block_code", "sex", "age_band"], how="left", suffixes=("_nov", "_dec"), validate="one_to_one")
    if follow.empty:
        return
    follow["type"] = "평가 불가"
    # December and its November/October lags use December's same matched grid.
    valid = follow.g0_dec.notna() & follow.g1_dec.gt(0) & follow.g2_dec.gt(0)
    change = 1 - follow.g0_dec / follow.g1_dec
    follow.loc[valid & change.ge(.1), "type"] = "지속 감소"
    recovered = valid & follow.g1_dec.lt(.9*follow.g2_dec) & follow.g0_dec.ge(.9*follow.g2_dec)
    follow.loc[valid & change.le(-.1) & ~recovered, "type"] = "반등"
    follow.loc[recovered, "type"] = "회복"
    follow.loc[valid & follow.type.eq("평가 불가"), "type"] = "정체·부분 회복"
    follow.groupby(["geo_code", "province", "type"]).size().reset_index(name="rows").assign(model_id=identity).to_csv(run / f"data/temporal_{identity}.csv", index=False)


def _stability(run, identity, model, train, calibration, evaluation, predicted):
    results, sensitivity = [], []
    if model.kind in ("IF", "DUAL_IF"):
        for seed in (43, 44):
            alternate = DualModel(seed) if model.kind == "DUAL_IF" else ExperimentModel(model.kind, model.feature_set, model.normalization, seed, model.regional)
            alternate.fit(train, calibration)
            alt = alternate.predict(evaluation)
            for (month, code), group in predicted.groupby(["month", "geo_code"]):
                other = alt.loc[alt.month.eq(month) & alt.geo_code.eq(code)]
                left = set(group.loc[group.candidate].sort_values("raw_score", ascending=False).drop_duplicates("block_code").head(20).block_code)
                right = set(other.loc[other.candidate].sort_values("raw_score", ascending=False).drop_duplicates("block_code").head(20).block_code)
                union = left | right
                merged = group[["row_id", "raw_score"]].merge(other[["row_id", "raw_score"]], on="row_id", suffixes=("_a", "_b"))
                correlation = spearmanr(merged.raw_score_a, merged.raw_score_b).statistic if len(merged) > 2 else np.nan
                results.append({"model_id": identity, "seed": seed, "month": month, "geo_code": code,
                                "top20_jaccard": len(left & right)/len(union) if union else None, "rank_correlation": correlation,
                                "status": "COMPLETE"})
    else:
        results.append({"model_id": identity, "status": "SKIPPED", "reason": "확률적 학습이 없는 결정적 모델: seed 재학습은 적용 대상 아님"})
    for q in (.9, .95, .975):
        threshold = float(np.quantile(model.reference, q))
        for (month, code), part in predicted.groupby(["month", "geo_code"]):
            candidate = part.raw_score.ge(threshold) & direction(part)
            sensitivity.append({"model_id": identity, "quantile": q, "month": month, "geo_code": code,
                                "candidate_rows": int(candidate.sum()), "candidate_blocks": int(part.loc[candidate, "block_code"].nunique())})
    pd.DataFrame(results).to_csv(run / f"data/stability_{identity}.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(run / f"data/sensitivity_{identity}.csv", index=False)


def run_experiments(config, suite="all", resume=None):
    if anomaly.selected_model_id(config) != "IF_BASELINE":
        raise ValueError("Comparison experiments require isolation.model_id=IF_BASELINE; use a separate config so IF_F2 deployment is preserved")
    root = experiment_root(config)
    run_id = resume or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Invalid run ID")
    run = root / run_id
    if not resume:
        run.mkdir(parents=True, exist_ok=False)
    elif not run.exists():
        raise FileNotFoundError(run)
    for folder in ("data", "config"):
        (run / folder).mkdir(exist_ok=True)
    if not resume:
        configure_model_directory(run, config.root)
    guard = _guard_files(config)
    if (run / "config/production_hashes.json").exists():
        old = json.loads((run / "config/production_hashes.json").read_text(encoding="utf-8"))
        old = {str(anomaly.relocated_model_path(path)): sha for path, sha in old.items()}
        if old != guard:
            raise ValueError("Production inputs changed since checkpoint; start a new run")
    else:
        write_json(run / "config/production_hashes.json", guard)
    settings = {"version": VERSION, "suite": suite, "seed": 42, "train_month": "202510", "evaluation_months": ["202511", "202512"],
                "scenario_version": 2,
                "train_fraction": .8, "calibration_fraction": .2, "threshold_quantile": .95,
                "injection_blocks_per_region_month": 24, "strengths": [.1, .2, .3, .4, .5],
                "fixed_models": {"IF": {"n_estimators": 200, "contamination": "auto"}, "LOF": {"n_neighbors":20, "novelty":True},
                                 "KNN": {"n_neighbors":20}, "ROBUST_Z": "max absolute median/MAD deviation; zero MAD falls back to std"}}
    if resume and (run/"config/experiment_config.json").exists():
        old_config=json.loads((run/"config/experiment_config.json").read_text(encoding="utf-8"))
        settings["scenario_version"]=old_config.get("scenario_version",1)
    write_json(run / "config/experiment_config.json", settings)
    write_json(run / "config/project_config.json", config.values)
    write_json(run / "config/feature_definitions.json", {"features": DEFINITIONS, "sets": FEATURE_SETS, "group_C": "설명 전용; 모델 입력 제외"})
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.root, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    manifest = {"run_id": run_id, "version": VERSION, "seed": 42, "suite": suite, "git_commit": commit,
                "started_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "plotly")}}
    write_json(run / "run_manifest.json", manifest)
    print(f"Experiment run: {run}", flush=True)
    if not (run / "data/features.parquet").exists():
        frame, diagnosis = build_features(config, run)
        write_json(run / "data/source_diagnostics.json", diagnosis)
    else:
        frame = pd.read_parquet(run / "data/features.parquet")
    _status(run, "DATA_FEATURES", "COMPLETE", rows=len(frame))
    train_blocks, calibration_blocks = split_blocks(frame)
    write_json(run / "config/data_split.json", {"train_blocks": sorted(train_blocks), "calibration_blocks": sorted(calibration_blocks),
                                              "train_month": "202510", "evaluation_months": ["202511", "202512"], "split_unit": "BLOCK_CD"})
    allowed, cutoffs = low_flow_mask(frame, train_blocks)
    frame["experiment_flow_cutoff"] = frame[["geo_code", "sex", "age_band"]].merge(cutoffs, on=["geo_code", "sex", "age_band"], how="left").flow_cutoff.to_numpy()
    cutoffs.to_csv(run / "data/flow_cutoffs.csv", index=False)
    common = allowed & eligible(frame, FEATURE_SETS["F4"])
    frame["common_eligible"] = common
    frame.to_parquet(run / "data/features.parquet", index=False)
    (frame.groupby(["month", "geo_code", "sex", "age_band"])
     .agg(observations=("block_code", "size"), common_valid=("common_eligible", "sum"))
     .reset_index().to_csv(run / "data/dataset_counts.csv", index=False))
    frame.loc[common & frame.month.eq("202510") & frame.block_code.isin(train_blocks), FEATURE_SETS["F4"]].corr().to_csv(run / "data/feature_correlations.csv")
    completed = {}
    # Operational snapshot is always separate from the group-calibrated comparison.
    try:
        baseline = OperatingBaseline(anomaly.read_model(config))
        path = model_directory(run) / "IF_BASELINE.pkl"
        if not path.exists():
            with path.open("wb") as stream:
                pickle.dump(baseline, stream)
            shutil.copy2(anomaly.model_path(config), model_directory(run) / "IF_BASELINE_ORIGINAL.pkl")
        future = frame.loc[frame.month.isin(["202511", "202512"])]
        predicted = _evaluate(run, "IF_BASELINE", baseline, future)
        saved = pd.read_parquet(config.output / "isolation/anomaly/detections.parquet")
        compare = predicted.merge(saved[[*KEY, "anomaly_raw", "candidate"]], on=KEY, suffixes=("_new", "_saved"))
        error = float((compare.raw_score - compare.anomaly_raw).abs().max())
        mismatch = int(compare.candidate_new.ne(compare.candidate_saved).sum())
        write_json(run / "data/baseline_reproduction.json", {"score_max_abs_error": error, "candidate_mismatches": mismatch,
                                                           "status": "COMPLETE" if error < 1e-10 and not mismatch else "FAILED"})
        if error > 1e-10 or mismatch:
            raise ValueError("Operating baseline reproduction mismatch")
        completed["IF_BASELINE"] = predicted
        if suite in ("all", "models", "injection") and not (run / "data/injection_IF_BASELINE.csv").exists():
            baseline_future = future.loc[predicted.raw_score.notna()].copy()
            baseline_future["experiment_flow_cutoff"] = baseline_future.baseline_cutoff
            _injections(run, "IF_BASELINE", baseline, baseline_future)
        _status(run, "IF_BASELINE", "COMPLETE", reason="기존 학습·임계값·결과 복사 및 재현 확인; 공정 비교용 보정 방식과 구분")
    except Exception as exc:
        _status(run, "IF_BASELINE", "FAILED", reason=str(exc))
    specs = [(f"IF_{f}", "IF", f, "N0") for f in FEATURE_SETS]
    specs += [(f"IF_F0_{n}", "IF", "F0", n) for n in ("N1", "N2", "N3")]
    specs += [(f"{kind}_F4", kind, "F4", "N0") for kind in ("LOF", "KNN", "ROBUST_Z")]
    specs += [("DUAL_IF", "DUAL_IF", "F4", "N0")]
    if suite in ("models", "injection", "temporal"):
        specs = [s for s in specs if s[0] in ("IF_F0", "IF_F4", "LOF_F4", "KNN_F4", "ROBUST_Z_F4", "DUAL_IF")]
    elif suite == "ablation":
        specs = specs[:7]
    elif suite == "regional":
        specs = [s for s in specs if s[0] in ("IF_F0", "IF_F6", "IF_F0_N1", "IF_F0_N2", "IF_F0_N3", "DUAL_IF")]
    if suite in ("all", "ablation"):
        specs += [(f"FULL_IF_{feature}", "IF", feature, "N0") for feature in FEATURE_SETS]
    for identity, kind, features, normalization in specs:
        try:
            regional = features == "F6"
            population = allowed & eligible(frame, FEATURE_SETS[features]) if identity.startswith("FULL_") else common
            data = regional_frame(frame.loc[population]) if regional else frame.loc[population].copy()
            if regional:
                regional_fit = data.loc[data.month.eq("202510") & data.block_code.isin(train_blocks)]
                regional_cuts = regional_fit.groupby("geo_code").baseline_flow.quantile(.1).clip(lower=1.)
                data["experiment_flow_cutoff"] = data.geo_code.map(regional_cuts)
                data = data.loc[data.baseline_flow.ge(data.experiment_flow_cutoff)]
            data = data.loc[eligible(data, FEATURE_SETS[features])].reset_index(drop=True)
            train = data.loc[data.month.eq("202510") & data.block_code.isin(train_blocks)]
            calibration = data.loc[data.month.eq("202510") & data.block_code.isin(calibration_blocks)]
            evaluation = data.loc[data.month.isin(["202511", "202512"])]
            if min(len(train), len(calibration), len(evaluation)) < 20:
                _status(run, identity, "SKIPPED", reason="학습/보정/평가 유효 관측 부족")
                continue
            path = model_directory(run) / f"{identity}.pkl"
            if path.exists():
                with path.open("rb") as stream:
                    model = pickle.load(stream)
            else:
                print(f"Fitting {identity}: train={len(train)}, calibration={len(calibration)}, evaluation={len(evaluation)}", flush=True)
                model = DualModel() if kind == "DUAL_IF" else ExperimentModel(kind, features, normalization, regional=regional)
                model.fit(train, calibration)
                with path.open("wb") as stream:
                    pickle.dump(model, stream)
            write_json(model_directory(run) / f"{identity}.json", {"input_features": model.columns, "normalization": normalization,
                       "unit": "block/month" if regional else "block/sex/age/month", "seed": 42, "threshold": model.threshold,
                       "threshold_source": "held-out October block calibration 95%", "scaler": model.scaler.diagnostics})
            pd.DataFrame(model.scaler.diagnostics).assign(model_id=identity).to_csv(run / f"data/normalization_{identity}.csv", index=False)
            predicted = _evaluate(run, identity, model, evaluation, completed.get("IF_BASELINE"))
            completed[identity] = predicted
            _temporal(run, identity, predicted)
            if not (run / f"data/stability_{identity}.csv").exists():
                _stability(run, identity, model, train, calibration, evaluation, predicted)
            if suite in ("all", "injection", "models", "ablation") and not (run / f"data/injection_{identity}.csv").exists():
                _injections(run, identity, model, evaluation)
            _status(run, identity, "COMPLETE", training_rows=len(train), calibration_rows=len(calibration), evaluation_rows=len(evaluation))
        except Exception as exc:
            _status(run, identity, "FAILED", reason=str(exc), traceback=traceback.format_exc())
    # Every feature set also records its full-valid population, separately from matched comparisons.
    for feature_set, columns in FEATURE_SETS.items():
        valid = allowed & eligible(frame, columns)
        full = frame.loc[valid]
        if feature_set == "F6":
            full = regional_frame(full)
        write_json(run / f"data/population_{feature_set}.json", {"feature_set": feature_set, "full_valid_rows": len(full),
            "common_group_rows": int(common.sum()), "full_population_model": f"FULL_IF_{feature_set}",
            "note": "주 비교는 공통 표본, FULL_IF 모델은 각 특징 집합의 전체 유효 표본을 사용"})
        if f"FULL_IF_{feature_set}" not in completed:
            _status(run, f"FULL_POPULATION_FIT_{feature_set}", "SKIPPED", reason="이 suite에서 전체 유효 표본 별도 적합을 선택하지 않았거나 해당 모델 실패; 모델 상태 참조")
    _status(run, "HOLIDAY_EVENT_CONTROL", "SKIPPED", reason="검증된 공휴일·지역 행사 원천 및 관측 단위 연계 자료 없음")
    _status(run, "LONG_TERM_PERSISTENCE", "SKIPPED", reason="2025년 12월 이후 후속 자료 없음")
    final_guard = _guard_files(config)
    if guard != final_guard:
        _status(run, "PRODUCTION_PRESERVATION", "FAILED", reason="운영 산출물 해시 변경 감지")
    else:
        _status(run, "PRODUCTION_PRESERVATION", "COMPLETE", reason="운영 모델·점수·라벨·설정 파일 해시 일치")
    consolidate(run)
    manifest.update(status="COMPLETE_WITH_STATUS", finished_utc=datetime.now(timezone.utc).isoformat(), production_unchanged=guard == final_guard)
    write_json(run / "run_manifest.json", manifest)
    from .validation import finalize_saved
    finalize_saved(run, measure_memory=True)
    from .report import render_report
    render_report(run)
    print(f"HTML report: {run / 'index.html'}", flush=True)
    return run


def consolidate(run):
    patterns = {"model_metrics": "metrics_*.json", "regional_metrics": "regional_*.csv", "injection_metrics": "injection_*.csv",
                "temporal_metrics": "temporal_*.csv", "stability_metrics": "stability_*.csv", "threshold_metrics": "sensitivity_*.csv",
                "failure_cases": "failures_*.csv", "normalization_diagnostics": "normalization_*.csv"}
    for name, pattern in patterns.items():
        parts = []
        for path in sorted((run / "data").glob(pattern)):
            if path.name == f"{name}.csv":
                continue
            try:
                parts.append(pd.DataFrame([json.loads(path.read_text(encoding="utf-8"))]) if path.suffix == ".json" else pd.read_csv(path))
            except pd.errors.EmptyDataError:
                continue
        (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"status": ["SKIPPED"], "reason": ["해당 실험 결과 없음"]})).to_csv(run / f"data/{name}.csv", index=False, encoding="utf-8-sig")
    metrics = pd.read_csv(run / "data/model_metrics.csv")
    fair_path=run/"data/predictions_IF_F0.parquet"
    if fair_path.exists():
        fair=pd.read_parquet(fair_path)
        fair_set=set(fair.loc[fair.candidate,"row_id"])
        for index,row in metrics.iterrows():
            path=run/f"data/predictions_{row.model_id}.parquet"
            if not path.exists():continue
            predicted=pd.read_parquet(path)
            if predicted.sex.eq("전체").all():continue
            chosen=set(predicted.loc[predicted.candidate,"row_id"])
            union=chosen|fair_set
            metrics.loc[index,"fair_F0_row_jaccard"]=len(chosen&fair_set)/len(union) if union else np.nan
            joined=predicted[["row_id","raw_score"]].merge(fair[["row_id","raw_score"]],on="row_id",suffixes=("_model","_F0")).dropna()
            metrics.loc[index,"fair_F0_rank_correlation"]=spearmanr(joined.raw_score_model,joined.raw_score_F0).statistic if len(joined)>2 else np.nan
        metrics.to_csv(run/"data/model_metrics.csv",index=False,encoding="utf-8-sig")
    metrics.loc[metrics.model_id.str.match(r"IF_F[0-6]$")].to_csv(run / "data/ablation_metrics.csv", index=False, encoding="utf-8-sig")
    statuses = json.loads((run / "data/status.json").read_text(encoding="utf-8"))
    write_json(run / "data/experiment_details.json", statuses)
    write_json(run / "evaluation.json", {"status_counts": pd.Series([v["status"] for v in statuses.values()]).value_counts().to_dict(),
                                        "real_isolation_accuracy_evaluated": False, "models": metrics.model_id.tolist()})
