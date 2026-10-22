"""Grouped train/calibration preprocessing and consistently oriented novelty scores."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from .features import INPUTS, GROUP, REGIONAL, FEATURE_SETS, regional_frame


def split_blocks(frame, seed=42):
    train, calibration = [], []
    rng = np.random.default_rng(seed)
    for _, group in frame.loc[frame.month.eq("202510")].groupby("geo_code"):
        blocks = np.array(sorted(group.block_code.unique()))
        rng.shuffle(blocks)
        cut = int(len(blocks) * .8)
        train.extend(blocks[:cut])
        calibration.extend(blocks[cut:])
    if set(train) & set(calibration):
        raise ValueError("Block split overlaps")
    return set(train), set(calibration)


def low_flow_mask(frame, train_blocks):
    valid = (frame[["quality", "previous_quality", "day_quality", "previous_day_quality", "weekend_quality", "previous_weekend_quality"]]
             .eq("ok").all(axis=1) & np.isfinite(frame[INPUTS]).all(axis=1) & frame.baseline_flow.gt(0))
    fit = frame.loc[valid & frame.month.eq("202510") & frame.block_code.isin(train_blocks)]
    rows = []
    for key, part in fit.groupby(["geo_code", "sex", "age_band"]):
        rows.append(dict(zip(["geo_code", "sex", "age_band"], key)) | {"flow_cutoff": max(1., float(part.baseline_flow.quantile(.1)))})
    cuts = pd.DataFrame(rows)
    cutoff = frame[["geo_code", "sex", "age_band"]].merge(cuts, on=["geo_code", "sex", "age_band"], how="left").flow_cutoff.to_numpy()
    return valid & frame.baseline_flow.ge(cutoff), cuts


class Normalizer:
    def __init__(self, columns, mode="N0", regional=False):
        self.columns, self.mode, self.regional = list(columns), mode, regional
        self.keys = ["geo_code"] if regional or mode in ("N1", "N2") else ["sex", "age_band"] if mode == "N3" else ["geo_code", "sex", "age_band"]

    def fit(self, frame):
        global_median = frame[self.columns].median()
        global_scale = frame[self.columns].quantile(.75) - frame[self.columns].quantile(.25)
        global_scale = global_scale.where(global_scale > .001, frame[self.columns].std()).fillna(.001).clip(lower=.001)
        self.global_median, self.global_scale = global_median, global_scale
        self.profiles, self.diagnostics = {}, []
        for key, part in frame.groupby(self.keys, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            med = part[self.columns].median()
            raw = part[self.columns].quantile(.75) - part[self.columns].quantile(.25)
            small = len(part) < 8
            if self.mode == "N0" and not self.regional:
                scale = raw.clip(lower=.001)
            else:
                scale = raw.where(raw >= .001, global_scale)
            if small:
                med, scale = global_median, global_scale
            self.profiles[key] = (med, scale)
            for column in self.columns:
                self.diagnostics.append(dict(zip(self.keys, key)) | {"feature": column, "training_rows": len(part),
                    "iqr": float(raw[column]), "iqr_near_zero": bool(raw[column] < .001),
                    "fallback": bool(small or ((self.mode != "N0" or self.regional) and raw[column] < .001)),
                    "floor_applied": bool(self.mode == "N0" and not self.regional and raw[column] < .001),
                    "scale": float(scale[column])})
        return self

    def transform(self, frame):
        out = np.empty((len(frame), len(self.columns)))
        positions = pd.DataFrame({k: frame[k].to_numpy() for k in self.keys})
        for key, indices in positions.groupby(self.keys, sort=False).groups.items():
            key = key if isinstance(key, tuple) else (key,)
            med, scale = self.profiles.get(key, (self.global_median, self.global_scale))
            out[indices] = ((frame.iloc[indices][self.columns] - med) / scale).clip(-20, 20).to_numpy()
        return out


class ExperimentModel:
    def __init__(self, model="IF", feature_set="F0", normalization="N0", seed=42, regional=False):
        self.kind, self.feature_set, self.normalization, self.seed, self.regional = model, feature_set, normalization, seed, regional
        self.columns = list(FEATURE_SETS[feature_set])

    def fit(self, training, calibration):
        start = time.perf_counter()
        self.scaler = Normalizer(self.columns, self.normalization, self.regional).fit(training)
        x = self.scaler.transform(training)
        self.estimators = {}
        regions = sorted(training.geo_code.unique()) if self.normalization == "N2" else ["all"]
        for region in regions:
            mask = training.geo_code.eq(region).to_numpy() if region != "all" else np.ones(len(training), bool)
            values = x[mask]
            k = min(20, len(values) - 1)
            if k < 2:
                raise ValueError("Insufficient training neighbors")
            if self.kind == "IF":
                estimator = IsolationForest(n_estimators=200, contamination="auto", random_state=self.seed, n_jobs=1).fit(values)
            elif self.kind == "LOF":
                estimator = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination="auto", n_jobs=1).fit(values)
            elif self.kind == "KNN":
                estimator = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=1).fit(values)
            elif self.kind == "ROBUST_Z":
                median = np.median(values, axis=0)
                mad = np.median(np.abs(values - median), axis=0) * 1.4826
                estimator = (median, np.where(mad > .001, mad, np.maximum(np.std(values, axis=0), .001)))
            else:
                raise ValueError(self.kind)
            self.estimators[region] = estimator
        self.reference = np.sort(self.raw_score(calibration))
        self.threshold = float(np.quantile(self.reference, .95))
        self.training_rows, self.calibration_rows = len(training), len(calibration)
        self.fit_seconds = time.perf_counter() - start
        return self

    def raw_score(self, frame):
        x = self.scaler.transform(frame)
        result = np.full(len(frame), np.nan)
        for region, estimator in self.estimators.items():
            mask = frame.geo_code.eq(region).to_numpy() if region != "all" else np.ones(len(frame), bool)
            if not mask.any():
                continue
            if self.kind in ("IF", "LOF"):
                result[mask] = -estimator.score_samples(x[mask])
            elif self.kind == "KNN":
                # Bounded query batches avoid an unbounded distance matrix.
                idx = np.flatnonzero(mask)
                for start in range(0, len(idx), 2048):
                    part = idx[start:start+2048]
                    result[part] = estimator.kneighbors(x[part], return_distance=True)[0][:, -1]
            else:
                median, scale = estimator
                result[mask] = np.max(np.abs((x[mask] - median) / scale), axis=1)
        return result

    def predict(self, frame):
        result = frame.copy()
        values = self.raw_score(frame)
        result["raw_score"] = values
        result["percentile"] = 100 * np.searchsorted(self.reference, values, side="right") / len(self.reference)
        result["candidate"] = (values >= self.threshold) & direction(frame)
        return result


class DualModel:
    columns = FEATURE_SETS["F4"]
    kind, feature_set, normalization, regional = "DUAL_IF", "F4", "N0", False

    def __init__(self, seed=42, weight=.5):
        self.seed, self.weight = seed, weight

    def fit(self, training, calibration):
        start = time.perf_counter()
        self.group = ExperimentModel("IF", "F5", seed=self.seed).fit(training, calibration)
        self.region = ExperimentModel("IF", "F6", "N1", self.seed, regional=True).fit(regional_frame(training), regional_frame(calibration))
        self.reference = np.sort(self.raw_score(calibration))
        self.threshold = float(np.quantile(self.reference, .95))
        self.training_rows, self.calibration_rows = len(training), len(calibration)
        self.fit_seconds = time.perf_counter() - start
        self.scaler = self.group.scaler
        return self

    def raw_score(self, frame):
        group_scores = self.group.raw_score(frame)
        regional = regional_frame(frame)
        rvalues = self.region.raw_score(regional)
        regional["r_percentile"] = np.searchsorted(self.region.reference, rvalues, side="right") / len(self.region.reference)
        regional_scores = frame[["block_code", "month"]].merge(regional[["block_code", "month", "r_percentile"]], on=["block_code", "month"], validate="many_to_one").r_percentile.to_numpy()
        group_pct = np.searchsorted(self.group.reference, group_scores, side="right") / len(self.group.reference)
        return self.weight * group_pct + (1-self.weight) * regional_scores

    predict = ExperimentModel.predict


def direction(frame):
    return frame.activity_decline.ge(.1) & (frame.daytime_decline.gt(0) | frame.weekend_decline.gt(0))


def eligible(frame, columns):
    return np.isfinite(frame[columns]).all(axis=1) & frame.new_quality.fillna(False)
