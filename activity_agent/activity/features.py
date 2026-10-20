from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from activity_agent.common.config import Config
from activity_agent.data.aggregation import compare_skt_window, decline, entropy, previous_month
from activity_agent.common.storage import load_table, save_table, write_json
from activity_agent.data.ingest import industry_mapping, load_mapping

ENTITY = ["signal", "area_id", "sex", "age"]
KEY = [*ENTITY, "month"]
DECLINES = ["activity_decline", "daytime_decline", "weekend_decline", "consumption_decline",
            "social_consumption_decline", "diversity_decline"]


def skt_features(config: Config) -> pd.DataFrame:
    outputs = []
    mapping = load_mapping(config)
    manifest_path = config.output / "clean_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Missing clean_manifest.json. Run preprocess first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for family, signal in [("age", "mobility"), ("time", "time_pattern"), ("wkdy", "weekday_pattern")]:
        paths = sorted(config.output / name for name in manifest["skt_files"] if name.startswith(f"clean/skt_{family}_"))
        if not paths:
            raise ValueError(f"No clean SKT {family} files. Run preprocess.")
        # Keep at most three raw months in memory.
        cache = {}
        for path in paths:
            month = path.stem.rsplit("_", 1)[1]
            cache[month] = pd.read_parquet(path)
            cache = {m: f for m, f in cache.items() if m in [month, previous_month(month), previous_month(month, 2)]}
            part = compare_skt_window(cache, month, family, config.values["minimum_coverage"])
            part = part.merge(mapping[["block_code", "region"]], on="block_code", how="left", validate="many_to_one")
            part["region"] = part.region.fillna("")
            part["area_id"] = part.block_code
            part["signal"] = signal
            outputs.append(part)
            if family == "age" and not mapping.empty:
                # Recompute on common grids at district level, rather than averaging block rates.
                mapped = {}
                for m, raw in cache.items():
                    raw = raw.merge(mapping[["block_code", "region"]], on="block_code", validate="many_to_one")
                    raw["original_block"] = raw.block_code
                    raw["block_code"] = raw.region
                    mapped[m] = raw
                district = compare_skt_window(mapped, month, family, config.values["minimum_coverage"])
                district["region"] = district.block_code
                district["area_id"] = district.block_code
                district["signal"] = "district_mobility"
                outputs.append(district)
    return pd.concat(outputs, ignore_index=True)


def consumption_monthly(config: Config) -> pd.DataFrame:
    card = load_table(config, "clean/card1")
    card = card.loc[card.sex.isin(["남성", "여성"])].copy()
    native = card.groupby(["month", "region", "sex", "age", "industry"], as_index=False)[["count", "amount"]].sum()
    save_table(config, "card_native_monthly", native)
    card["age"] = np.where(pd.to_numeric(card.age) >= 60, "60+", card.age)
    industries = industry_mapping(config)
    card = card.merge(industries[["industry", "group"]], on="industry", how="left", validate="many_to_one")
    card["group"] = card.group.fillna("other")
    card["social"] = card["count"].where(card.group.eq("social"), 0)
    card["essential"] = card["count"].where(card.group.eq("essential"), 0)
    card["classified"] = card["count"].where(card.group.ne("other"), 0)
    key = ["month", "region", "sex", "age"]
    total = card.groupby(key)[["count", "amount", "social", "essential", "classified"]].sum()
    by_industry = card.groupby([*key, "industry"])["count"].sum()
    diversity = by_industry.groupby(level=key).agg(entropy)
    total["diversity"] = diversity
    total["social_ratio"] = total.social / total["count"].where(total["count"] > 0)
    total["classification_coverage"] = total.classified / total["count"].where(total["count"] > 0)
    total = total.reset_index()
    save_table(config, "card_monthly", total)
    return total


def card_features(config: Config) -> pd.DataFrame:
    monthly = consumption_monthly(config)
    key = ["region", "sex", "age"]
    outputs = []
    for month, now in monthly.groupby("month"):
        now = now.set_index(key)
        current = now.drop(columns="month").add_prefix("current_")
        out = current.copy()
        past = [previous_month(month, i) for i in (1, 2)]
        out["quality"] = "insufficient_history"
        if all(m in set(monthly.month) for m in past):
            previous = [monthly.loc[monthly.month.eq(m)].set_index(key).drop(columns="month") for m in past]
            baseline = (previous[0] + previous[1]) / 2
            out = out.join(baseline.add_prefix("baseline_"))
            for source, target in [("count", "consumption_decline"), ("social", "social_consumption_decline"), ("diversity", "diversity_decline")]:
                out[target] = decline(out[f"current_{source}"], out[f"baseline_{source}"])
            out["social_ratio_change"] = out.current_social_ratio - out.baseline_social_ratio
            out["quality"] = np.where(out.baseline_count.gt(0) & out.current_count.notna(), "ok", "zero_baseline")
        outputs.append(out.reset_index().assign(month=month, signal="consumption"))
    frame = pd.concat(outputs, ignore_index=True)
    frame["area_id"] = frame.region
    return frame


def build_features(config: Config) -> pd.DataFrame:
    frame = pd.concat([skt_features(config), card_features(config)], ignore_index=True)
    for col in DECLINES:
        if col not in frame:
            frame[col] = np.nan
    frame = frame.sort_values(KEY).reset_index(drop=True)
    if frame.duplicated(KEY).any():
        raise ValueError("Feature keys are not unique.")
    # Match by calendar month, never by the preceding observed row.
    primary = frame.activity_decline.where(frame.signal.ne("consumption"), frame.consumption_decline)
    frame["primary_decline"] = primary
    prior = frame[[*ENTITY, "month", "primary_decline", "quality", "social_consumption_decline", "diversity_decline"]].copy()
    prior["month"] = prior.month.map(lambda m: previous_month(m, -1))
    prior = prior.rename(columns={"primary_decline": "previous_decline", "quality": "previous_quality",
                                 "social_consumption_decline": "previous_social_consumption_decline",
                                 "diversity_decline": "previous_diversity_decline"})
    frame = frame.merge(prior, on=KEY, how="left", validate="one_to_one")
    frame["feature_id"] = [hashlib.sha256("|".join(map(str, row)).encode()).hexdigest()[:24]
                           for row in frame[KEY].itertuples(index=False, name=None)]
    quality_path = config.output / "quality.json"
    manifest = quality_path.read_bytes() if quality_path.exists() else b"no-source-manifest"
    version = hashlib.sha256(config.version.encode() + manifest + b"features-v1").hexdigest()[:16]
    frame["feature_version"] = version
    # Comparisons are meaningful only when district mapping is verified.
    peer_key = ["month", "signal", "region", "sex", "age"]
    peers = frame.loc[frame.quality.eq("ok") & frame.region.ne("") & frame.signal.eq("mobility")].copy()
    if not peers.empty:
        peers["peer_median_decline"] = peers.groupby(peer_key).primary_decline.transform("median")
        peers["excess_decline"] = peers.primary_decline - peers.peer_median_decline
        frame = frame.merge(peers[["feature_id", "peer_median_decline", "excess_decline"]], on="feature_id", how="left", validate="one_to_one")
    save_table(config, "features", frame)
    write_json(config.output / "features_manifest.json", {"config_version": config.version, "rows": len(frame),
               "months": sorted(frame.month.unique()), "signals": frame.signal.value_counts().to_dict(),
               "quality": frame.quality.value_counts().to_dict()})
    print(f"Built {len(frame):,} feature rows", flush=True)
    return frame
