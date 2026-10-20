from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from activity_agent.common.config import Config
from activity_agent.common.storage import fingerprint, save_table, write_json

SPATIAL_KEY = ["month", "block_code", "x", "y"]
AGE_COLUMNS = [f"{sex}_FLOW_POP_CNT_{age}" for sex in ("MAN", "WMAN")
               for age in ("10G", "20G", "30G", "40G", "50G", "60GU")]
TIME_COLUMNS = [f"TMST_{hour:02d}" for hour in range(24)]
DAY_COLUMNS = [f"FLOW_POP_CNT_{day}" for day in ("MON", "TUS", "WED", "THU", "FRI", "SAT", "SUN")]
FAMILIES = {"age": AGE_COLUMNS, "time": TIME_COLUMNS, "wkdy": DAY_COLUMNS}


def read_skt(path: Path, family: str):
    raw = pd.read_csv(path, sep="|", encoding="utf-8-sig", dtype=str)
    expected = ["STD_YM", "BLOCK_CD", "X_COORD", "Y_COORD", *FAMILIES[family]]
    if list(raw.columns) != expected:
        raise ValueError(f"Unexpected columns: {path.name}")
    frame = raw.rename(columns=dict(zip(expected[:4], SPATIAL_KEY))).copy()
    numeric = ["x", "y", *FAMILIES[family]]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    invalid = (~np.isfinite(frame[numeric]).all(axis=1)
               | frame[SPATIAL_KEY].isna().any(axis=1)
               | ~frame["month"].str.fullmatch(r"\d{4}(0[1-9]|1[0-2])", na=False)
               | (frame[FAMILIES[family]] < 0).any(axis=1)
               | frame["block_code"].eq(""))
    bad = frame.loc[invalid].assign(reason="invalid_value_or_key")
    valid = frame.loc[~invalid].copy()
    exact = valid.duplicated().sum()
    valid = valid.drop_duplicates()
    conflicts = valid.duplicated(SPATIAL_KEY, keep=False)
    bad = pd.concat([bad, valid.loc[conflicts].assign(reason="conflicting_spatial_key")])
    clean = valid.loc[~conflicts].copy()
    profile = {
        "file": path.name, "sha256": fingerprint(path), "rows_read": len(raw),
        "exact_duplicates_removed": int(exact), "quarantined": len(bad),
        "rows_clean": len(clean), "blocks": int(clean.block_code.nunique()),
        "months": sorted(clean.month.unique()),
        "raw_numeric_totals": frame[numeric[2:]].sum().to_dict(),
        "clean_numeric_totals": clean[numeric[2:]].sum().to_dict(),
    }
    return clean, bad, profile


def load_mapping(config: Config) -> pd.DataFrame:
    columns = ["block_code", "region", "source_url", "verified"]
    path = config.path("mapping_file")
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    if not set(columns).issubset(frame):
        raise ValueError("Mapping requires block_code, region, source_url, verified.")
    frame = frame.loc[frame.verified.str.lower().eq("true") & frame.source_url.str.startswith("https://")].copy()
    if frame.block_code.duplicated().any():
        raise ValueError("Verified mapping must have one row per block_code.")
    if (frame.region.str.strip() == "").any():
        raise ValueError("Verified mapping contains an empty region.")
    return frame[columns]


def industry_mapping(config: Config) -> pd.DataFrame:
    frame = pd.read_csv(config.path("industry_file"), dtype=str).fillna("")
    if not {"industry", "group", "rationale"}.issubset(frame):
        raise ValueError("Industry mapping requires industry, group, rationale.")
    if frame.industry.duplicated().any() or not frame.group.isin(["essential", "social", "other"]).all():
        raise ValueError("Industry mapping must be unique and use essential/social/other.")
    return frame


def read_card(path: Path, dataset: int, output: Path | None = None):
    group_cols = ["month", "region", "sex", "age", "industry", "time" if dataset == 1 else "customer_region"]
    result, rejected = [], 0
    invalid_chunks = []
    n = 0
    expected = {"TA_YMD", "MCT_SGG_CD", "MCT_RY_CD", "SEX_CCD", "AGE_CCD", "TS_AT", "USE_CNT",
                "TIME_GB" if dataset == 1 else "CLN_SGG_CD"}
    for chunk in pd.read_csv(path, sep="\t", encoding="cp949", dtype=str, chunksize=150_000):
        if set(chunk.columns) != expected:
            raise ValueError(f"Unexpected card columns: {path.name}")
        n += len(chunk)
        frame = chunk.rename(columns={"MCT_SGG_CD": "region", "MCT_RY_CD": "industry", "SEX_CCD": "sex",
                                      "AGE_CCD": "age", "TS_AT": "amount", "USE_CNT": "count",
                                      "TIME_GB": "time", "CLN_SGG_CD": "customer_region"}).copy()
        frame["date"] = pd.to_datetime(frame.TA_YMD, format="%Y%m%d", errors="coerce")
        frame["month"] = frame.date.dt.strftime("%Y%m")
        frame["age"] = frame.age.str.replace(" ", "", regex=False).str.replace("대", "", regex=False)
        for col in ["amount", "count"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if dataset == 2:
            frame["customer_region"] = frame.customer_region.fillna("정보없음")
        invalid = (frame.date.isna() | ~np.isfinite(frame[["amount", "count"]]).all(axis=1)
                   | (frame[["amount", "count"]] < 0).any(axis=1)
                   | frame[group_cols].isna().any(axis=1)
                   | ~frame.sex.isin(["남성", "여성", "법인"])
                   | ~frame.age.isin([str(x) for x in range(10, 100, 10)] + ["법인"])
                   | (frame.sex.eq("법인") != frame.age.eq("법인")))
        if dataset == 1:
            invalid |= ~frame.time.isin(["00_05", "06_11", "12_17", "18_23"])
        if invalid.any():
            rejected += int(invalid.sum())
            invalid_chunks.append(frame.loc[invalid].assign(reason="invalid_card_value_or_key"))
        valid = frame.loc[~invalid]
        result.append(valid.groupby(group_cols, dropna=False, observed=True)[["amount", "count"]].sum().reset_index())
    if output is not None:
        quarantined = pd.concat(invalid_chunks) if invalid_chunks else frame.iloc[:0].assign(reason=pd.Series(dtype=str))
        quarantined.to_parquet(output, index=False)
    clean = pd.concat(result).groupby(group_cols, observed=True)[["amount", "count"]].sum().reset_index()
    profile = {"file": path.name, "sha256": fingerprint(path), "rows_read": n,
               "quarantined": rejected, "aggregate_rows": len(clean), "months": sorted(clean.month.unique()),
               "regions": sorted(clean.region.unique()), "industries": sorted(clean.industry.unique()),
               "total_count": float(clean["count"].sum()), "total_amount": float(clean.amount.sum())}
    return clean, profile


def preprocess(config: Config, validate_only: bool = False) -> dict:
    report = {"config_version": config.version, "files": [], "warnings": []}
    data = config.path("data_dir")
    skt = sorted(data.glob("**/flow_*_pop_*.csv"))
    cards = sorted(data.glob("**/*데이터[12].txt"))
    if not skt or len(cards) != 2:
        raise ValueError("Expected SKT CSVs and exactly two card data files under data_dir.")
    quarantine = config.output / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    discovered = set()
    for path in skt:
        family = path.name.split("_")[1]
        if family not in FAMILIES:
            raise ValueError(f"Unknown SKT family: {path.name}")
        frame, bad, profile = read_skt(path, family)
        month = path.stem.rsplit("_", 1)[1]
        if set(frame.month) - {month}:
            raise ValueError(f"Filename/month mismatch: {path.name}")
        if (family, month) in discovered:
            raise ValueError(f"Duplicate family/month file: {family}/{month}")
        discovered.add((family, month))
        report["files"].append(profile)
        if not validate_only:
            save_table(config, f"clean/skt_{family}_{month}", frame)
            # Always overwrite so a resolved quarantine does not leave stale rows.
            bad.to_parquet(quarantine / f"{path.stem}.parquet", index=False)
    for path in cards:
        dataset = int(path.stem[-1])
        frame, profile = read_card(path, dataset, None if validate_only else quarantine / f"card{dataset}.parquet")
        report["files"].append(profile)
        if not validate_only:
            save_table(config, f"clean/card{dataset}", frame)
    mapping = load_mapping(config)
    if mapping.empty:
        report["warnings"].append("No verified block/region mapping: combined signals and regional mobility comparisons are unavailable.")
    if not (config.values["source_crs"] and config.values["verified_crs"]):
        report["warnings"].append("Source CRS is unverified: geographic map disabled.")
    if not validate_only:
        industry = industry_mapping(config)
        inventory = pd.DataFrame({"industry": sorted(set().union(*(set(p.get("industries", [])) for p in report["files"])))})
        inventory = inventory.merge(industry, on="industry", how="left", validate="one_to_one").fillna({"group": "other", "rationale": "미분류: 업종 검토 필요"})
        inventory.to_csv(config.output / "industry_inventory.csv", index=False, encoding="utf-8-sig")
    if validate_only:
        write_json(config.output / "validation_quality.json", report)
    else:
        write_json(config.output / "quality.json", report)
        write_json(config.output / "clean_manifest.json", {
            "skt_files": [f"clean/skt_{family}_{month}.parquet" for family, month in sorted(discovered)],
            "card_files": ["clean/card1.parquet", "clean/card2.parquet"]})
    print(f"{'Validated' if validate_only else 'Preprocessed'} {len(report['files'])} files", flush=True)
    return report
