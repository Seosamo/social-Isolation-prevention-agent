"""Normalize external aggregate statistics without manufacturing geographic detail."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from activity_agent.common.config import Config
from activity_agent.common.storage import fingerprint
from activity_agent.isolation.config import reference_path, report_json, save

PROVINCES = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시",
             "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도",
             "전북특별자치도", "전라남도", "경상북도", "경상남도", "제주특별자치도"]
ALIASES = {"서울": "서울특별시", "강원": "강원특별자치도", "강원도": "강원특별자치도",
           "전라북도": "전북특별자치도"}
NAMES = ["province", "district", "dong"]
STRUCTURAL_COLUMNS = ["source_id", "reference_period", "available_at", "source_url", "source_hash", "geo_level", "geo_code",
                      *NAMES, "raw_geography", "sex", "age_band", "metric", "value", "denominator_kind", "quality", "mapping_quality"]


def province_name(value: str) -> str:
    value = str(value).strip()
    return ALIASES.get(value, value)


def read_rows(path: Path):
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return list(csv.reader(raw.decode(encoding).splitlines())), encoding
        except UnicodeError:
            continue
    raise ValueError(f"Unsupported CSV encoding: {path}")


def number(value) -> float:
    # '-' is missing, never an implicit zero.
    try:
        result = float(str(value).replace(",", "").strip())
        return result if np.isfinite(result) and result >= 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def read_geography(config: Config) -> pd.DataFrame:
    columns = ["geo_level", "geo_code", *NAMES, "parent_code", "reference_year", "source_url", "verified"]
    path = reference_path(config, "geography_file")
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    if not set(columns).issubset(frame):
        raise ValueError("Invalid geography registry columns")
    frame = frame.loc[frame.verified.str.lower().eq("true") & frame.source_url.str.startswith("https://")].copy()
    frame["province"] = frame.province.map(province_name)
    for col in ["geo_code", "district", "dong", "reference_year", "parent_code"]:
        frame[col] = frame[col].str.strip()
    key = ["geo_level", "geo_code", "reference_year"]
    name_key = ["geo_level", *NAMES, "reference_year"]
    if frame.duplicated(key).any() or frame.duplicated(name_key).any():
        raise ValueError("Geography codes and full names must be unique per level/year")
    if not frame.geo_level.isin(["sido", "sigungu", "dong"]).all() or frame.geo_code.eq("").any():
        raise ValueError("Invalid geography level/code")
    if not frame.reference_year.str.fullmatch(r"\d{4}").all():
        raise ValueError("Geography registry requires four-digit reference_year")
    return frame[columns]


def location(parts: list[str], current_province="", current_district="") -> dict:
    """Resolve only explicit hierarchy, leaving ambiguous names unmapped."""
    original = " / ".join(parts)
    parts = [p.strip() for p in parts if p.strip() not in ("", "소계", "합계")]
    if not parts or parts[0] == "전국":
        return dict(geo_level="national", province="", district="", dong="", raw_geography=original)
    if parts[0] in ("동부", "읍부", "면부"):
        return dict(geo_level="settlement", province="", district=parts[0], dong="", raw_geography=original)
    explicit = province_name(parts[0])
    province_suffixes = ("특별시", "광역시", "특별자치시", "특별자치도", "도")
    if explicit in PROVINCES or explicit.endswith(province_suffixes):
        province, rest = explicit, parts[1:]
    else:
        tokens = parts[0].split()
        if province_name(tokens[0]) in PROVINCES or province_name(tokens[0]).endswith(province_suffixes):
            province, rest = province_name(tokens[0]), [*tokens[1:], *parts[1:]]
        else:
            province, rest = current_province, parts
    # A province nested under another province (e.g. merged statistical
    # hierarchies) needs an explicit vintage crosswalk. Never drop the third
    # name or silently reinterpret it as a dong.
    if rest and province_name(rest[0]) in PROVINCES:
        return dict(geo_level="unresolved", province=province, district=rest[0],
                    dong=" / ".join(rest[1:]), raw_geography=original)
    if len(rest) > 1 and rest[-1].endswith(("동", "읍", "면")):
        district, dong = " ".join(rest[:-1]), rest[-1]
    else:
        district, dong = " ".join(rest), ""
    if district.endswith(("동", "읍", "면")) and not dong:
        dong, district = district, current_district
    return dict(geo_level="dong" if dong else "sigungu" if district else "sido",
                province=province, district=district, dong=dong, raw_geography=original)


def consistent(total, *parts) -> bool:
    return all(np.isfinite(v) for v in (total, *parts)) and abs(total - sum(parts)) < 1e-6


def normalize_source(source: dict, rows: list[list[str]]) -> pd.DataFrame:
    source_id = source["source_id"]
    out = []

    def emit(loc, sex, age, metric, value, denominator, period=None, quality="ok"):
        value = number(value)
        out.append({"source_id": source_id, "reference_period": period or source["reference_period"],
                    "available_at": source.get("available_at", ""), "source_url": source.get("url", ""),
                    "source_hash": source["source_hash"], **loc, "sex": sex, "age_band": age,
                    "metric": metric, "value": value, "denominator_kind": denominator,
                    "quality": quality if np.isfinite(value) else "missing_value"})

    if source_id == "lonely_deaths":
        if len(rows[0]) != 4:
            raise ValueError("Unexpected lonely-deaths layout")
        for row in rows[2:]:
            loc = location([row[0]])
            for metric, val, denominator in zip(["lonely_deaths_count", "lonely_deaths_per_100k", "lonely_deaths_share"], row[1:],
                                                ["persons", "population_100k", "all_deaths"]):
                emit(loc, "전체", "전체", metric, val, denominator)
    elif source_id == "elderly":
        if len(rows[0]) != 15:
            raise ValueError("Unexpected elderly layout")
        for row in rows[5:]:
            loc = location(row[:3])
            vals = [number(v) for v in row[3:]]
            groups = [vals[i:i + 3] for i in (0, 3, 6, 9)]
            mismatch = any(all(np.isfinite(g)) and not consistent(*g) for g in groups)
            mismatch |= any(all(np.isfinite(v) for v in [vals[j], vals[j+3], vals[j+6], vals[j+9]])
                            and not consistent(vals[j], vals[j+3], vals[j+6], vals[j+9]) for j in range(3))
            quality = "sum_mismatch" if mismatch else "ok"
            for index, sex in enumerate(["전체", "남성", "여성"]):
                total, benefit, low = vals[index], vals[index + 3], vals[index + 6]
                emit(loc, sex, "65+", "elderly_count", total, "persons", quality=quality)
                ratio = (benefit + low) / total if total > 0 else np.nan
                ratio_quality = quality if not np.isfinite(ratio) or 0 <= ratio <= 1 else "ratio_mismatch"
                emit(loc, sex, "65+", "elderly_poverty_share", ratio, "elderly_living_alone", quality=ratio_quality)
    elif source_id == "elderly_households":
        if len(rows[0]) != 6:
            raise ValueError("Unexpected elderly-household layout")
        for row in rows[2:]:
            loc = location(row[:3])
            rate, count, total = map(number, row[3:])
            valid = total > 0 and np.isfinite(count) and np.isfinite(rate) and abs(rate - count / total * 100) <= 0.11 and count <= total
            quality = "ok" if valid else "ratio_mismatch"
            emit(loc, "전체", "65+", "elderly_household_ratio", rate / 100, "general_households", quality=quality)
            emit(loc, "전체", "65+", "elderly_household_count", count, "general_households", quality=quality)
            emit(loc, "전체", "전체", "general_households", total, "general_households", quality=quality)
    elif source_id == "single_households":
        if len(rows[0]) != 5:
            raise ValueError("Expected region/age/total/male/female single-household layout")
        province, district = "", ""
        parsed = []
        for row in rows[2:]:
            loc = location([row[0]], province, district)
            if loc["geo_level"] == "sido":
                province, district = loc["province"], ""
            elif loc["geo_level"] == "sigungu":
                district = loc["district"]
            total, male, female = map(number, row[2:])
            quality = "ok" if consistent(total, male, female) else "sum_mismatch"
            for sex, val in zip(["전체", "남성", "여성"], [total, male, female]):
                parsed.append({**loc, "sex": sex, "raw_age": row[1].strip(), "value": val, "quality": quality})
        frame = pd.DataFrame(parsed)
        keys = ["geo_level", *NAMES, "raw_geography"]
        for _, part in frame.groupby(keys, dropna=False, sort=False):
            loc = part.iloc[0][keys].to_dict()
            totals = part.loc[part.raw_age.eq("합계")]
            denominator = totals.loc[totals.sex.eq("전체"), "value"]
            denominator = denominator.iloc[0] if len(denominator) == 1 else np.nan
            for sex in ["전체", "남성", "여성"]:
                age_rows = part.loc[part.sex.eq(sex) & ~part.raw_age.eq("합계")].copy()
                buckets = {}
                for r in age_rows.itertuples():
                    numbers = [int(v) for v in re.findall(r"\d+", r.raw_age)]
                    if not numbers:
                        continue
                    if "미만" in r.raw_age:
                        age, covered = "under20", set(range(numbers[0]))
                    else:
                        lower = numbers[0]
                        upper = numbers[1] if len(numbers) > 1 else 120 if "이상" in r.raw_age else lower + 9 if "대" in r.raw_age else lower + 4
                        age = "60+" if lower >= 60 else str(lower // 10 * 10)
                        covered = set(range(lower, upper + 1))
                    buckets.setdefault(age, []).append((r.value, r.quality, covered))
                for age, values in buckets.items():
                    covered = set.union(*(v[2] for v in values))
                    required = set(range(60, 121)) if age == "60+" else set(range(20)) if age == "under20" else set(range(int(age), int(age) + 10))
                    overlaps = sum(len(v[2]) for v in values) != len(covered)
                    complete = required == covered and not overlaps
                    count = sum(v[0] for v in values)
                    valid = (complete and all(v[1] == "ok" and np.isfinite(v[0]) for v in values)
                             and denominator > 0 and count <= denominator and totals.quality.eq("ok").all())
                    quality = "ok" if valid else "incomplete_age_or_invalid_total"
                    emit(loc, sex, age, "single_household_count", count, "single_households", quality=quality)
                    emit(loc, sex, age, "single_household_share", count / denominator if denominator > 0 else np.nan,
                         "all_single_households", quality=quality)
    elif source_id == "registered_households":
        province, district = "", ""
        for row in rows[1:]:
            loc = location([row[0]], province, district)
            if loc["geo_level"] == "sido":
                province, district = loc["province"], ""
            elif loc["geo_level"] == "sigungu":
                district = loc["district"]
            for period, val in zip(rows[0][1:], row[1:]):
                emit(loc, "전체", "전체", "registered_households", val, "registered_households", period=period.replace(".", ""))
    else:
        raise ValueError(f"Unknown structural source_id: {source_id}")
    return pd.DataFrame(out)


def load_structural_sources(config: Config) -> pd.DataFrame:
    metadata_path = reference_path(config, "sources_file")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing source registry: {metadata_path}")
    sources = pd.read_csv(metadata_path, dtype=str).fillna("")
    required = {"source_id", "path", "reference_period", "available_at", "url"}
    if not required.issubset(sources) or sources.source_id.duplicated().any():
        raise ValueError("Source registry requires unique source_id, path, reference_period, available_at, url")
    registry = read_geography(config)
    outputs, profiles, warnings = [], [], []
    for source in sources.to_dict("records"):
        path = Path(source["path"])
        path = path if path.is_absolute() else config.root / path
        if not path.exists():
            warnings.append(f"Missing source: {source['source_id']} ({path})")
            continue
        rows, encoding = read_rows(path)
        if not rows:
            raise ValueError(f"Empty structural source: {path}")
        if any(len(row) != len(rows[0]) for row in rows):
            raise ValueError(f"Inconsistent CSV column counts: {path.name}")
        if not re.fullmatch(r"20\d{2}", source["reference_period"]):
            raise ValueError(f"This source layout requires a four-digit reference year: {path.name}")
        source["source_hash"] = fingerprint(path)
        years = {v[:4] for v in rows[0][1:] if re.fullmatch(r"20\d{2}(?:\.\d{2})?", v)}
        # Elderly and ratio files have geographic columns before numeric years.
        if years and years != {source["reference_period"][:4]}:
            raise ValueError(f"Registry/header year mismatch in {path.name}")
        available = source["available_at"]
        if available and (not re.fullmatch(r"\d{4}-\d{2}-\d{2}", available) or pd.isna(pd.to_datetime(available, errors="coerce"))):
            raise ValueError(f"Invalid available_at for {source['source_id']}")
        part = normalize_source(source, rows)
        if part.empty:
            continue
        part["reference_year"] = part.reference_period.str[:4]
        part = part.merge(registry[["geo_level", *NAMES, "reference_year", "geo_code"]],
                          on=["geo_level", *NAMES, "reference_year"], how="left", validate="many_to_one")
        part["geo_code"] = part.geo_code.fillna("")
        part["mapping_quality"] = np.where(part.geo_code.ne(""), "verified", "unmapped")
        key = ["reference_period", "geo_level", *NAMES, "sex", "age_band", "metric"]
        duplicate = part.duplicated(key, keep=False)
        part.loc[duplicate, "quality"] = "duplicate_structural_key"
        outputs.append(part[STRUCTURAL_COLUMNS])
        profiles.append({"source_id": source["source_id"], "file": path.name, "encoding": encoding,
                         "sha256": source["source_hash"], "rows": len(part), "reference_period": source["reference_period"],
                         "available_at": available or None, "geo_levels": part.geo_level.value_counts().to_dict(),
                         "quality": part.quality.value_counts().to_dict(), "mapped_rows": int(part.geo_code.ne("").sum())})
        if source["source_id"] == "single_households" and not part.geo_level.isin(["sigungu", "dong"]).any():
            warnings.append("single_households contains no district/dong observations; filename does not establish geographic scope")
        if part.geo_level.eq("unresolved").any():
            warnings.append(f"{source['source_id']} contains nested province hierarchies; verify boundary vintage/crosswalk before use")
    frame = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=STRUCTURAL_COLUMNS)
    save(config, "structural", frame)
    report_json(config, "structural_quality", {"sources": profiles, "warnings": warnings,
                "geography_registry_rows": len(registry), "rows": len(frame)})
    print(f"Structural: {len(frame):,} normalized rows; {len(warnings)} scope/source warnings", flush=True)
    return frame
