"""Audit SKT coordinate assignments against an official province boundary file.

The provider documents UTM-K without its datum. Compare both known candidates;
write draft mappings, never silently certify a provider coordinate reference system.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from activity_agent.common.config import Config
from activity_agent.common.storage import fingerprint
from activity_agent.isolation.config import destination, report_json
from activity_agent.isolation.structural import province_name

BOUNDARY_SOURCE = "https://www.data.go.kr/data/15129688/fileData.do"
CANDIDATE_CRS = (5179, 5178)


def assign_coordinates(points, polygons, boundary_crs):
    """Return one code per coordinate only if both UTM-K interpretations agree."""
    import shapely

    assignments = {}
    for candidate in CANDIDATE_CRS:
        x, y = Transformer.from_crs(candidate, boundary_crs, always_xy=True).transform(
            points.x.to_numpy(), points.y.to_numpy())
        codes = np.full(len(points), "", dtype=object)
        hits = np.zeros(len(points), dtype=int)
        for code, polygon in polygons.items():
            shapely.prepare(polygon)
            inside = shapely.intersects_xy(polygon, x, y)
            codes[inside] = code
            hits += inside
        codes[hits != 1] = ""
        assignments[candidate] = codes
    result = points.copy()
    for candidate, codes in assignments.items():
        result[f"sido_{candidate}"] = codes
    same = (assignments[5179] == assignments[5178]) & (assignments[5179] != "")
    result["geo_code"] = np.where(same, assignments[5179], "")
    result["assignment_status"] = np.where(same, "candidate_crs_agree", "unresolved_or_crs_disagreement")
    return result


def audit_spatial_mapping(config: Config, boundary_file: str | Path) -> dict:
    import shapefile
    import shapely

    boundary = Path(boundary_file).resolve()
    boundary_crs = CRS.from_wkt(boundary.with_suffix(".prj").read_text())
    reader = shapefile.Reader(str(boundary), encoding="utf-8")
    polygons, names = {}, {}
    dates = set()
    for item in reader.iterShapeRecords():
        record = item.record.as_dict()
        code = str(record["SIDO_CD"])
        names[code] = province_name(record["SIDO_NM"])
        dates.add(record["BASE_DATE"])
        polygons[code] = shapely.make_valid(shapely.geometry.shape(item.shape.__geo_interface__))
    manifest = json.loads((config.output / "clean_manifest.json").read_text(encoding="utf-8"))
    pieces = []
    for filename in manifest["skt_files"]:
        month = filename.rsplit("_", 1)[1].split(".")[0]
        part = pd.read_parquet(config.output / filename, columns=["block_code", "x", "y"])
        pieces.append(part.assign(reference_year=month[:4]))
    if not pieces:
        raise ValueError("No clean SKT coordinates; run preprocess first")
    points = pd.concat(pieces, ignore_index=True).drop_duplicates().reset_index(drop=True)
    assigned = assign_coordinates(points, polygons, boundary_crs)
    # A block can contain many coordinates. Retain only blocks whose every point
    # across every family/month agrees on one province; never use a mean point.
    rows = []
    for (block, year), group in assigned.groupby(["block_code", "reference_year"], sort=True):
        codes = set(group.geo_code)
        accepted = len(codes) == 1 and "" not in codes
        code = next(iter(codes)) if accepted else ""
        rows.append({"block_code": block, "geo_level": "sido", "geo_code": code,
                     "province": names.get(code, ""), "reference_year": year,
                     "source_url": BOUNDARY_SOURCE, "verified": False,
                     "assignment_status": "candidate_crs_agree" if accepted else "unresolved_block",
                     "coordinate_count": len(group)})
    blocks = pd.DataFrame(rows)
    destination(config, "").mkdir(parents=True, exist_ok=True)
    assigned.to_parquet(destination(config, "spatial_coordinates.parquet"), index=False)
    blocks.to_csv(destination(config, "block_geography_candidates.csv"), index=False, encoding="utf-8-sig")
    accepted = blocks.loc[blocks.geo_code.ne("")]
    report = {
        "boundary_source": BOUNDARY_SOURCE, "boundary_file": str(boundary),
        "boundary_dates": sorted(dates), "boundary_sha256": fingerprint(boundary),
        "boundary_crs": boundary_crs.to_string(), "candidate_source_epsg": list(CANDIDATE_CRS),
        "coordinate_rows": len(assigned), "block_year_rows": len(blocks),
        "candidate_mapping_sha256": fingerprint(destination(config, "block_geography_candidates.csv")),
        "input_manifest_sha256": fingerprint(config.output / "clean_manifest.json"),
        "coordinate_status": assigned.assignment_status.value_counts().to_dict(),
        "block_status": blocks.assignment_status.value_counts().to_dict(),
        "provinces": [{"geo_code": code, "province": names[code], "blocks": len(group)}
                      for code, group in accepted.groupby("geo_code")],
        "verified": False,
        "note": "Draft spatial inference: UTM-K datum is undocumented. Both candidate CRSs must agree at every observed block coordinate. No authoritative reference mapping is overwritten.",
    }
    report_json(config, "spatial_audit", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report
