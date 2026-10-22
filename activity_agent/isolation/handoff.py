"""Versioned, JSON-safe evidence contract for LLM tool consumers."""
from __future__ import annotations

import json
import re
import math
import numpy as np
import pandas as pd

from .anomaly import read_model, load_candidates, model_path
from .config import load


def candidate_evidence(config, month=None, sex=None, age=None, region=None, limit=10):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    if month is not None and not re.fullmatch(r"\d{4}(0[1-9]|1[0-2])", str(month)):
        raise ValueError("month must be YYYYMM")
    model = read_model(config)
    # Reuse the operating version/scope checks and candidate ordering.
    candidates = load_candidates(config, month, sex, age, region, limit)
    data = load(config, "anomaly/detections")
    selected_month = month or (str(data.month.max()) if not data.empty else None)
    selected = data.loc[data.month.eq(selected_month)].copy()
    for key, value in (("sex", sex), ("age_band", age)):
        if value:
            selected = selected.loc[selected[key].eq(value)]
    if region:
        from .structural import province_name
        selected = selected.loc[selected.geo_code.eq(region) | selected.province.eq(province_name(region))]
    evaluated = selected.anomaly_raw.notna()
    total = int(selected.candidate.sum())
    status = ("NO_DATA" if selected.empty else "NOT_EVALUABLE" if not evaluated.any()
              else "OK" if total else "NO_CANDIDATES")
    items = []
    for row in candidates.to_dict("records"):
        items.append({"evidence_id": ":".join(str(row[k]) for k in ("month", "block_code", "sex", "age_band")),
            "observation": {k: row[k] for k in ("block_code", "geo_code", "province", "sex", "age_band", "month")},
            "scores": {"raw": row["anomaly_raw"], model.get("score_reference", "training") + "_percentile": row["anomaly_percentile"]},
            "group_activity": {k: row.get(k) for k in ("activity_decline", "previous_activity_decline", "current_flow", "baseline_flow")},
            "regional_activity": {k: row[k] for k in ("daytime_decline", "previous_daytime_decline", "weekend_decline", "previous_weekend_decline")},
            "quality": {k: row.get(k) for k in ("quality_eligible", "quality", "previous_quality", "day_quality", "weekend_quality", "coverage", "status")},
            "decision": {"candidate": bool(row["candidate"]), "score_threshold_met": bool(row["anomaly_raw"] >= model["score_cutoff"]),
                "group_decline_met": bool(row["activity_decline"] >= model["anomaly_settings"]["minimum_decline"]),
                "regional_decline_met": bool(row["daytime_decline"] > 0 or row["weekend_decline"] > 0),
                "persistent_20pct": row["persistent_20pct"]}})
    result = {"schema_version": "activity-anomaly-evidence-v1", "status": status,
        "filters": {"month": selected_month, "sex": sex, "age": age, "region": region, "limit": limit},
        "model": {"id": model.get("model_id", "IF_BASELINE"), "score_reference": model.get("score_reference", "training"), "version": model["model_version"], "training_months": model["training_months"],
            "region_codes": model["region_codes"], "score_cutoff": model["score_cutoff"],
            "minimum_group_decline": model["anomaly_settings"]["minimum_decline"], "reference_quantile": model["anomaly_settings"]["reference_quantile"]},
        "summary": {"matched_rows": len(selected), "evaluated_rows": int(evaluated.sum()),
            "row_status_counts": selected.status.value_counts().to_dict(),
            "unscored_rows": int((~evaluated).sum()), "candidate_rows": total, "returned_rows": len(items), "truncated": total > len(items)},
        "semantics": {"observation_unit": "소지역 × 성별 × 연령 × 월", "regional_activity_unit": "소지역 × 월: 성·연령 집단 공통",
            "decline_unit": "비율: 0.2 = 20% 감소, 음수 = 증가", "percentile_unit": "0~100: " + ("보정 표본" if model.get("score_reference") == "calibration" else "학습") + " 점수 분포 대비 백분위",
            "raw_score": "-score_samples: 클수록 이상", "baseline": "직전 두 달 평균 유동량",
            "decision_rule": "평가 가능·학습 이후 관측 AND 원점수 >= 임계값 AND 집단 감소 >= 설정값 AND (낮 감소 > 0 OR 주말 감소 > 0)"},
        "limitations": ["이상 점수와 백분위는 고립 확률이 아닙니다.", "실제 사회고립 정답 라벨로 정확도를 검증하지 않았습니다.",
            "집단 자료로 개인의 고립 여부를 판단할 수 없습니다.", "후보 없음·관측 없음·평가 불가는 정상 확인을 의미하지 않습니다.",
            "낮·주말 변화는 지역 공통 관측이며 특정 성·연령 집단의 독립 관측이 아닙니다.", "단일 학습 월과 짧은 이력의 사후 분석이며 원인·장기 계절성을 확정할 수 없습니다."],
        "provenance": {"predictions": str(config.output / "isolation/anomaly/detections.parquet"), "model": str(model_path(config).relative_to(config.root)) if model_path(config).is_relative_to(config.root) else str(model_path(config))},
        "candidates": items}
    # Replace missing/nonfinite scalars, preserve Unicode and strict JSON types.
    def clean(value):
        if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, np.generic): value = value.item()
        if isinstance(value, float) and not math.isfinite(value): return None
        return None if value is None or (not isinstance(value, (str, bool)) and pd.isna(value)) else value
    return json.loads(json.dumps(clean(result), ensure_ascii=False, allow_nan=False))
