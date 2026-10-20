from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from activity_agent.common.config import Config
from activity_agent.common.storage import load_table, save_table, write_json


def report(config: Config):
    features = load_table(config, "features")
    detections = load_table(config, "detections")
    labels = load_table(config, "weak_labels")
    quality = json.loads((config.output / "quality.json").read_text(encoding="utf-8"))
    sensitivity = []
    for (month, signal), part in labels.groupby(["month", "signal"]):
        if signal not in ("mobility", "district_mobility", "consumption"):
            continue
        valid = part.quality.eq("ok") & part.previous_quality.eq("ok")
        reference = set(part.loc[part.label.eq(1), "feature_id"])
        for threshold in (0.15, 0.20, 0.25, 0.30):
            selected = valid & part.primary_decline.ge(threshold) & part.previous_decline.ge(threshold)
            if signal == "consumption":
                selected &= part.social_consumption_decline.ge(threshold) & part.previous_social_consumption_decline.ge(threshold)
            ids = set(part.loc[selected, "feature_id"])
            union = ids | reference
            sensitivity.append({"month": month, "signal": signal, "threshold": threshold, "suspect_count": len(ids),
                                "jaccard_vs_configured_labels": len(ids & reference) / len(union) if union else None})
    save_table(config, "threshold_sensitivity", pd.DataFrame(sensitivity))
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config.values,
               "config_version": config.version, "feature_version": features.feature_version.unique().tolist(),
               "feature_rows": len(features), "scored_rows": int(detections.score.notna().sum()),
               "labels": labels.label_name.value_counts().to_dict(), "warnings": quality["warnings"]}
    write_json(config.output / "run_manifest.json", summary)
    lines = ["# 지역 활동 위축 분석 보고서", "", "개인 고립 판정이나 검증된 고립 확률을 제공하지 않습니다.", "",
             f"- 분석 기간: {features.month.min()} ~ {features.month.max()}",
             f"- 특징 행: {len(features):,}", f"- 점수 산출 행: {summary['scored_rows']:,}",
             f"- 라벨 분포: {summary['labels']}", f"- 설정 버전: {config.version}",
             "", "## 품질과 가용 범위", ""]
    lines += [f"- {w}" for w in quality["warnings"]]
    for item in quality["files"]:
        lines.append(f"- {item['file']}: 원본 {item['rows_read']:,}행, 격리 {item['quarantined']:,}행, 완전 중복 제거 {item.get('exact_duplicates_removed', 0):,}행")
    lines += ["", "## 해석", "", "- 규칙 라벨은 활동 위축의 대리지표이며 독립적인 정답이 아닙니다.",
              "- 주말·시간대는 지역 전체 집계이며 성·연령별 교차통계가 아닙니다.",
              "- 카드 지표는 가맹점 소재지 기준이며 주민의 소비나 동일인의 이동을 의미하지 않습니다.",
              "- threshold_sensitivity.parquet에서 라벨 임계값 민감도를 확인할 수 있습니다.",
              "- 분석 자료가 6개월이므로 연간 계절성·전년 동월 비교를 검증할 수 없습니다."]
    (config.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {config.output / 'REPORT.md'}", flush=True)
    return summary
