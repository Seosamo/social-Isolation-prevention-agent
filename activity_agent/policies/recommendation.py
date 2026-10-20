"""Retrieve recent official policy evidence without asserting eligibility."""
import json
from datetime import datetime, timezone

from activity_agent.policies.collector import official


def recommend(config, region=None, limit=3, now=None):
    path = config.output / "policies" / "catalog.json"
    if not path.exists() or not region:
        return []
    now = now or datetime.now(timezone.utc)
    max_days = config.values.get("policies", {}).get("max_age_days", 7)
    rows = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        checked = datetime.fromisoformat(row["checked_at"])
        if checked.tzinfo is None:
            continue
        age_days = (now - checked).total_seconds() / 86400
        if 0 <= age_days <= max_days and official(row["url"]) and row["region"].split()[-1] == region.split()[-1]:
            rows.append(row)
    return sorted(rows, key=lambda r: r["checked_at"], reverse=True)[:limit]


def policy_lines(config, region=None):
    rows = recommend(config, region)
    if not rows:
        return ["- 최근 수집된 공식 정책 후보가 없습니다. policies-refresh 실행 또는 수집 상태 확인이 필요합니다."]
    return [f"- 공식 정책 확인 후보: [{r['title']}]({r['url']}) / 수집일 {r['checked_at'][:10]}. "
            f"{r['eligibility']}; {r['recruitment']}. 현재 지원 가능 여부를 원문에서 확인해 주세요." for r in rows]
