"""Read-only evidence presentation. Background deaths never alter predictions."""
from __future__ import annotations

import json
import re

import pandas as pd

from activity_agent.common.presentation import percentage
from activity_agent.common.config import Config
from activity_agent.isolation.config import destination, load


class IsolationQueryTools:
    def __init__(self, config: Config):
        self.config = config
        labels = load(config, "labels")
        cohorts = load(config, "cohorts")
        # Do not show stale decisions after rebuilding the cohort dataset.
        self.labels = labels.merge(cohorts[["cohort_id", "feature_version"]], on=["cohort_id", "feature_version"], validate="one_to_one")
        prediction_path = destination(config, "detections.parquet")
        if prediction_path.exists():
            prediction = pd.read_parquet(prediction_path)
            keys = ["cohort_id", "feature_version", "label_version"]
            self.labels = self.labels.merge(prediction[[*keys, "score", "score_kind", "engine_version"]], on=keys,
                                            how="left", validate="one_to_one")

    def candidates(self, month=None, sex=None, age=None, region=None, limit=10):
        data = self.labels
        if data.empty:
            return data.copy()
        data = data.loc[data.month.eq(month or data.month.max())]
        for col, value in (("sex", sex), ("age_band", age)):
            if value:
                data = data.loc[data[col].eq(value)]
        if region:
            from activity_agent.isolation.structural import province_name
            data = data.loc[data.district.str.contains(region.split()[-1], regex=False)
                            | data.province.eq(province_name(region)) | data.geo_code.eq(region)]
        candidates = data.label.eq(1)
        if "score" in data:
            candidates |= data.score.ge(50)
        return data.loc[candidates].sort_values(["activity_decline", "cohort_id"], ascending=[False, True]).head(limit).copy()

    def background(self, cohort_id):
        row = self.labels.loc[self.labels.cohort_id.eq(cohort_id)]
        data = load(self.config, "structural")
        if len(row) != 1:
            return data.iloc[:0].copy()
        row = row.iloc[0]
        data = data.loc[data.source_id.eq("lonely_deaths") & data.geo_level.eq("sido") & data.province.eq(row.province)
                        & data.reference_period.lt(row.month[:4]) & data.quality.eq("ok")].copy()
        if row.time_mode == "as_of":
            dates = pd.to_datetime(data.available_at, errors="coerce", utc=True)
            data = data.loc[dates.notna() & dates.le(pd.Timestamp(row.decision_at, tz="UTC"))]
        return data


class IsolationEvidenceAgent:
    def __init__(self, config: Config, backend=None):
        self.config = config
        self.is_anomaly = config.values.get("isolation", {}).get("method") == "isolation_forest"
        self.tools = AnomalyQueryTools(config) if self.is_anomaly else IsolationQueryTools(config)
        self.backend = backend

    def answer(self, question, month=None, sex=None, age=None, region=None, _resolved=False):
        if self.backend:
            return self.backend.answer_isolation(question, self, dict(month=month, sex=sex, age=age, region=region))
        if not _resolved:
            date = re.search(r"(20\d{2})[-./년 ]?(0[1-9]|1[0-2])", question)
            if date:
                month = "".join(date.groups())
            for candidate in ("남성", "여성"):
                if candidate in question:
                    sex = candidate
            band = re.search(r"([1-9]0)\s*대", question)
            if band:
                if int(band.group(1)) > 60:
                    return "통합 활동 자료는 60세 이상을 하나의 집단으로 제공합니다. 더 세분된 고립 의심 판정은 제공하지 않습니다."
                age = "60+" if band.group(1) == "60" else band.group(1)
            for token, name in (("강남", "강남구"), ("춘천", "춘천시")):
                if token in question:
                    region = name
            if self.config.values.get("isolation", {}).get("geo_level", "sido") == "sido":
                from activity_agent.isolation.structural import PROVINCES, ALIASES
                for token in sorted([*PROVINCES, *ALIASES], key=len, reverse=True):
                    if token in question:
                        region = ALIASES.get(token, token)
                        break
        candidates = self.tools.candidates(month, sex, age, region, limit=3)
        if self.is_anomaly:
            if candidates.empty:
                return "선택 조건에서 활동 위축 이상 후보가 없습니다. 고립이 없다는 확인 결과는 아닙니다."
            lines = ["서울·강원 관측 소지역의 활동 위축 이상 후보입니다. 고립 확률이나 개인 판정이 아닙니다."]
            for row in candidates.itertuples(index=False):
                lines += [f"\n**{row.province} / 소지역 {row.block_code} / {row.sex} / {row.age_band} / {row.month}**",
                          f"- 활동 이상 백분위 {row.anomaly_percentile:.1f}/100: 학습 분포와 비교한 순위이며 확률이 아닙니다.",
                          f"- 유동 감소율 {percentage(row.activity_decline)}, 직전 비교 {percentage(row.previous_activity_decline)}.",
                          f"- 낮 감소율 {percentage(row.daytime_decline)}, 주말 감소율 {percentage(row.weekend_decline)}.",
                          f"- 두 연속 20% 위축: {'해당' if row.persistent_20pct else '미해당'}. 기준 유동량 {row.baseline_flow:.2f}.",
                          "- 한 달의 학습 분포를 사용한 활동 탐색 결과입니다. 계절성이나 실제 고립 탐지 성능은 검증되지 않았습니다."]
                if pd.notna(row.province_structure_context):
                    lines.append(f"- 시도 단위 배경: 전체 1인가구 중 해당 성·연령 구성비 {percentage(row.province_structure_context)}. 소지역의 독립 관측값은 아닙니다.")
                if any(word in question for word in ("예방", "추천", "지원", "프로그램", "정책")):
                    from activity_agent.policies.recommendation import policy_lines
                    lines.extend(policy_lines(self.config, row.province))
            return "\n".join(lines)
        if candidates.empty:
            readiness_path = destination(self.config, "readiness.json")
            blockers = json.loads(readiness_path.read_text(encoding="utf-8")).get("blockers", []) if readiness_path.exists() else []
            return "선택 조건에서 고립 의심 약한 라벨 후보가 없습니다. 고립이 없다는 확인 결과는 아닙니다." + ("\n\n현재 준비 상태: " + "; ".join(blockers) if blockers else "")
        lines = ["지역·집단의 고립 의심 규칙·모델 후보입니다. 개인 판정이나 검증된 고립 확률이 아닙니다."]
        for row in candidates.itertuples(index=False):
            lines += ["", f"**{row.province} {row.district} {row.dong} / {row.sex} / {row.age_band} / {row.month}**",
                      f"- 유동 감소율 {percentage(row.activity_decline)}, 직전 비교 감소율 {percentage(row.previous_activity_decline)}.",
                      f"- 구조 근거: {row.structure_profile}, 값 {percentage(row.structure_value)}; 학습 기준 {percentage(row.S_threshold)}.",
                      f"- A={row.A_state}, S={row.S_state}, B={row.B_state}. 근거 ID `{row.cohort_id}`.",
                      f"- 시간 모드: {row.time_mode}. 연간 자료를 사용한 사후 분석은 당시 실시간 예측과 구분합니다."]
            lines.append(f"- 규칙 라벨: {row.label_name}.")
            if hasattr(row, "score") and pd.notna(row.score):
                lines.append(f"- 모델 분류 점수: {row.score:.1f}/100. 규칙 라벨과 다를 수 있으며 고립 확률이 아닙니다.")
            background = self.tools.background(row.cohort_id)
            rate = background.loc[background.metric.eq("lonely_deaths_per_100k")]
            if not rate.empty:
                r = rate.sort_values("reference_period").iloc[-1]
                lines.append(f"- 별도 배경: {r.reference_period}년 {r.province} 고독사 발생률은 인구 10만 명당 {r.value:.1f}명. 라벨·모델 입력에는 사용하지 않았습니다.")
            if any(word in question for word in ("예방", "추천", "지원", "프로그램", "정책")):
                from activity_agent.policies.recommendation import policy_lines
                lines.extend(policy_lines(self.config, row.district or row.province))
        return "\n".join(lines)


class AnomalyQueryTools:
    def candidate_evidence(self, month=None, sex=None, age=None, region=None, limit=10):
        from activity_agent.isolation.handoff import candidate_evidence
        return candidate_evidence(self.config, month, sex, age, region, limit)

    def __init__(self, config):
        from activity_agent.isolation.anomaly import read_model
        read_model(config)
        self.config = config
        self.labels = load(config, "anomaly/detections")

    def candidates(self, month=None, sex=None, age=None, region=None, limit=10):
        from activity_agent.isolation.anomaly import load_candidates
        return load_candidates(self.config, month, sex, age, region, limit)
