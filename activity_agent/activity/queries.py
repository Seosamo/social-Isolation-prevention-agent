from __future__ import annotations

import re
from typing import Protocol
from urllib.parse import urlparse

import pandas as pd

from activity_agent.common.config import Config
from activity_agent.common.presentation import percentage
from activity_agent.common.storage import load_table

OFFICIAL_HOSTS = ("bokjiro.go.kr", "gangnam.go.kr", "chuncheon.go.kr", "gov.kr", "129.go.kr")


class QueryTools:
    """Read-only tools shared by the dashboard and optional language-model adapter."""

    def __init__(self, config: Config):
        self.config = config
        self.results = load_table(config, "detections")

    def rank_activity_declines(self, signal="mobility", month=None, sex=None, age=None, region=None, limit=10):
        data = self.results
        month = month or data.month.max()
        data = data.loc[data.signal.eq(signal) & data.month.eq(month)]
        for col, value in (("sex", sex), ("age", age), ("region", region)):
            if value:
                data = data.loc[data[col].eq(value)]
        return data.loc[data.score.notna()].sort_values(["score", "feature_id"], ascending=[False, True]).head(limit).copy()

    def get_evidence(self, feature_id):
        data = self.results.loc[self.results.feature_id.eq(feature_id)]
        if len(data) != 1:
            raise ValueError("Unknown feature_id.")
        return data.iloc[0].copy()

    def compare_peers(self, feature_id, limit=10):
        row = self.get_evidence(feature_id)
        if not row.region or row.signal != "mobility" or pd.isna(row.get("baseline_flow")):
            return self.results.iloc[:0].copy()
        peers = self.results.loc[
            self.results.signal.eq(row.signal) & self.results.month.eq(row.month)
            & self.results.region.eq(row.region) & self.results.sex.eq(row.sex) & self.results.age.eq(row.age)
            & self.results.quality.eq("ok") & self.results.feature_id.ne(feature_id)].copy()
        peers = peers.loc[peers.baseline_flow.between(row.baseline_flow * 0.5, row.baseline_flow * 2)]
        peers["baseline_distance"] = (peers.baseline_flow - row.baseline_flow).abs()
        return peers.sort_values("baseline_distance").head(limit)

    def get_card_context(self, feature_id):
        row = self.get_evidence(feature_id)
        if not row.region or row.sex == "전체":
            return self.results.iloc[:0].copy()
        return self.results.loc[self.results.signal.eq("consumption") & self.results.month.eq(row.month)
                                & self.results.region.eq(row.region) & self.results.sex.eq(row.sex)
                                & self.results.age.eq(row.age)].copy()

    def get_local_patterns(self, feature_id):
        row = self.get_evidence(feature_id)
        return self.results.loc[self.results.signal.isin(["time_pattern", "weekday_pattern"])
                                & self.results.area_id.eq(row.area_id) & self.results.month.eq(row.month)].copy()

    def find_prevention_resources(self, region=None, age=None):
        path = self.config.path("resources_file")
        columns = ["resource_id", "title", "region", "age_min", "age_max", "url", "published_at", "checked_at", "description"]
        if not path.exists():
            return pd.DataFrame(columns=columns)
        data = pd.read_csv(path, dtype=str).fillna("")
        if not set(columns).issubset(data):
            raise ValueError("Resource schema is invalid.")
        if data.empty:
            return data
        def official(url):
            parsed = urlparse(url)
            return parsed.scheme == "https" and any(parsed.hostname == h or (parsed.hostname or "").endswith("." + h) for h in OFFICIAL_HOSTS)
        checked = pd.to_datetime(data.checked_at, errors="coerce", utc=True)
        data = data.loc[data.url.map(official).astype(bool) & checked.notna() & checked.le(pd.Timestamp.now(tz="UTC"))].copy()
        if region:
            data = data.loc[data.region.isin([region, "전국"])]
        else:
            # An unmapped block cannot silently receive resources for another city.
            data = data.loc[data.region.eq("전국")]
        if age and age != "전체":
            lower = int(str(age).replace("+", ""))
            upper = 120 if str(age).endswith("+") else lower + 9
            minimum = pd.to_numeric(data.age_min, errors="coerce")
            maximum = pd.to_numeric(data.age_max, errors="coerce")
            # Overlap permits age-limited resources, with the exact eligibility shown in the response.
            data = data.loc[minimum.le(upper) & maximum.ge(lower)]
        return data


class AgentBackend(Protocol):
    """Inject a provider adapter without coupling analysis to an API or credentials.

    Adapters receive read-only tools and must preserve numeric evidence and source
    attribution. No provider is configured or contacted by the default application.
    """
    def answer(self, question: str, tools: QueryTools, filters: dict) -> str: ...


from activity_agent.policies.recommendation import with_policy_response


class EvidenceAgent:
    def __init__(self, tools: QueryTools, backend: AgentBackend | None = None):
        self.tools, self.backend = tools, backend

    @with_policy_response
    def answer(self, question: str, **filters) -> str:
        if self.backend:
            return self.backend.answer(question, self.tools, filters)
        filters = {k: v for k, v in filters.items() if v is not None}
        resolved = filters.pop("_resolved", False)
        question = question.strip()
        if not question:
            return "질문을 입력해 주세요."
        if not resolved:
            # Offline routing is deliberately bounded; it is not presented as an LLM.
            month = re.search(r"(20\d{2})[-./년 ]?(0[1-9]|1[0-2])(?:월)?", question)
            if month:
                filters["month"] = "".join(month.groups())
            if "이번 달" in question or "최신" in question:
                filters["month"] = self.tools.results.month.max()
            age = re.search(r"([1-9]0)\s*대\s*(이상)?", question)
            if age:
                if int(age.group(1)) > 60 or (age.group(2) and age.group(1) != "60"):
                    return "통합 탐지의 연령 구간은 10·20·30·40·50대와 60대 이상입니다. 카드의 세부 연령은 소비 원자료 화면에서 확인할 수 있습니다."
                filters["age"] = "60+" if age.group(1) == "60" else age.group(1)
                if age.group(1) == "60" and not age.group(2):
                    return "SKT의 60대는 60대 이상으로만 제공됩니다. ‘60대 이상’으로 질문하거나 연령 필터를 선택해 주세요."
            for sex in ("남성", "여성"):
                if sex in question:
                    filters["sex"] = sex
            for token, region in (("강남", "서울 강남구"), ("춘천", "강원 춘천시")):
                if token in question:
                    filters["region"] = region
            if "주말" in question or "요일" in question:
                filters.update(signal="weekday_pattern", sex=None, age=None)
            elif "시간대" in question or "심야" in question:
                filters.update(signal="time_pattern", sex=None, age=None)
            elif "소비" in question or "카드" in question:
                filters["signal"] = "consumption"
        selected_month = filters.get("month") or self.tools.results.month.max()
        candidates = self.tools.rank_activity_declines(**filters, limit=3)
        from activity_agent.policies.matching import observe_candidates
        observe_candidates(candidates, 'activity')
        if candidates.empty:
            return (f"{selected_month} / 선택 조건에서 계산 가능한 결과가 없습니다. "
                    "기간 부족, 공간 매핑 부재, 연령 구간 또는 0인 기준값을 확인해 주세요. 자료에 없는 현황은 추정하지 않습니다.")
        lines = [f"자료 기준 {selected_month}의 활동 위축 후보입니다. 점수는 고립 확률이 아닙니다."]
        for row in candidates.itertuples(index=False):
            lines += ["", f"**{row.area_id} / {row.sex} / {row.age} — {row.score:.1f}점**",
                      f"- 신호: {row.signal}; 산출 방식: {row.engine}; 품질: {row.quality}"]
            if row.signal in ("mobility", "district_mobility"):
                lines.append(f"- 직전 2개월 평균 대비 유동 변화: 감소율 {percentage(row.activity_decline)}. 음수는 증가입니다.")
                lines.append(f"- 공통 관측 좌표 비율: {percentage(row.coverage)}.")
            elif row.signal == "consumption":
                lines += [f"- 결제건수 감소율 {percentage(row.consumption_decline)}; 사회·여가 건수 감소율 {percentage(row.social_consumption_decline)}.",
                          f"- 업종 분류가 적용된 건수 비중: {percentage(row.current_classification_coverage)}. 미분류 업종은 기타입니다."]
            elif row.signal == "time_pattern":
                lines.append(f"- 지역 전체 낮 유동 감소율 {percentage(row.daytime_decline)}. 성·연령별 시간대 통계는 아닙니다.")
            else:
                lines.append(f"- 지역 전체 주말 유동 감소율 {percentage(row.weekend_decline)}. 성·연령별 주말 통계는 아닙니다.")
            peers = self.tools.compare_peers(row.feature_id)
            lines.append(f"- 유사 규모 비교 지역 {len(peers)}곳." if not peers.empty else "- 검증된 매핑·유사 규모 비교 표본이 없어 지역 비교를 보류합니다.")
            card = self.tools.get_card_context(row.feature_id)
            if row.signal != "consumption" and not card.empty:
                lines.append(f"- 시군구 소비 맥락의 결제건수 감소율: {percentage(card.iloc[0].consumption_decline)}. 소지역 소비값이 아닙니다.")
            lines.append(f"- 근거 ID: `{row.feature_id}`")
        lines += ["", "활동 감소의 원인은 계절·상권·통근·관측 변화 등일 수 있습니다. 현장 확인 없이 고립이나 원인을 확정하지 않습니다."]
        return "\n".join(lines)
