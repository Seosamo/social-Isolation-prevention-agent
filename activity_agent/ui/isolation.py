"""Dashboard for the separate, versioned isolation-suspicion task."""
from __future__ import annotations

import json

import streamlit as st

from activity_agent.common.config import Config
from activity_agent.isolation.queries import IsolationEvidenceAgent, IsolationQueryTools
from activity_agent.isolation.config import destination, load
from activity_agent.isolation.labeling import LABEL_NAMES, isolation_reviews, review_isolation


def render_isolation_dashboard(config: Config):
    st.title("고립 의심 라벨 · 학습 근거")
    st.caption("지역·집단에 대한 약한 라벨입니다. 고독사 통계는 별도 배경으로만 제공하며 판정·학습에 사용하지 않습니다.")
    readiness_path = destination(config, "readiness.json")
    if readiness_path.exists():
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        a, b, c = st.columns(3)
        a.metric("분석 표본", f"{readiness['cohort_rows']:,}")
        b.metric("독립 시군구", readiness["independent_districts"])
        c.metric("학습 준비", "완료" if readiness["training_ready"] else "자료 보완 필요")
        for reason in readiness["blockers"]:
            st.warning(reason)
    if not destination(config, "labels.parquet").exists():
        st.info("먼저 python -m activity_agent run-isolation 을 실행해 주세요.")
        return
    tools = IsolationQueryTools(config)
    labels = tools.labels
    evidence_tab, review_tab, question_tab, quality_tab = st.tabs(["고립 의심 근거", "고립 라벨 검토", "고립 근거 질의", "준비 상태·평가"])
    with evidence_tab:
        if labels.empty:
            st.info("검증된 공간 연결과 대응하는 구조 자료가 없어 결합 표본을 만들지 않았습니다. 기존 활동 분석은 별도 과제에서 사용할 수 있습니다.")
        else:
            month = st.selectbox("고립 판정 월", sorted(labels.month.unique(), reverse=True), key="iso_month")
            sex = st.selectbox("고립 분석 성별", ["전체", *sorted(labels.sex.unique())], key="iso_sex")
            age = st.selectbox("고립 분석 연령", ["전체", *sorted(labels.age_band.unique())], key="iso_age")
            selected = labels.loc[labels.month.eq(month)]
            if sex != "전체":
                selected = selected.loc[selected.sex.eq(sex)]
            if age != "전체":
                selected = selected.loc[selected.age_band.eq(age)]
            columns = ["cohort_id", "province", "district", "dong", "sex", "age_band", "label_name", "A_state", "S_state", "B_state", "label_reason"]
            if "score" in selected:
                columns += ["score", "score_kind"]
            st.dataframe(selected[columns].sort_values("label_name"), hide_index=True)
            st.download_button("고립 라벨 CSV", selected.to_csv(index=False).encode("utf-8-sig"), "isolation_labels.csv")
            if not selected.empty:
                names = {r.cohort_id: f"{r.district} {r.dong} / {r.sex} / {r.age_band} / {r.label_name}" for r in selected.itertuples()}
                chosen = st.selectbox("고립 상세 대상", list(names), format_func=names.get, key="iso_cohort")
                row = selected.loc[selected.cohort_id.eq(chosen)].iloc[0]
                st.dataframe(row.dropna().astype(str).rename("값"))
                st.caption("구조 기준은 학습 기간·지역에서만 적합했습니다. 위축 근거와 구조적 취약성의 결합은 독립적인 정답이 아닙니다.")
                st.subheader("별도 배경: 과거 고독사 통계")
                background = tools.background(chosen)
                st.dataframe(background[["province", "reference_period", "metric", "value", "available_at", "source_url"]], hide_index=True)
                with review_tab:
                    st.caption("외부 확인 자료가 없는 화면 검토는 독립 평가 정답으로 표시하지 않습니다. 기존 활동 과제의 검토 기록과 분리됩니다.")
                    with st.form("iso_review_form"):
                        value = st.selectbox("고립 검토 라벨", [-1, 0, 1], format_func=LABEL_NAMES.get)
                        reviewer = st.text_input("고립 검토자")
                        note = st.text_area("고립 판단 근거·출처")
                        independent = st.checkbox("규칙과 독립된 외부 확인 근거가 있음", value=False)
                        if st.form_submit_button("고립 검토 저장"):
                            try:
                                review_isolation(config, chosen, value, reviewer, note, independent)
                                st.success("새 과제의 검토 이력에 추가했습니다.")
                            except ValueError as error:
                                st.error(str(error))
                    reviews = isolation_reviews(config)
                    st.dataframe(reviews.loc[reviews.cohort_id.eq(chosen)], hide_index=True)
    with question_tab:
        question = st.text_input("고립 의심 질문", placeholder="2025년 12월 50대 남성 고립 의심 근거를 알려줘")
        if st.button("고립 근거 조회"):
            from activity_agent.llm.vllm import create_isolation_agent
            with st.spinner("근거를 조회하고 답변을 작성하고 있습니다."):
                st.markdown(create_isolation_agent(config, st.session_state.get("agent_backend")).answer(question))
    with quality_tab:
        for name in ("structural_quality", "cohort_manifest", "readiness", "label_summary", "label_reference", "evaluation", "models/logistic"):
            path = destination(config, name + ".json")
            if path.exists():
                with st.expander(name):
                    st.json(json.loads(path.read_text(encoding="utf-8")))
