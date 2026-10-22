"""Dashboard for the separate, versioned isolation-suspicion task."""
from __future__ import annotations

import json

import streamlit as st

from activity_agent.common.config import Config
from activity_agent.isolation.queries import IsolationEvidenceAgent, IsolationQueryTools
from activity_agent.isolation.config import destination, load
from activity_agent.isolation.labeling import LABEL_NAMES, isolation_reviews, review_isolation


def render_isolation_dashboard(config: Config):
    if config.values.get("isolation", {}).get("method") == "isolation_forest":
        render_anomaly_dashboard(config)
        return
    st.title("고립 의심 라벨 · 학습 근거")
    st.caption("지역·집단에 대한 약한 라벨입니다. 고독사 통계는 별도 배경으로만 제공하며 판정·학습에 사용하지 않습니다.")
    readiness_path = destination(config, "readiness.json")
    if readiness_path.exists():
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        a, b, c = st.columns(3)
        a.metric("분석 표본", f"{readiness['cohort_rows']:,}")
        b.metric("독립 시도" if readiness["geo_level"] == "sido" else "독립 시군구", readiness["independent_districts"])
        c.metric("학습 준비", "완료" if readiness["training_ready"] else "자료 보완 필요")
        for reason in readiness["blockers"]:
            st.warning(reason)
        if readiness.get("region_codes"):
            provinces = sorted({row["province"] for row in readiness.get("by_region", [])})
            st.caption("학습·평가 범위: " + ", ".join(provinces or readiness["region_codes"]))
        for warning in readiness.get("warnings", []):
            st.caption(warning)
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
                names = {r.cohort_id: f"{r.province} {r.district} {r.dong} / {r.sex} / {r.age_band} / {r.label_name}" for r in selected.itertuples()}
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


def render_anomaly_dashboard(config: Config):
    st.title("서울·강원 활동 위축 이상 탐색")
    st.caption("비지도 모델로 활동 패턴의 이상을 탐색합니다. 점수는 고립 확률이 아니며, 관측 소지역에 한정됩니다.")
    path = destination(config, "anomaly/detections.parquet")
    if not path.exists():
        st.info("먼저 python -m activity_agent detect --task isolation 으로 선택한 모델의 탐지 결과를 생성해 주세요.")
        return
    from activity_agent.isolation.anomaly import read_model
    try:
        model = read_model(config)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        return
    data = load(config, "anomaly/detections")
    if set(data.engine_version) != {model["model_version"]}:
        st.error("Anomaly predictions do not match the selected model; rerun detect --task isolation")
        return
    reference_label = "보정 표본" if model.get("score_reference") == "calibration" else "학습"
    st.caption(f"탐지 모델: {model.get('model_id', 'IF_BASELINE')} · 점수 기준: {reference_label} 분포")
    months = sorted(data.loc[data.month.gt(model["train_end"]), "month"].unique(), reverse=True)
    if not months:
        st.info("학습 종료 이후의 자료가 없습니다.")
        return
    a, b, c = st.columns(3)
    a.metric("학습 관측 행", f"{model['training_rows']:,}")
    b.metric("학습 소지역", f"{model['training_blocks']:,}" if model["training_blocks"] is not None else "미확인")
    c.metric("유효 학습 월", len(model["training_months"]))
    st.caption("성·연령별 반복 행은 독립 지역 수나 사람 수가 아닙니다. 한 달의 학습 분포를 비교하며 계절성을 학습한 모델은 아닙니다.")
    month = st.selectbox("활동 이상 분석 월", months, key="anomaly_month")
    province = st.selectbox("분석 지역", ["전체", *sorted(data.province.unique())], key="anomaly_region")
    sex = st.selectbox("분석 성별", ["전체", *sorted(data.sex.unique())], key="anomaly_sex")
    age = st.selectbox("분석 연령", ["전체", *sorted(data.age_band.unique())], key="anomaly_age")
    selected = data.loc[data.month.eq(month)].copy()
    for column, value in (("province", province), ("sex", sex), ("age_band", age)):
        if value != "전체":
            selected = selected.loc[selected[column].eq(value)]
    candidates = selected.loc[selected.candidate].sort_values("anomaly_raw", ascending=False)
    st.metric("위축 이상 후보 소지역", candidates.block_code.nunique())
    params = model["anomaly_settings"]
    st.caption(f"{reference_label} 이상 점수의 상위 {1-params['reference_quantile']:.0%} 기준을 고정 적용하고, 현재 유동 {params['minimum_decline']:.0%} 이상 감소 및 낮·주말 감소 근거가 있는 후보입니다. 실제 이상 비율을 가정하지 않습니다.")
    columns = ["province", "block_code", "sex", "age_band", "anomaly_percentile", "anomaly_raw", "activity_decline",
               "previous_activity_decline", "daytime_decline", "weekend_decline", "persistent_20pct",
               "baseline_flow", "province_structure_context"]
    st.dataframe(candidates[columns], hide_index=True)
    st.download_button("활동 이상 후보 CSV", candidates.to_csv(index=False).encode("utf-8-sig"), "activity_anomaly_candidates.csv")
    st.caption("시도 1인가구 구성비는 설명용 배경이며 이상 탐지 입력에 사용하지 않습니다.")
    with st.expander("활동 변화·후보 설명 질의"):
        question = st.text_input("활동 위축 이상 질문", placeholder="2025년 12월 강원 활동 이상 후보와 근거")
        if st.button("활동 이상 근거 조회"):
            from activity_agent.llm.vllm import create_isolation_agent
            st.markdown(create_isolation_agent(config, st.session_state.get("agent_backend")).answer(question))
    evaluation = destination(config, "anomaly/evaluation.json")
    if evaluation.exists():
        with st.expander("후보 분포·설정 안정성 점검"):
            st.json(json.loads(evaluation.read_text(encoding="utf-8")))
