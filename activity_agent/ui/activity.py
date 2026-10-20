from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from activity_agent.activity.queries import EvidenceAgent, QueryTools
from activity_agent.common.config import Config
from activity_agent.activity.labels import add_review, latest_reviews


def render_activity_dashboard(config: Config, config_path: str):
    st.title("지역 활동 위축 탐색 · 예방 근거")
    st.caption("지역·집단 집계 자료의 활동 위축 신호입니다. 개인의 고립 판정이나 고립 확률을 제공하지 않습니다.")
    if not (config.output / "detections.parquet").exists():
        st.info("먼저 python -m activity_agent run-all 을 실행해 주세요.")
        return

    @st.cache_resource
    def tools_for(version, config_path):
        return QueryTools(Config.load(config_path))

    stamp = (config.output / "detections.parquet").stat().st_mtime_ns
    tools = tools_for(stamp, str(config.root / config_path))
    data = tools.results
    st.sidebar.header("분석 조건")
    month = st.sidebar.selectbox("기준 월", sorted(data.month.unique(), reverse=True))
    signal_names = {"mobility": "소지역 성·연령 유동", "consumption": "시군구 소비", "time_pattern": "지역 전체 시간대",
                    "weekday_pattern": "지역 전체 요일", "district_mobility": "시군구 성·연령 유동"}
    signal = st.sidebar.selectbox("신호", [s for s in signal_names if s in set(data.signal)], format_func=signal_names.get)
    available = data.loc[data.signal.eq(signal)]
    region = st.sidebar.selectbox("지역", ["전체", *sorted(r for r in available.region.unique() if r)])
    sex = st.sidebar.selectbox("성별", ["필터 없음", *sorted(available.sex.unique())])
    age = st.sidebar.selectbox("연령대", ["필터 없음", *sorted(available.age.unique())])
    filters = dict(month=month, signal=signal, region=None if region == "전체" else region,
                   sex=None if sex == "필터 없음" else sex, age=None if age == "필터 없음" else age)
    selected = available.loc[available.month.eq(month)].copy()
    area_search = st.sidebar.text_input("소지역·지역 코드 검색")
    if area_search:
        selected = selected.loc[selected.area_id.str.contains(area_search, regex=False)]
    for col in ("region", "sex", "age"):
        if filters[col]:
            selected = selected.loc[selected[col].eq(filters[col])]
    a, b, c = st.columns(3)
    a.metric("선택 결과", f"{len(selected):,}")
    b.metric("점수 산출", f"{selected.score.notna().sum():,}")
    c.metric("판단 보류", f"{selected.score.isna().sum():,}")
    explore, evidence_tab, reviews_tab, chat_tab, quality_tab = st.tabs(["탐색", "근거·비교", "라벨 검토", "질의", "품질·실행 정보"])
    columns = ["feature_id", "area_id", "region", "sex", "age", "score", "quality", "engine", "score_kind"]
    with explore:
        st.dataframe(selected[columns].sort_values("score", ascending=False), hide_index=True)
        st.download_button("선택 결과 CSV", selected.to_csv(index=False).encode("utf-8-sig"), file_name=f"activity_{signal}_{month}.csv")
        if not (config.values["verified_crs"] and config.values["source_crs"]):
            st.info("정확한 원본 CRS가 아직 검증되지 않아 지리 지도를 보류했습니다. 좌표계를 추정해 위치를 표시하지 않습니다.")
        elif signal not in ("mobility", "time_pattern", "weekday_pattern"):
            st.info("이 신호는 시군구 단위입니다. 소지역 좌표 지도를 제공하지 않습니다.")
        else:
            from pyproj import Transformer
            import pydeck as pdk
            locations = selected.loc[selected.score.notna() & selected.x.notna() & selected.y.notna()].copy()
            if not locations.empty:
                locations["lon"], locations["lat"] = Transformer.from_crs(config.values["source_crs"], "EPSG:4326", always_xy=True).transform(locations.x, locations.y)
                if not (locations.lon.between(124, 132) & locations.lat.between(33, 39.5)).all():
                    st.error("좌표 변환 결과가 예상 한국 영역을 벗어납니다. 원본 CRS를 확인해 주세요.")
                else:
                    locations["color"] = [[int(255 * s / 100), 90, int(255 * (1 - s / 100)), 180] for s in locations.score]
                    layer = pdk.Layer("ScatterplotLayer", id="activity", data=locations[["feature_id", "area_id", "lon", "lat", "score", "color"]],
                                      get_position="[lon, lat]", get_fill_color="color", get_radius=100, pickable=True)
                    event = st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=locations.lat.mean(), longitude=locations.lon.mean(), zoom=9),
                                                    tooltip={"text": "{area_id}: {score}"}), on_select="rerun", key="activity_map")
                    objects = event.selection.get("objects", {}).get("activity", [])
                    if objects:
                        st.session_state["map_feature"] = objects[0]["feature_id"]
    if selected.empty:
        with evidence_tab:
            st.info("선택 조건에 결과가 없습니다.")
    else:
        detail_candidates = selected.sort_values("score", ascending=False).head(500)
        ids = detail_candidates.feature_id.tolist()
        preferred = st.session_state.get("map_feature")
        if preferred in set(selected.feature_id) and preferred not in ids:
            ids.insert(0, preferred)
        names = selected.set_index("feature_id").apply(lambda r: f"{r.area_id} / {r.sex} / {r.age}", axis=1).to_dict()
        st.caption("상세 선택은 점수 상위 500건입니다. 코드 검색·필터로 다른 대상을 확인할 수 있습니다.")
        chosen = st.selectbox("상세 근거 대상", ids, index=ids.index(preferred) if preferred in ids else 0,
                              format_func=names.get)
        row = tools.get_evidence(chosen)
        with evidence_tab:
            history = data.loc[data.signal.eq(row.signal) & data.area_id.eq(row.area_id) & data.sex.eq(row.sex) & data.age.eq(row.age)].sort_values("month")
            value_cols = [col for col in ["all_flow", "current_flow", "baseline_flow", "current_count", "baseline_count", "current_social", "baseline_social"] if col in history and history[col].notna().any()]
            if value_cols:
                st.plotly_chart(px.line(history, x="month", y=value_cols, markers=True), width="stretch")
            st.caption("all_flow는 전체 관측 합계, current/baseline은 해당 비교 창의 공통 관측 기준입니다. 서로 다른 창의 좌표 집합은 다를 수 있습니다.")
            st.dataframe(row.dropna().astype(str).rename("값"), width="stretch")
            st.subheader("같은 집단·유사 규모 비교")
            peers = tools.compare_peers(chosen)
            st.dataframe(peers[[col for col in columns + ["primary_decline", "baseline_flow"] if col in peers]], hide_index=True)
            if peers.empty:
                st.caption("검증된 지역 매핑 또는 유사 규모 비교 표본이 없습니다.")
            st.subheader("지역 전체 시간대·요일 맥락")
            st.dataframe(tools.get_local_patterns(chosen)[["signal", "daytime_decline", "weekend_decline", "quality"]], hide_index=True)
            st.subheader("시군구 소비 맥락")
            st.dataframe(tools.get_card_context(chosen)[["region", "sex", "age", "consumption_decline", "social_consumption_decline"]], hide_index=True)
        with reviews_tab:
            st.caption("검토 없음과 판단 보류는 정상 라벨이 아닙니다. 외부 확인 자료가 없는 화면 검토는 독립 평가 정답으로 사용하지 않습니다.")
            if row.signal in ("mobility", "district_mobility", "consumption"):
                with st.form("review"):
                    value = st.selectbox("검토 라벨", [-1, 0, 1], format_func=lambda v: {-1: "판단 보류", 0: "위축 근거 미약", 1: "위축 의심"}[v])
                    reviewer = st.text_input("검토자")
                    note = st.text_area("판단 근거·자료 출처")
                    independent = st.checkbox("규칙 판정과 독립된 외부 확인 자료에 근거함", value=False)
                    if st.form_submit_button("검토 이력 저장"):
                        try:
                            add_review(config, chosen, value, reviewer, note, independent)
                            st.success("기존 기록을 보존하고 새 검토를 저장했습니다.")
                        except ValueError as error:
                            st.error(str(error))
            else:
                st.info("시간대·요일 맥락에는 독립 학습 라벨을 부여하지 않습니다.")
            reviews = latest_reviews(config)
            st.dataframe(reviews.loc[reviews.feature_id.eq(chosen)], hide_index=True)
    with chat_tab:
        st.caption("공식 사이트 정책 후보는 policies-refresh로 수집한 자료에서 조회합니다. 대상 조건과 현재 모집 여부는 원문 확인이 필요합니다.")
        question = st.text_input("질문", placeholder="이번 달 50대 남성 활동 위축 후보와 예방 자원을 추천해줘")
        if st.button("근거 조회"):
            from activity_agent.llm.vllm import create_activity_agent
            with st.spinner("근거를 조회하고 답변을 작성하고 있습니다."):
                st.markdown(create_activity_agent(config, tools, st.session_state.get("agent_backend")).answer(question, **filters))
    with quality_tab:
        for name in ("quality.json", "validation_quality.json", "features_manifest.json", "run_manifest.json", "model_evaluation.json"):
            path = config.output / name
            if path.exists():
                with st.expander(name):
                    st.json(json.loads(path.read_text(encoding="utf-8")))
        sensitivity = config.output / "threshold_sensitivity.parquet"
        if sensitivity.exists():
            st.dataframe(pd.read_parquet(sensitivity), hide_index=True)

