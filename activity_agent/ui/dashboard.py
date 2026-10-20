"""Dashboard entrypoint; task views are loaded only when selected."""
from __future__ import annotations

import argparse

import streamlit as st

from activity_agent.common.config import Config


def main(config_path: str | None = None):
    if config_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="config/default.json")
        args, _ = parser.parse_known_args()
        config_path = args.config
    config = Config.load(config_path)
    st.set_page_config(page_title="지역 활동 위축 탐색", layout="wide")
    from activity_agent.llm.vllm import backend_mode
    modes = ["local", "vllm"]
    st.sidebar.selectbox("Agent 응답 방식", modes, index=modes.index(backend_mode(config)),
                         format_func=lambda x: "vLLM · LLM 해석 및 설명" if x == "vllm" else "로컬 · 근거 조회", key="agent_backend")
    task = st.sidebar.radio("분석 과제", ["활동 위축", "고립 의심"], key="analysis_task")
    if task == "고립 의심":
        from activity_agent.ui.isolation import render_isolation_dashboard
        render_isolation_dashboard(config)
    else:
        from activity_agent.ui.activity import render_activity_dashboard
        render_activity_dashboard(config, config_path)
