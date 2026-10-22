import io
import json

import pytest

from activity_agent.llm.vllm import VLLMBackend, VLLMClient, VLLMSettings


def response_opener(summary):
    responses = iter(["{}", summary])

    def respond(request, timeout):
        body = {"choices": [{"message": {"role": "assistant", "content": next(responses)}, "finish_reason": "stop"}]}
        return io.BytesIO(json.dumps(body, ensure_ascii=False).encode())

    return respond


@pytest.mark.parametrize("evidence,summary,new_numbers", [
    ("감소율 25.0%.", "감소율은 80%입니다.", ["80"]),
    ("감소율 25.0%.", "감소율은 25%입니다.", ["25"]),
    ("분석 월 202512.", "2025년 12월의 분석입니다.", ["2025", "12"]),
])
def test_numeric_mismatch_warns_and_preserves_generated_answer(config, evidence, summary, new_numbers):
    client = VLLMClient(VLLMSettings(), response_opener(summary))
    answer = VLLMBackend(config, client).run("활동 근거", lambda q, f: evidence, "202512", {}, "activity")
    assert "vLLM 생성 설명" in answer and "수치 검증 경고" in answer
    assert "vLLM 처리 실패" not in answer
    assert summary in answer and evidence in answer
    assert answer.index("수치 검증 경고") < answer.index(summary)
    for number in new_numbers:
        assert f"`{number}`" in answer
    # The source values are shown without claiming a one-to-one correction.
    assert "`25.0`" in answer if "25.0" in evidence else "`202512`" in answer


def test_matching_numbers_need_no_warning(config):
    summary = "감소율은 25.0%입니다."
    client = VLLMClient(VLLMSettings(), response_opener(summary))
    answer = VLLMBackend(config, client).run("활동", lambda q, f: "감소율 25.0%.", "202512", {}, "activity")
    assert summary in answer and "vLLM 생성 설명" in answer
    assert "수치 검증 경고" not in answer


@pytest.mark.parametrize("summary", ["https://invented.example", "감소율 80%. https://invented.example"])
def test_unknown_urls_still_reject_the_summary(config, summary):
    client = VLLMClient(VLLMSettings(), response_opener(summary))
    answer = VLLMBackend(config, client).run("활동", lambda q, f: "감소율 25.0%.", "202512", {}, "activity")
    assert "vLLM 처리 실패" in answer and "25.0%" in answer
    assert "invented.example" not in answer and "80%" not in answer


def test_cli_shows_numeric_warning_without_debug_flag(complete_pipeline, monkeypatch, capsys):
    from activity_agent.cli import main
    import activity_agent.llm.vllm as vllm
    monkeypatch.setattr(vllm, "urlopen", response_opener("생성된 설명: 999999.9%"))
    assert main(["--config", str(complete_pipeline.root / "config/default.json"),
                 "ask", "--backend", "vllm", "활동 근거"]) == 0
    output = capsys.readouterr()
    assert "수치 검증 경고" in output.out
    assert "생성된 설명: 999999.9%" in output.out
    assert "도구 조회 근거 원문" in output.out and output.err == ""


def test_dashboard_shows_numeric_warning_and_answer_without_diagnostic_controls(complete_pipeline, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import activity_agent.llm.vllm as vllm
    monkeypatch.setattr(vllm, "urlopen", response_opener("생성된 설명: 999999.9%"))
    path = str(complete_pipeline.root / "config/default.json")
    app = AppTest.from_string(f"from app import main\nmain({path!r})").run(timeout=30)
    app.selectbox(key="agent_backend").set_value("vllm").run(timeout=30)
    next(w for w in app.text_input if w.label == "질문").set_value("활동 근거")
    next(w for w in app.button if w.label == "근거 조회").click().run(timeout=30)
    assert not app.exception
    assert any("수치 검증 경고" in w.value and "생성된 설명: 999999.9%" in w.value for w in app.markdown)
    assert not any(w.label == "vLLM 진단 정보 표시" for w in app.checkbox)
