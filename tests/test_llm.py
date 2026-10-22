import io
import json
from urllib.error import HTTPError, URLError

import pytest

from activity_agent.common.config import Config
from activity_agent.llm.vllm import VLLMBackend, VLLMClient, VLLMError, VLLMSettings, backend_mode, parse_plan


def test_transport_uses_exact_model_and_chat_endpoint():
    captured = []
    def open_request(request, timeout):
        captured.append((request, timeout))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "응답"}, "finish_reason": "stop"}]}).encode())
    client = VLLMClient(VLLMSettings(api_key="test-secret"), open_request)
    assert client.chat("system", "question") == "응답"
    request, timeout = captured[0]
    assert request.full_url == "http://localhost:8000/v1/chat/completions"
    assert json.loads(request.data)["model"] == "skt/A.X-4.0"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert timeout == 60
    assert "test-secret" not in repr(client.settings)


@pytest.mark.parametrize("error", [URLError("secret"), TimeoutError("secret"), HTTPError("secret", 401, "secret", {}, None)])
def test_transport_errors_do_not_expose_credentials(error):
    def fail(*args, **kwargs):
        raise error
    with pytest.raises(VLLMError) as result:
        VLLMClient(VLLMSettings(), fail).chat("s", "q")
    assert "secret" not in str(result.value)


def test_model_check_and_invalid_response():
    client = VLLMClient(VLLMSettings(), lambda *a, **kw: io.BytesIO(b'{"data":[{"id":"skt/A.X-4.0"}]}'))
    assert client.check()["connected"]
    client = VLLMClient(VLLMSettings(), lambda *a, **kw: io.BytesIO(b'{"choices":[]}'))
    with pytest.raises(VLLMError):
        client.chat("s", "q")


@pytest.mark.parametrize("text", ['{"age":"70"}', '{"month":"202513"}', '{"command":"delete"}', '[]', '{"sex":[]}', 'bad'])
def test_invalid_plans_are_rejected(text):
    with pytest.raises(VLLMError):
        parse_plan(text)


class FakeClient:
    settings = VLLMSettings()
    def __init__(self, plan, summary="관측된 활동 변화를 확인하고 현장 확인을 진행해 주세요."):
        self.responses = [json.dumps(plan, ensure_ascii=False), summary]
        self.messages = []
    def chat(self, system, user, **kwargs):
        self.messages.append(json.loads(user))
        return self.responses.pop(0)


def test_plan_constraints_explicit_filters_and_evidence(config):
    client = FakeClient({"month": "202512", "sex": "여성", "age": "20", "region": "강원 춘천시"})
    calls = []
    def lookup(question, filters):
        calls.append(filters)
        return "202509 근거: 감소율 25.0%. 고립 확률이 아닙니다."
    answer = VLLMBackend(config, client).run("2025년 9월 강남구 50대 남성", lookup, "202512", {"sex": "여성"}, "activity")
    assert calls[0] == {"month": "202509", "sex": "여성", "age": "50", "region": "서울 강남구", "_resolved": True}
    assert "vLLM 생성 설명" in answer and "도구 조회 근거 원문" in answer
    assert client.messages[1]["evidence"] in answer


def test_unsubstantiated_summary_and_plan_failure_fall_back(config):
    lookup = lambda question, filters: "감소율 25.0%. 고립 확률이 아닙니다."
    client = FakeClient({}, "고립 확률 99.9%. https://invented.example")
    answer = VLLMBackend(config, client).run("활동", lookup, "202512", {}, "activity")
    assert "vLLM 처리 실패" in answer and "25.0%" in answer
    assert "99.9" not in answer and "invented.example" not in answer
    client = FakeClient({"age": "70"})
    assert "로컬 근거" in VLLMBackend(config, client).run("활동", lookup, "202512", {}, "activity")


def test_activity_agent_integration(complete_pipeline):
    from activity_agent.activity.queries import EvidenceAgent, QueryTools
    client = FakeClient({"month": "202512", "sex": "남성", "age": "50", "signal": "mobility", "policies": True})
    result = EvidenceAgent(QueryTools(complete_pipeline), VLLMBackend(complete_pipeline, client)).answer("최신 남성 활동과 지원")
    assert "도구 조회 근거 원문" in result and "근거 ID" in result and "고립 확률이 아닙니다" in result
    assert len(client.messages) == 2


def test_settings_environment_and_analysis_version(config, monkeypatch):
    monkeypatch.delenv("AGENT_BACKEND", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://example:8000/v1/")
    monkeypatch.setenv("VLLM_MODEL", "served-model")
    assert backend_mode(config) == "vllm" and backend_mode(config, "local") == "local"
    assert VLLMSettings.load(config).base_url == "http://example:8000/v1"
    assert VLLMSettings.load(config).model == "served-model"
    values = dict(config.values, agent={"backend": "vllm"})
    assert Config(config.root, values).version == config.version
