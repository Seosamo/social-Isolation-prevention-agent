"""The editable response document reaches the actual chat-completions request."""
import io
import json
from types import SimpleNamespace

import pytest

from activity_agent.llm import vllm


def client_with_requests():
    requests = []

    def opener(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        content = '{}' if '허용 키' in payload['messages'][0]['content'] else '조회 근거를 확인해 주세요.'
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': content},
            'finish_reason': 'stop'}]}, ensure_ascii=False).encode())

    return vllm.VLLMClient(vllm.VLLMSettings(), opener), requests


def resource_root(monkeypatch, tmp_path):
    directory = tmp_path / 'prompts'
    directory.mkdir()
    monkeypatch.setattr(vllm, 'resources', SimpleNamespace(files=lambda package: tmp_path), raising=False)
    return directory / 'activity_response.md'


@pytest.mark.parametrize('task', ['activity', 'isolation'])
def test_response_document_is_sent_only_to_final_generation(config, task, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client, requests = client_with_requests()
    answer = vllm.VLLMBackend(config, client).run(
        '활동 근거 알려줘', lambda q, f: '조회된 지역·집단 활동 근거', '202512', {}, task)
    assert 'vLLM 생성 설명' in answer
    assert len(requests) == 2
    plan_system = requests[0]['messages'][0]['content']
    answer_system = requests[1]['messages'][0]['content']
    assert 'TYPE A. 현황 및 후보 조회' not in plan_system
    assert 'TYPE A. 현황 및 후보 조회' in answer_system
    assert '### 주요 탐지 결과' in answer_system
    assert '### 관련 정책 및 후속 확인' in answer_system
    assert '새 수치·정책·URL을 만들지 않는다' in answer_system


def test_prompt_edits_apply_to_next_request_without_recreating_backend(config, tmp_path, monkeypatch):
    path = resource_root(monkeypatch, tmp_path)
    path.write_text('첫 번째 편집 지침', encoding='utf-8')
    client, requests = client_with_requests()
    backend = vllm.VLLMBackend(config, client)
    backend.run('활동', lambda q, f: '조회 근거', '202512', {}, 'activity')
    path.write_text('수정한 편집 지침', encoding='utf-8')
    backend.run('활동', lambda q, f: '조회 근거', '202512', {}, 'activity')
    assert '첫 번째 편집 지침' in requests[1]['messages'][0]['content']
    assert '수정한 편집 지침' in requests[3]['messages'][0]['content']
    assert '첫 번째 편집 지침' not in requests[3]['messages'][0]['content']


@pytest.mark.parametrize('content', [None, b' \n\t', b'\xff'])
def test_invalid_prompt_preserves_failure_diagnostics_and_local_evidence(
        config, tmp_path, monkeypatch, content):
    path = resource_root(monkeypatch, tmp_path)
    if content is not None:
        path.write_bytes(content)
    client, requests = client_with_requests()
    answer = vllm.VLLMBackend(config, client).run(
        '활동', lambda q, f: '로컬 조회 근거', '202512', {}, 'activity')
    assert 'vLLM 처리 실패' in answer
    assert '실패 단계: 설명 생성' in answer
    assert '응답 프롬프트' in answer
    assert answer.endswith('로컬 조회 근거')
    assert len(requests) == 1
