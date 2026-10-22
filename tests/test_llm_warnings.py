"""Rejected LLM output stays inspectable as diagnostics in CLI and dashboard."""
import io
import json
import re
from urllib.error import HTTPError

import pytest

from activity_agent.llm.vllm import VLLMBackend, VLLMClient, VLLMSettings


def response_opener(summary):
    responses = iter(['{}', summary])

    def respond(request, timeout):
        body = {'choices': [{'message': {'role': 'assistant', 'content': next(responses)}, 'finish_reason': 'stop'}]}
        return io.BytesIO(json.dumps(body, ensure_ascii=False).encode())

    return respond


def diagnostics(answer):
    match = re.search(r'(`{3,})text\n(.*?)\n\1', answer, flags=re.S)
    assert match, answer
    return match.group(2)


@pytest.mark.parametrize('evidence,summary,new_numbers', [
    ('감소율 25.0%', '감소율은 80%입니다.', ['80']),
    ('감소율 25.0%', '감소율은 25%입니다.', ['25']),
    ('분석 월 202512.', '2025년 12월의 분석입니다.', ['12', '2025']),
])
def test_numeric_mismatch_reports_exact_diagnostics(config, evidence, summary, new_numbers):
    client = VLLMClient(VLLMSettings(), response_opener(summary))
    answer = VLLMBackend(config, client).run('활동 근거', lambda q, f: evidence, '202512', {}, 'activity')
    block = diagnostics(answer)
    assert 'vLLM 처리 실패' in answer
    assert '실패 단계: 근거 검증' in block
    assert f'조회 근거: {evidence}' in block
    assert f'생성 답변: {summary}' in block
    assert '추가로 감지된 숫자: ' + json.dumps(new_numbers, ensure_ascii=False) in block
    assert '추가로 감지된 URL: []' in block
    assert '도구 조회 근거 원문' in answer
    assert answer.endswith(evidence)


def test_matching_numbers_need_no_failure_diagnostics(config):
    summary = '감소율은 25.0%입니다.'
    client = VLLMClient(VLLMSettings(), response_opener(summary))
    answer = VLLMBackend(config, client).run('활동', lambda q, f: '감소율 25.0%.', '202512', {}, 'activity')
    assert summary in answer and 'vLLM 생성 설명' in answer
    assert '실패 단계:' not in answer


@pytest.mark.parametrize('summary', ['https://invented.example', '감소율 80%. https://invented.example'])
def test_unknown_urls_appear_only_in_failed_response_diagnostics(config, summary):
    client = VLLMClient(VLLMSettings(), response_opener(summary))
    answer = VLLMBackend(config, client).run('활동', lambda q, f: '감소율 25.0%.', '202512', {}, 'activity')
    block = diagnostics(answer)
    assert '생성 답변: ' + summary in block
    assert '추가로 감지된 URL: ["https://invented.example"]' in block
    assert 'vLLM 처리 실패' in answer
    assert 'invented.example' not in answer.replace(block, '')


def test_invalid_plan_preserves_raw_plan_and_failure_stage(config):
    def respond(request, timeout):
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'JSON이 아닌 해석 원문'}, 'finish_reason': 'stop'}]}, ensure_ascii=False).encode())
    client = VLLMClient(VLLMSettings(), respond)
    answer = VLLMBackend(config, client).run('활동', lambda q, f: '로컬 근거', '202512', {}, 'activity')
    block = diagnostics(answer)
    assert '실패 단계: 질문 해석' in block
    assert '질문 해석 응답: JSON이 아닌 해석 원문' in block
    assert '생성 답변: 수신한 생성 원문 없음' in block
    assert answer.endswith('로컬 근거')


def test_truncated_explanation_preserves_partial_content(config):
    responses = iter([('{}', 'stop'), ('잘린 생성 설명 원문', 'length')])
    def respond(request, timeout):
        content, reason = next(responses)
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': content}, 'finish_reason': reason}]}, ensure_ascii=False).encode())
    client = VLLMClient(VLLMSettings(), respond)
    answer = VLLMBackend(config, client).run('활동', lambda q, f: '로컬 근거', '202512', {}, 'activity')
    block = diagnostics(answer)
    assert '실패 단계: 설명 생성' in block
    assert '생성 답변: 잘린 생성 설명 원문' in block
    assert '응답 종료 사유: length' in block


def test_summary_timeout_preserves_plan_and_reports_no_generated_response(config):
    responses = iter(['{}', None])
    def respond(request, timeout):
        content = next(responses)
        if content is None:
            raise TimeoutError('private transport detail')
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}).encode())
    answer = VLLMBackend(config, VLLMClient(VLLMSettings(), respond)).run('활동', lambda q, f: '로컬 근거', '202512', {}, 'activity')
    block = diagnostics(answer)
    assert '실패 단계: 설명 생성' in block
    assert '질문 해석 응답: {}' in block
    assert '생성 답변: 수신한 생성 원문 없음' in block
    assert 'private transport detail' not in answer


def test_failed_response_cannot_escape_code_block(config):
    summary = '감소율 80%\n```\n[가짜 링크](https://invented.example)\n```'
    answer = VLLMBackend(config, VLLMClient(VLLMSettings(), response_opener(summary))).run('활동', lambda q, f: '감소율 25.0%', '202512', {}, 'activity')
    assert summary in diagnostics(answer)
    assert 'invented.example' not in answer.replace(diagnostics(answer), '')


def test_http_error_response_redacts_configured_credentials(config):
    def respond(request, timeout):
        raise HTTPError(request.full_url, 401, 'Unauthorized', {}, io.BytesIO(b'error: test-api-secret'))
    client = VLLMClient(VLLMSettings(api_key='test-api-secret'), respond)
    answer = VLLMBackend(config, client).run('활동', lambda q, f: '로컬 근거', '202512', {}, 'activity')
    assert 'test-api-secret' not in answer
    assert 'error: [REDACTED]' in diagnostics(answer)


def test_cli_shows_failure_diagnostics_without_debug_flag(complete_pipeline, monkeypatch, capsys):
    from activity_agent.cli import main
    import activity_agent.llm.vllm as vllm
    monkeypatch.setattr(vllm, 'urlopen', response_opener('생성된 설명: 999999.9%'))
    assert main(['--config', str(complete_pipeline.root / 'config/default.json'),
                 'ask', '--backend', 'vllm', '활동 근거']) == 0
    output = capsys.readouterr()
    block = diagnostics(output.out)
    assert '실패 단계: 근거 검증' in block
    assert '생성 답변: 생성된 설명: 999999.9%' in block
    assert '추가로 감지된 숫자: ["999999.9"]' in block
    assert '도구 조회 근거 원문' in output.out and output.err == ''


def test_dashboard_shows_failure_diagnostics_without_diagnostic_controls(complete_pipeline, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import activity_agent.llm.vllm as vllm
    monkeypatch.setattr(vllm, 'urlopen', response_opener('생성된 설명: 999999.9%'))
    path = str(complete_pipeline.root / 'config/default.json')
    app = AppTest.from_string(f'from app import main\nmain({path!r})').run(timeout=30)
    app.selectbox(key='agent_backend').set_value('vllm').run(timeout=30)
    next(w for w in app.text_input if w.label == '질문').set_value('활동 근거')
    next(w for w in app.button if w.label == '근거 조회').click().run(timeout=30)
    assert not app.exception
    answers = [w.value for w in app.markdown if '실패 단계: 근거 검증' in w.value]
    assert len(answers) == 1
    assert '생성 답변: 생성된 설명: 999999.9%' in diagnostics(answers[0])
    assert not any(w.label == 'vLLM 진단 정보 표시' for w in app.checkbox)


@pytest.mark.parametrize('summary', [
    '\x1b[2J\x1b[H정상 답변: 80%',
    '\x1b]8;;https://invented.example\x07가짜 링크\x1b]8;;\x07 80%',
])
def test_failed_response_displays_terminal_controls_literally(config, summary):
    answer = VLLMBackend(config, VLLMClient(VLLMSettings(), response_opener(summary))).run(
        '활동', lambda q, f: '감소율 25.0%', '202512', {}, 'activity')
    assert '\x1b' not in answer and '\x07' not in answer
    block = diagnostics(answer)
    assert '\\x1b' in block
    if '\x07' in summary:
        assert '\\x07' in block
    assert '실패 단계: 근거 검증' in block
