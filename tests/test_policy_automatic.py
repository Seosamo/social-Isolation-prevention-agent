"""Every analysis answer discovers policies, with an explicit national default."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from activity_agent.llm.vllm import VLLMBackend, VLLMError, VLLMSettings, create_isolation_agent
from activity_agent.policies import search
from activity_agent.policies.matching import enrich_context

NATIONAL_URL = 'https://www.mohw.go.kr/board/policy/common.do'


@pytest.fixture
def live_sources(config, monkeypatch):
    """Replace external I/O; keep search, validation, ranking, and history real."""
    queries, fetched = [], []

    def provider(self, query, domains, timeout):
        queries.append((query, domains))
        return [{'url': NATIONAL_URL, 'title': '전국 관계망 사업'}]

    def fetch(self, url):
        fetched.append(url)
        return ('<main><h1>전국 1인가구 사회적 고립 예방 사업</h1>'
                '<p>지원대상: 전국 국민 만 40~59세</p><p>신청기간: 상시</p>'
                '<p>신청방법: 주민센터 방문 신청</p>'
                '<p>사회적 관계망 형성과 지역 활동 참여를 위한 교류 모임을 지원합니다.</p></main>')

    monkeypatch.setattr(search.TavilyClient, 'search', provider)
    monkeypatch.setattr(search.MunicipalClient, 'fetch', fetch)
    config.values['policies']['rerank'] = False
    return queries, fetched


@pytest.fixture
def empty_isolation(config, monkeypatch):
    # The behavior under test is answer orchestration, independent of model scoring.
    import activity_agent.isolation.queries as queries
    monkeypatch.setattr(queries, 'AnomalyQueryTools', lambda cfg: SimpleNamespace(
        candidates=lambda *a, **kw: pd.DataFrame(), model={}, labels=pd.DataFrame(),
        candidate_evidence=lambda **kw: {}))
    return create_isolation_agent(config, backend='local')


def saved_lookup(config):
    files = list((config.output / 'policies/lookups').glob('*.json'))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding='utf-8'))


@pytest.mark.parametrize('task', ['activity', 'isolation'])
def test_unrequested_policies_appear_in_real_analysis_answers(
        complete_pipeline, empty_isolation, live_sources, task):
    from activity_agent.llm.vllm import create_activity_agent
    agent = (create_activity_agent(complete_pipeline, backend='local')
             if task == 'activity' else empty_isolation)
    answer = agent.answer('2025년 12월 50대 남성 활동 위축 근거 알려줘')
    assert '정책 조회' in answer and '전국 관계망 사업' in answer
    result = saved_lookup(complete_pipeline)
    assert result['context']['region'] == '전국'
    assert result['context']['age'] == '50' and result['context']['sex'] == '남성'
    assert result['policies'][0]['applies_to'] == '전국'
    assert '정책 검색 범위: 전국 공통' in answer
    queries, fetched = live_sources
    assert fetched == [NATIONAL_URL]
    assert queries and all(set(domains) == {'mohw.go.kr', 'bokjiro.go.kr', 'gov.kr'}
                           for _, domains in queries)


def test_explicit_region_keeps_local_and_national_discovery(
        config, empty_isolation, live_sources):
    answer = empty_isolation.answer('서울 50대 남성 활동 위축 근거 알려줘')
    assert '전국 관계망 사업' in answer
    result = saved_lookup(config)
    assert result['context']['region'] == '서울특별시'
    assert any('seoul.go.kr' in domains for _, domains in live_sources[0])
    assert any('mohw.go.kr' in domains for _, domains in live_sources[0])


@pytest.mark.parametrize('failure', [None, 'plan', 'summary'])
def test_vllm_without_policy_intent_searches_once_even_on_failure(
        config, empty_isolation, live_sources, failure):
    class Client:
        settings = VLLMSettings()

        def chat(self, system, user, **kwargs):
            if '허용 키' in system:
                if failure == 'plan':
                    raise VLLMError('plan offline')
                return json.dumps({'age': '50', 'sex': '남성'})
            if 'available_actions' in user:
                return json.dumps({'action': 'stop', 'reason': 'enough'})
            if 'eligibility' in system:
                return '{}'
            if failure == 'summary':
                raise VLLMError('summary offline')
            payload = json.loads(user)
            assert '전국 관계망 사업' in payload['evidence']
            return '조회 근거를 확인해 주세요.'

    empty_isolation.backend = VLLMBackend(config, Client())
    answer = empty_isolation.answer('50대 남성 활동 위축 근거 알려줘')
    assert '정책 조회' in answer and '전국 관계망 사업' in answer
    assert ('vLLM 처리 실패' in answer) == bool(failure)
    assert saved_lookup(config)['context']['region'] == '전국'
    assert live_sources[1] == [NATIONAL_URL]


def test_blank_question_does_not_search(empty_isolation, live_sources):
    answer = empty_isolation.answer('  ')
    assert '질문을 입력' in answer
    assert not live_sources[0] and not live_sources[1]


@pytest.mark.parametrize('question,region', [
    ('50대 남성 활동 위축 알려줘', None),
    ('점수 11 활동 위축 알려줘', None),
    ('강원 활동 위축 알려줘', '잘못된 지역'),
    ('부산 활동 위축 알려줘', None),
    ('서울 춘천시 활동 위축 알려줘', None),
    ('수원시 활동 위축 알려줘', None),
    ('청주시 활동 위축 알려줘', None),
])
def test_national_default_only_applies_when_region_is_absent(config, question, region):
    ctx = search.question_context(config, question, region=region)
    expected = '전국' if question in ('50대 남성 활동 위축 알려줘', '점수 11 활동 위축 알려줘') else None
    assert ctx['region'] == expected


@pytest.mark.parametrize('question', [
    '1인가구 고립 위험 알려줘', '고립가구 활동 위축 알려줘',
    '유동인구 감소 알려줘', '관계 형성 욕구 알려줘',
    '정책 예시 알려줘', '위험군 활동 위축 알려줘',
])
def test_common_domain_words_are_not_municipalities(config, question):
    assert search.question_context(config, question)['region'] == '전국'


@pytest.mark.parametrize('target,count', [
    ('전국 국민 만 40~59세', 1),
    ('서울시 주민 만 40~59세', 0),
    ('강원도 주민 만 40~59세', 0),
    ('춘천시 주민 만 40~59세', 0),
    ('부산시 주민 만 40~59세', 0),
    ('전국 국민 중 서울시 주민 만 40~59세', 0),
    ('수원시 관내 1인가구 누구나', 0),
    ('전국 국민 중 수원시 관내 1인가구', 0),
    ('청주시 관내 전국 국민', 0),
    ('1인가구 누구나', 0),
    ('전국 1인가구 누구나', 1),
    ('전국 여성 만 40~59세', 0),
    ('전국 국민 만 65세 이상', 0),
])
def test_nationwide_search_checks_scope_and_demographics(config, target, count):
    from test_policy_search import NOW, page
    ctx = search.question_context(config, '50대 남성 활동 위축 알려줘')
    result = search.lookup(config, ctx, now=NOW,
        search=lambda *a: [{'url': NATIONAL_URL, 'title': '사회적 고립 예방 사업'}],
        fetch=lambda url: page(target))
    assert len(result['policies']) == count


def test_nationwide_support_purpose_uses_valid_observations_without_localizing(config):
    ctx = search.question_context(config, '50대 남성 활동 위축 알려줘')
    enriched = enrich_context(config, ctx, [{
        'evidence_id': 'observed-decline', 'region': '서울특별시', 'age': '50',
        'sex': '남성', 'persistent_20pct': True}])
    assert enriched['region'] == '전국'
    assert {h['purpose_id'] for h in enriched['hypotheses']} == {'participation', 'checkin'}
    assert all(h['evidence_ids'] == ['observed-decline'] for h in enriched['hypotheses'])
