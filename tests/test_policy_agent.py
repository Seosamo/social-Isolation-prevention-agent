"""Exercise model-directed tool selection without a running model or network."""
import json

import pytest

from activity_agent.policies.agent import decide, search_policies
from activity_agent.policies.search import question_context
from test_policy_search import NOW, URL, page


class Planner:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.states = []

    def chat(self, system, user, **kwargs):
        # Extraction/reranking have independent validation and must not consume actions.
        if 'available_actions' not in user:
            return '{}'
        self.states.append(json.loads(user))
        return json.dumps({'action': next(self.actions), 'reason': 'few_candidates'})


def test_model_observes_failure_then_changes_tool_and_preserves_filters(config):
    ctx = question_context(config, '강남구 50대 남성 고립 정책 추천')
    client = Planner(['local_services', 'stop'])
    queries, pages = [], []

    def provider(query, domains, timeout):
        queries.append(query)
        return [{'url': URL, 'title': '사회적 고립 지원 신청'}] if '복지관 주민센터' in query else []

    result = search_policies(config, ctx, search=provider,
        fetch=lambda url: pages.append(url) or page(), now=NOW, client=client)
    assert len(result['policies']) == 1
    assert [s['tool'] for s in result['agent']['steps']] == ['initial', 'local_services']
    assert client.states[0]['observation']['candidates'] == 0
    assert client.states[1]['observation']['candidates'] == 1
    assert result['context'] == ctx and ctx['age'] == '50' and ctx['sex'] == '남성'
    assert result['agent']['steps'][0]['decision']['method'] == 'vllm'
    assert pages == [URL]
    assert len(list((config.output / 'policies/lookups').glob('*.json'))) == 1


def test_repeated_documents_are_fetched_once_and_candidates_accumulate(config):
    ctx = question_context(config, '강남구 고립 정책 추천')
    def provider(query, domains, timeout):
        return [{'url': URL, 'title': '고립 지원 신청'}]
    fetched = []
    result = search_policies(config, ctx, search=provider,
        fetch=lambda url: fetched.append(url) or page(), now=NOW,
        client=Planner(['national_programs', 'local_services']))
    assert fetched == [URL]
    assert len(result['policies']) == 1
    assert len(result['agent']['steps']) == 3
    assert result['agent']['stop_reason'] == 'round_limit'
    assert result['agent']['tool_calls'] == {'search': 8, 'detail': 1}


@pytest.mark.parametrize('output', ['not json', '{"action":"shell","reason":"few_candidates"}',
    '{"action":"evidence_support","reason":"few_candidates"}',
    '{"action":"local_services","reason":"invented"}'])
def test_invalid_or_unavailable_action_falls_back(output):
    class Invalid:
        def chat(self, *args, **kwargs):
            return output
    state = {'candidates': 0, 'purpose_matched': 0}
    assert decide(Invalid(), state, ['local_services'], 1) == ('local_services', 'few_candidates', 'rule_fallback')


def test_stop_avoids_further_calls(config):
    result = search_policies(config, question_context(config, '강남구 정책 추천'),
        search=lambda *a: [], now=NOW, client=Planner(['stop']))
    assert result['agent']['tool_calls']['search'] == 4
    assert len(result['agent']['steps']) == 1


def test_missing_region_or_key_stops_without_wasting_rounds(config):
    def unexpected(*a):
        raise AssertionError('No network expected')
    result = search_policies(config, {'region': None}, search=unexpected, fetch=unexpected, now=NOW)
    assert result['status'] == 'region_required'
    assert result['agent']['stop_reason'] == 'region_required'
    assert result['agent']['tool_calls'] == {'search': 0, 'detail': 0}
    from activity_agent.policies.search import MissingSearchKey
    def missing(*a):
        raise MissingSearchKey()
    result = search_policies(config, question_context(config, '강원 정책 추천'),
                             search=missing, fetch=unexpected, now=NOW)
    assert result['agent']['stop_reason'] == 'missing_search_key'
    assert result['agent']['tool_calls']['search'] == 1


def test_research_never_accepts_another_city_or_age(config):
    result = search_policies(config, question_context(config, '강남구 50대 정책 추천'),
        search=lambda *a: [{'url': URL, 'title': '고립 지원 신청'}],
        fetch=lambda url: page('강남구 주민 만 65세 이상'), now=NOW,
        client=Planner(['local_services', 'national_programs']))
    assert result['policies'] == []
    assert any(e['reason'] == 'incompatible' for e in result['search_trace']['excluded'])


def test_total_deadline_stops_research(config):
    elapsed = [0.]
    def clock():
        return elapsed[0]
    def provider(*args):
        elapsed[0] += 70
        return []
    result = search_policies(config, question_context(config, '강원 정책 추천'),
                             search=provider, now=NOW, clock=clock)
    assert len(result['agent']['steps']) == 1
    assert result['agent']['stop_reason'] == 'time_limit'


def test_local_backend_uses_same_loop_with_rule_decisions(config):
    result = search_policies(config, question_context(config, '강원 정책 추천'),
                             search=lambda *a: [], now=NOW)
    assert [s['tool'] for s in result['agent']['steps']] == ['initial', 'local_services', 'national_programs']
    assert result['agent']['steps'][0]['decision']['method'] == 'rule'
