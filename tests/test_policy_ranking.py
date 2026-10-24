"""Policy identity, auditable relevance, and fail-closed optional reranking."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from activity_agent.policies import ranking, search
from activity_agent.policies.evidence import select_chunks
from activity_agent.policies.matching import enrich_context, observe_candidates, purpose_matches


def observed(persistent=False, **overrides):
    return {'month': '202512', 'province': '서울특별시', 'block_code': 'b1', 'sex': '남성',
        'age_band': '50', 'activity_decline': .3, 'candidate': True, 'quality_eligible': True,
        'persistent_20pct': persistent, **overrides}


def detection_context(config, persistent=False):
    with search.policy_session():
        observe_candidates(pd.DataFrame([observed(persistent)]), 'anomaly')
        return enrich_context(config, search.question_context(config, '서울 50대 남성 지원'),
                              search.SESSION.get()['observations'])


def item(name, support, context, *, status='접수 중', year='2026', round='1', agency='서울시', region='서울특별시'):
    text = f'사업명: {name}\n담당기관: {agency}\n사업연도: {year}\n모집차수: {round}차\n지원대상: 서울시 주민\n신청기간: 2026.10.01 ~ 2026.10.31\n신청방법: 주민센터\n지원내용: {support}'
    fields = search.extract_fields(text)
    chunks = select_chunks(text, ['안부', '관계망'])
    for c in chunks:
        c['id'] = name + ':' + c['id']
    return {'title': name + ' 모집 공고', 'url': 'https://www.seoul.go.kr/' + name,
        'applies_to': region, 'region': region, 'recruitment': status, 'identity': ranking.metadata(text, name),
        'evidence': fields, 'evidence_chunks': chunks, 'purpose_matches': purpose_matches(text, context),
        'unknown': [], 'eligibility': fields['eligibility'], 'period': fields['period'],
        'application': fields['application'], 'content_hash': 'hash', 'checked_at': '2026-10-24T00:00:00+09:00',
        'search_rank': 1, 'search_hits': []}


def test_rrf_keeps_original_rank_and_allocates_query_opportunities():
    lists = [('q1', [{'url': f'https://www.seoul.go.kr/{i}', 'provider_rank': i + 1} for i in range(5)]),
             ('q2', [{'url': 'https://www.seoul.go.kr/special'}]),
             ('q3', [{'url': 'https://www.seoul.go.kr/4', 'provider_rank': 3}])]
    fused = ranking.fuse_results(lists)
    shared = next(row for row in fused if row['url'].endswith('/4'))
    assert shared['search_score'] == pytest.approx(1 / 65 + 1 / 63)
    chosen = ranking.select_documents(fused, ['q1', 'q2', 'q3'], 4)
    assert len(chosen) == 4
    assert any(row['url'].endswith('/special') for row in chosen)


def test_document_identity_preserves_ids_year_and_unknown_query_parameters():
    base = 'https://www.chuncheon.go.kr/view?nttId='
    assert ranking.document_key(base + '1') != ranking.document_key(base + '2')
    assert ranking.document_key(base + '1&year=2025') != ranking.document_key(base + '1&year=2026')
    assert ranking.document_key(base + '1&mid=a') == ranking.document_key(base + '1&mid=b')
    path = 'https://www.gangnam.go.kr/board/policy/1/view.do'
    assert ranking.document_key(path + '?bad=1') != ranking.document_key(path)


def test_same_policy_groups_sources_but_not_unknown_year_round_or_region(config):
    context = search.question_context(config, '서울 관계망 지원')
    a = item('관계망 회복 사업', '관계망 교류 모임', context)
    b = deepcopy(a)
    b.update(url='https://www.seoul.go.kr/press', title='관계망 회복 사업 보도자료')
    grouped = ranking.group_policies([b, a])
    assert len(grouped) == 1 and grouped[0]['url'] == a['url']
    assert len(grouped[0]['related_sources']) == 2
    for field, value in [('year', '2025'), ('round', '2'), ('agency', '다른기관'), ('round', '')]:
        changed = deepcopy(b)
        changed['identity'][field] = value
        assert len(ranking.group_policies([a, changed])) == 2
    b['applies_to'] = '강원특별자치도'
    assert len(ranking.group_policies([a, b])) == 2


def test_condition_conflict_and_explicit_correction(config):
    context = search.question_context(config, '서울 관계망 지원')
    a = item('관계망사업', '관계망 교류', context)
    b = deepcopy(a)
    b.update(url='https://www.seoul.go.kr/correction', title='관계망사업 정정 공고',
             period='신청기간: 2026.10.02 ~ 2026.10.31')
    b['evidence']['period'] = b['period']
    grouped = ranking.group_policies([a, b])
    assert grouped[0]['recruitment'] == '확인 필요' and grouped[0]['condition_conflict']
    b['corrects_url'] = a['url']
    corrected = ranking.group_policies([a, b])[0]
    assert corrected['url'] == b['url'] and corrected['recruitment'] == '접수 중'


def test_conflicting_or_corrected_conditions_cannot_reuse_old_group_linkage(config):
    context = detection_context(config, True)
    a = item('사업', '안부 확인', context)
    b = deepcopy(a)
    b.update(url='https://www.seoul.go.kr/new', title='사업 정정 공고', eligibility='지원대상: 서울시 여성')
    b['evidence']['eligibility'] = b['eligibility']
    b['purpose_matches'] = []
    conflict = ranking.group_policies([a, b])[0]
    assert not any(m['evidence_ids'] for m in conflict['purpose_matches'])
    b['corrects_url'] = a['url']
    corrected = ranking.group_policies([a, b])[0]
    assert not corrected['purpose_matches']


def test_full_body_conditions_after_24000_chars_are_not_lost():
    text = '1인가구 지원 안내\n' + ('배경 설명입니다.\n' * 3000) + '\n지원대상: 여성 만 20~29세\n신청기간: 2025.01.01 ~ 2025.01.31\n신청방법: 방문'
    body = search.extract_text('<main>' + text.replace('\n', '<br>') + '</main>')
    fields = search.extract_fields(body)
    assert '여성' in fields['eligibility']
    assert search.recruitment(fields['period'], search.date(2026, 10, 24)) == '마감'
    chunks = select_chunks(body, ['지원'], limit=4)
    assert any('여성 만 20~29세' in c['text'] for c in chunks)
    for c in chunks:
        assert c['text'] == body[c['start']:c['end']]


@pytest.mark.parametrize('override', [
    {'candidate': False}, {'quality_eligible': False}, {'activity_decline': float('nan')},
    {'activity_decline': -.1},
])
def test_unusable_anomaly_does_not_create_hypothesis(config, override):
    with search.policy_session():
        observe_candidates(pd.DataFrame([observed(**override)]), 'anomaly')
        ctx = enrich_context(config, search.question_context(config, '서울 지원'), search.SESSION.get()['observations'])
    assert not ctx.get('hypotheses')


def test_persistence_changes_ranking_without_using_anomaly_score(config):
    normal = detection_context(config)
    persistent = detection_context(config, True)
    docs = lambda ctx: [item('가-관계망', '관계망 교류 모임', ctx), item('나-안부', '안부 확인 정기 방문', ctx)]
    normal_rank = ranking.rank_policies(ranking.group_policies(docs(normal)), normal)
    persistent_rank = ranking.rank_policies(ranking.group_policies(docs(persistent)), persistent)
    assert normal_rank[0]['title'].startswith('가-관계망')
    assert persistent_rank[0]['title'].startswith('나-안부')
    assert all(h['status'].endswith('정책 탐색 가설') for h in persistent['hypotheses'])
    assert 'anomaly_percentile' not in json.dumps(persistent)


def test_recruitment_tiers_cannot_be_overridden_by_purpose_score(config):
    context = detection_context(config, True)
    current = item('현재', '일반 정보', context)
    future = item('예정', '안부 관계망 교류', context, status='접수 예정')
    unknown = item('미확인', '안부 관계망 교류', context, status='확인 필요')
    rows = ranking.rank_policies(ranking.group_policies([unknown, future, current]), context)
    assert [r['recruitment'] for r in rows] == ['접수 중', '접수 예정', '확인 필요']


def test_direct_user_purpose_precedes_broader_related_support(config):
    context = search.question_context(config, '서울 50대 남성 사회적 고립 예방 지원')
    rows = ranking.rank_policies(ranking.group_policies([
        item('안부사업', '스마트 안부 전화', context),
        item('관계지원', '사회적 고립 위험군 관계망 형성', context)]), context)
    assert rows[0]['title'].startswith('관계지원')


def test_province_observation_cannot_be_assigned_to_unmapped_district(config):
    with search.policy_session():
        observe_candidates(pd.DataFrame([observed(True)]), 'anomaly')
        ctx = enrich_context(config, search.question_context(config, '강남구 50대 남성 지원'),
                             search.SESSION.get()['observations'])
    assert not ctx.get('hypotheses')


def test_regional_activity_only_and_invalid_group_do_not_create_hypothesis(config):
    frame = pd.DataFrame([{'signal': 'time_pattern', 'quality': 'ok', 'activity_decline': .3,
                           'score': 30., 'sex': '전체', 'age': '전체', 'feature_id': 'local'}])
    with search.policy_session():
        observe_candidates(frame, 'activity')
        assert search.SESSION.get()['observations'] == []


def test_real_agent_supplies_structured_observations_to_policy_search(config, monkeypatch):
    from activity_agent.isolation.queries import IsolationEvidenceAgent
    from activity_agent.policies import recommendation
    seen = []
    monkeypatch.setattr(recommendation, 'lookup', lambda cfg, ctx, **kw: seen.append(ctx) or {
        'status': 'no_results', 'checked_at': '2026-10-24', 'policies': [], 'historical': []})
    row = observed(True, anomaly_percentile=98., previous_activity_decline=.2, daytime_decline=.1,
        weekend_decline=.1, baseline_flow=100., province_structure_context=None)
    agent = IsolationEvidenceAgent.__new__(IsolationEvidenceAgent)
    agent.config, agent.backend, agent.is_anomaly = config, None, True
    agent.tools = SimpleNamespace(candidates=lambda *a, **kw: pd.DataFrame([row]), model={})
    answer = agent.answer('서울 50대 남성 지원 정책 추천')
    assert '정책 조회' in answer
    assert seen[0]['hypotheses'][1]['purpose_id'] == 'checkin'
    assert seen[0]['observations'][0]['evidence_id'] == '202512:b1:남성:50'


def test_search_queries_do_not_leak_observation_ids_or_scores(config):
    context = detection_context(config, True)
    queries = []
    result = search.lookup(config, context, search=lambda query, *args: queries.append(query) or [])
    assert result['status'] == 'no_results'
    assert any('안부' in query for query in queries)
    assert all('b1' not in query and '202512' not in query and '0.3' not in query for query in queries)


class RerankClient:
    def __init__(self, transform=None): self.transform, self.calls = transform, []
    def chat(self, prompt, raw, **kwargs):
        self.calls.append(json.loads(raw))
        policies = self.calls[-1]['policies']
        items = []
        for policy in policies:
            chunk = policy['chunks'][0]
            items.append({'policy_id': policy['policy_id'], 'grade': 3, 'purpose_id': 'participation',
                'chunk_id': chunk['id'], 'quote': '관계망 교류 모임'})
        if self.transform: self.transform(items)
        return json.dumps({'items': items})


def test_optional_reranker_is_bounded_and_retains_rule_audit(config):
    context = detection_context(config)
    rows = ranking.rank_policies(ranking.group_policies([item('사업', '관계망 교류 모임', context)]), context)
    client = RerankClient()
    result, report = ranking.rerank(rows, context, client, 5)
    assert report['status'] == 'ok' and len(client.calls) == 1
    assert result[0]['ranking']['method'] == 'vllm'
    assert result[0]['ranking']['rule_rank'] == 1


@pytest.mark.parametrize('transform', [
    lambda rows: rows[0].update(policy_id='invented'),
    lambda rows: rows[0].update(quote='존재하지 않는 인용'),
    lambda rows: rows[0].update(grade=True),
    lambda rows: rows[0].update(purpose_id='diagnosis'),
    lambda rows: rows.append(rows[0]),
    lambda rows: rows[0].update(chunk_id='invented'),
])
def test_invalid_rerank_keeps_all_rule_scores(config, transform):
    context = detection_context(config)
    rows = ranking.rank_policies(ranking.group_policies([item('사업', '관계망 교류 모임', context)]), context)
    before = deepcopy(rows)
    result, report = ranking.rerank(rows, context, RerankClient(transform), 5)
    assert report['status'] == 'fallback' and result == before


def test_reranker_deadline_skips_call(config):
    context = detection_context(config)
    rows = ranking.rank_policies(ranking.group_policies([item('사업', '관계망 교류 모임', context)]), context)
    client = RerankClient()
    assert ranking.rerank(rows, context, client, 0)[1]['status'] == 'not_run'
    assert not client.calls
