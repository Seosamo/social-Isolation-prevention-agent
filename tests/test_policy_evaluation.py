"""Fixed scenario evaluation against the prior source/keyword ordering."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from activity_agent.policies import search
from activity_agent.policies.matching import observe_candidates, enrich_context

CASES = json.loads((Path(__file__).parent / 'fixtures/policy_ranking_cases.json').read_text(encoding='utf-8'))


def test_twelve_policy_scenarios_meet_gates_and_match_or_improve_baseline(config, monkeypatch):
    assert len(CASES) >= 12
    baseline_hits = new_hits = expected_total = 0
    baseline_first = new_first = first_total = 0
    reports = []
    original_group = search.group_policies
    accepted = []
    def capture(rows):
        accepted[:] = deepcopy(rows)
        return original_group(rows)
    monkeypatch.setattr(search, 'group_policies', capture)
    for case in CASES:
        ctx = search.question_context(config, case['question'])
        if case.get('detection'):
            row = {'month': '202512', 'province': '서울특별시', 'block_code': 'b1', 'sex': '남성',
                   'age_band': '50', 'activity_decline': .3, 'candidate': True,
                   'quality_eligible': True, 'persistent_20pct': case['detection'] == 'persistent'}
            with search.policy_session():
                observe_candidates(pd.DataFrame([row]), 'anomaly')
                ctx = enrich_context(config, ctx, search.SESSION.get()['observations'])
        pages, results = {}, []
        for doc in case['docs']:
            url = 'https://www.seoul.go.kr/policies/' + doc['id']
            name = doc.get('name', doc['id'] + ' 사업')
            body = (f"사업명: {name}\n담당기관: 서울시\n사업연도: 2026\n모집차수: {doc.get('round', 1)}차\n"
                    f"지원대상: {doc.get('target', '서울시 주민')}\n"
                    f"신청기간: {doc.get('period', '2026.10.01 ~ 2026.10.31')}\n"
                    f"신청방법: 주민센터 방문\n지원내용: {doc['support']}\n"
                    '1인가구 사회적 고립 예방 사업의 공식 안내입니다. 신청 시 세부 자격을 확인해야 합니다.')
            pages[url] = '<main>' + body.replace('\n', '<br>') + '</main>'
            results.append({'url': url, 'title': doc.get('title', name + ' 모집 공고')})
        result = search.lookup(config, ctx, now=datetime(2026, 10, 24, tzinfo=search.KST),
                               search=lambda *args: results, fetch=pages.__getitem__)
        assert result['status'] == 'ok', (case['id'], result['errors'])
        actual = [p['url'].rsplit('/', 1)[-1] for p in result['policies']]
        all_links = {s['url'].rsplit('/', 1)[-1] for p in result['policies'] for s in p['related_sources']}
        assert not set(case.get('excluded', [])) & all_links, case['id']
        assert set(case['expected']) <= set(actual), (case['id'], actual)
        if case.get('expected_first'):
            assert actual[0] == case['expected_first'], (case['id'], actual)
        if 'groups' in case:
            assert len(result['policies']) == case['groups'], case['id']
        # Earlier implementation sorted accepted documents by source-region,
        # literal purpose overlap, recruitment, URL; all fixture hosts are Seoul.
        old = sorted(accepted, key=lambda p: (-p['relevance'],
            {'접수 중': 0, '상시': 1, '접수 예정': 2, '확인 필요': 3}[p['recruitment']], p['url']))[:3]
        expected = set(case['expected'])
        old_ids = {p['url'].rsplit('/', 1)[-1] for p in old}
        if case.get('expected_first'):
            first_total += 1
            baseline_first += int(old[0]['url'].rsplit('/', 1)[-1] == case['expected_first'])
            new_first += int(actual[0] == case['expected_first'])
        baseline_hits += len(expected & old_ids)
        new_hits += len(expected & set(actual))
        expected_total += len(expected)
        reports.append({'case': case['id'], 'old_top3': sorted(old_ids), 'new_top3': actual})
    assert new_hits >= baseline_hits
    assert new_first >= baseline_first
    # Persist the exact fixed-corpus comparison in the isolated test output.
    directory = config.output / 'policies'
    (directory / 'evaluation.json').write_text(json.dumps({
        'cases': reports, 'expected_total': expected_total, 'baseline_hits': baseline_hits,
        'new_hits': new_hits, 'first_rank_cases': first_total, 'baseline_first': baseline_first,
        'new_first': new_first, 'clear_condition_violations': 0, 'incorrect_merges': 0},
        ensure_ascii=False, indent=2), encoding='utf-8')
