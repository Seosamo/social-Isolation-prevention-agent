"""Regressions identified by the Seoul/Gangwon live-search experiments."""
import json
from threading import Barrier

from activity_agent.policies import ranking, search
from activity_agent.policies.collector import source_registry
from activity_agent.policies.tables import policy_rows


def catalog():
    return '''<main><table><tr><th>사업명</th><th>지원대상</th><th>지원내용</th><th>신청방법 및 제출서류</th><th>담당부서 및 연락처</th></tr>
    <tr><td>사회적 고립 관계망 지원</td><td>속초시 거주 1인가구</td><td>사회관계망 형성 및 안부 확인</td><td>복지관 방문 신청</td><td>복지정책과</td></tr>
    <tr><td>노인 안부 지원</td><td>65세 이상<br>속초시 주민</td><td>정기 안부확인</td><td>주민센터 방문 신청</td><td>노인복지과</td></tr></table></main>'''


def test_pdf_list_and_background_are_removed_before_detail_fetch():
    for url in ['https://www.bokjiro.go.kr/book.pdf', 'https://state.gwd.go.kr/egf/bp/common/front/123/download',
                'https://www.seoul.go.kr/list.do']:
        assert search.document_exclusion(url)
    assert search.document_exclusion('https://www.mohw.go.kr/menu.es?mid=policy') is None
    a = 'https://www.mohw.go.kr/board.es?list_no=123&mid=a&nPage=1'
    b = 'https://www.mohw.go.kr/board.es?list_no=123&mid=b&nPage=8'
    assert ranking.document_key(a) == ranking.document_key(b)
    assert ranking.document_key('https://www.mohw.go.kr/menu.es?mid=a') != ranking.document_key('https://www.mohw.go.kr/menu.es?mid=b')


def test_service_questions_and_alternative_age_branches(config):
    text = ('일상돌봄 서비스 사업\n누가 이용할 수 있나요?\n'
            '돌봄이 필요한 청·중장년(13~64세) 또는 가족돌봄청년(39세 이하)\n'
            '어떤 서비스를 제공하나요?\n관계망 교류 지원\n서비스 신청은 어떻게 하나요?\n'
            '주소지 주민센터 방문 신청\n어떤 서류가 필요한가요?\n신분증')
    fields = search.extract_fields(text)
    assert '13~64세' in fields['eligibility'] and '39세 이하' in fields['eligibility']
    assert '주소지 주민센터' in fields['application']
    registry = source_registry(config)
    ctx = search.question_context(config, '강원 50대 남성 돌봄 지원')
    assert search.compatible(fields, ctx, {'region': '전국', 'national_program': True}, registry)


def test_bare_column_headings_do_not_count_as_verified_conditions():
    fields = search.extract_fields('지원대상\n지원내용\n신청방법 및 제출서류\n담당부서 및 연락처')
    assert fields == {'eligibility': '', 'period': '', 'application': ''}


def test_table_rows_keep_age_and_conditions_separate():
    rows = policy_rows(catalog())
    assert len(rows) == 2
    assert '65세' not in rows[0]['text'] and '65세' in rows[1]['text']
    assert rows[0]['application'] == '복지관 방문 신청'


def test_table_rowspan_preserves_shared_target():
    html = '''<table><tr><th>사업명</th><th>지원대상</th><th>신청방법</th></tr>
    <tr><td>안부 지원</td><td rowspan="2">지역 1인가구</td><td>전화 신청</td></tr>
    <tr><td>관계망 지원</td><td>방문 신청</td></tr></table>'''
    rows = policy_rows(html)
    assert len(rows) == 2 and rows[1]['target'] == '지역 1인가구'


def test_province_request_offers_city_choices_without_claiming_provincewide_eligibility(config):
    url = 'https://state.gwd.go.kr/portal/partinfo/welfare/singleperson/localpresent/sokcho'
    provider = lambda *args: [{'url': url, 'title': '시군별 지원사업 현황 속초시'}]
    ctx = search.question_context(config, '강원 50대 남성 고립 지원')
    result = search.lookup(config, ctx, search=provider, fetch=lambda url: catalog())
    assert not result['policies'] and result['status'] == 'needs_municipality'
    assert len(result['regional_options']) == 1
    assert result['regional_options'][0]['region'] == '속초시'
    assert result['regional_options'][0]['title'] == '사회적 고립 관계망 지원'
    municipal = search.lookup(config, search.question_context(config, '속초시 50대 남성 고립 지원'),
                             search=provider, fetch=lambda url: catalog())
    assert len(municipal['policies']) == 1
    assert municipal['policies'][0]['applies_to'] == '속초시'
    assert municipal['policies'][0]['source_cells']['지원대상'] == '속초시 거주 1인가구'


def test_city_query_cannot_recommend_another_city_catalog_with_generic_target(config):
    sokcho = 'https://state.gwd.go.kr/sokcho'
    youngwol = 'https://state.gwd.go.kr/youngwol'
    provider = lambda *args: [{'url': sokcho, 'title': '시군별 지원사업 현황 속초시'},
                              {'url': youngwol, 'title': '시군별 지원사업 현황 영월군'}]
    # Even when the target cell itself omits a city, the official catalog's
    # explicit municipality must constrain the recommendation.
    pages = {sokcho: catalog(), youngwol: catalog().replace('속초시 거주', '관내 거주')}
    ctx = search.question_context(config, '속초시 50대 남성 고립 지원')
    result = search.lookup(config, ctx, search=provider, fetch=pages.__getitem__)
    assert all(row['url'] == sokcho for row in result['policies'])
    assert any(r['reason'] == 'other_municipality' for r in result['search_trace']['excluded'])


def test_searches_can_run_concurrently_with_separate_domain_scopes(config):
    barrier = Barrier(4)
    def provider(query, domains, timeout):
        barrier.wait(timeout=3)
        return []
    provider.supports_parallel = True
    result = search.lookup(config, search.question_context(config, '강원 고립 지원'), search=provider)
    assert result['status'] == 'no_results'
    qs = result['search_trace']['queries']
    assert qs[0]['domains'] == qs[1]['domains'] == ['gwd.go.kr']
    assert 'mohw.go.kr' in qs[2]['domains'] and 'gwd.go.kr' not in qs[2]['domains']


def test_reference_service_is_refetched_not_a_saved_policy_result(config):
    url = 'https://www.mohw.go.kr/menu.es?mid=service'
    path = config.root / 'references/policy_service_sources.json'
    path.write_text(json.dumps([{'url': url, 'required_title': '일상돌봄 서비스',
        'region': '전국', 'national_program': True, 'purpose_terms': ['고립'],
        'note': '지역별 시행 및 자격 확인 필요'}], ensure_ascii=False), encoding='utf-8')
    calls = []
    def fetch(url):
        calls.append(url)
        return '<main>일상돌봄 서비스\n누가 이용할 수 있나요?<br>고립 등 돌봄 필요 청·중장년(13~64세)<br>어떤 서비스를 제공하나요?<br>관계망 교류 서비스 안내입니다.<br>서비스 신청은 어떻게 하나요?<br>주민등록상 주소지 주민센터 방문 신청<br>어떤 서류가 필요한가요?</main>'
    ctx = search.question_context(config, '강원 50대 남성 고립 지원')
    for _ in range(2):
        result = search.lookup(config, ctx, search=lambda *args: [], fetch=fetch)
        assert len(result['policies']) == 1
        assert result['policies'][0]['recruitment'] == '확인 필요'
        assert '지역별 시행' in ' '.join(result['policies'][0]['unknown'])
    assert calls == [url, url]
