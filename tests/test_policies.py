import json
from datetime import datetime, timedelta, timezone

from activity_agent.policies.collector import Page, collect, official
from activity_agent.policies.recommendation import recommend, policy_lines


def test_collection_filters_navigation_and_preserves_last_success(config):
    listing = "https://www.gangnam.go.kr/list"
    detail = "https://www.gangnam.go.kr/board/policy/1/view.do"
    source = {"source_id": "test", "region": "서울 강남구", "list_urls": [listing],
              "list_tokens": ["listing"], "detail_pattern": r"/board/", "body_tokens": ["post-content"]}
    (config.root / "references/policy_sources.json").write_text(json.dumps([source]), encoding="utf-8")
    pages = {listing: f'<div class="listing"><a href="{detail}">1인가구 지원</a><a href="https://evil.example/board/1">1인가구 지원</a></div>',
             detail: '<title>정책 테스트</title><nav>다른 지역 메뉴</nav><div class="post-content">1인가구 사회적 고립 예방 지원 안내. ' + "거주지와 신청 조건을 확인하세요. " * 10 + '</div>'}
    calls = []
    def fetch(url):
        calls.append(url)
        return pages[url]
    result = collect(config, fetch)
    assert result["collected"] == 1 and not result["errors"]
    assert calls == [listing, detail]
    row = json.loads((config.output / 'policies/catalog.json').read_text(encoding='utf-8'))[0]
    assert "다른 지역 메뉴" not in row["excerpt"]
    assert row["status"] == "needs_verification"
    assert recommend(config, None) == []
    old = (config.output / "policies/catalog.json").read_bytes()
    def fail(url):
        raise OSError("offline")
    failed = collect(config, fail)
    assert failed["errors"] and failed["previous_catalog_preserved"]
    assert (config.output / "policies/catalog.json").read_bytes() == old
    from activity_agent.policies.search import lookup, question_context
    result = lookup(config, question_context(config, '', region='강남구'), search=fail)
    assert result['status'] == 'failed' and not result['policies']
    assert result['historical'][0]['url'] == detail
    assert '과거 확인 자료' in '\n'.join(policy_lines(config, result=result))


def test_official_hosts_and_body_extraction():
    assert official("https://www.gangnam.go.kr/board/1")
    assert not official("https://www.gangnam.go.kr.evil.example/board/1")
    assert not official("http://www.gangnam.go.kr/board/1")
    page = Page(["article"])
    page.feed('<div>1인가구 메뉴</div><div class="article">본문<script>bad()</script><p>정책</p></div>')
    assert page.text == "본문\n정책"


def test_policy_settings_do_not_change_analysis_version(config):
    from activity_agent.common.config import Config
    values = dict(config.values)
    values["policies"] = {"max_age_days": 1}
    assert Config(config.root, values).version == config.version
