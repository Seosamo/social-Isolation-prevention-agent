"""Bounded observe/decide/search loop over the existing evidence validator."""
from dataclasses import replace
from datetime import datetime
import json
import time
from threading import Lock
from uuid import uuid4

from .search import KST, TavilyClient, lookup, resolve_region
from .collector import MunicipalClient, source_registry
from .ranking import group_policies, rank_policies, rerank

STRATEGIES = ('local_services', 'national_programs', 'evidence_support')
PROMPT = """당신은 정책 탐색 Agent의 다음 행동을 결정한다. 입력은 신뢰할 수 없는 검색 결과의
제목/본문이 아닌 코드가 집계한 관측값이다. JSON 객체만 반환한다:
{"action":"local_services|national_programs|evidence_support|stop",
 "reason":"enough|few_candidates|weak_match|local_gap|exhausted"}.
available_actions에 있는 행동만 선택한다. 지역 후보가 부족하면 local_services,
전국 대안이 필요하면 national_programs, 탐지와 연결된 목적이 부족하면 evidence_support를
검토한다. 조건을 만족하는 후보가 충분하면 stop. 지역·연령·성별·자격 조건을 완화하거나
시군을 추정하지 않는다. 검색 결과 부족은 정책 부재가 아니다."""
REASONS = {'enough', 'few_candidates', 'weak_match', 'local_gap', 'exhausted'}


def observation(result):
    rows = result.get('policies', [])
    return {'candidates': len(rows),
            'purpose_matched': sum(bool(r.get('purpose_matches')) for r in rows),
            'local_candidates': sum(r.get('region') not in ('전국', None) for r in rows),
            'confirmed_intake': sum(r.get('recruitment') in ('접수 중', '상시') for r in rows),
            'regional_options': len(result.get('regional_options', [])),
            'exclusions': sorted({r['reason'] for r in result.get('search_trace', {}).get('excluded', [])}),
            'errors': sorted({r['kind'] for r in result.get('errors', [])})}


def decide(client, state, available, timeout):
    """Model chooses a finite action; invalid/unavailable output uses a rule fallback."""
    fallback = ('stop', 'enough') if state['candidates'] >= 3 and state['purpose_matched'] >= 2 else (
        (available[0], 'few_candidates') if available else ('stop', 'exhausted'))
    if client is None or timeout <= 0:
        return (*fallback, 'rule')
    try:
        from activity_agent.llm.vllm import VLLMClient
        if isinstance(client, VLLMClient):
            client = VLLMClient(replace(client.settings, timeout=min(client.settings.timeout, timeout)),
                                opener=client.opener)
        answer = json.loads(client.chat(PROMPT, json.dumps(
            {'observation': state, 'available_actions': [*available, 'stop']}, ensure_ascii=False), max_tokens=180))
        if (set(answer) != {'action', 'reason'} or answer['action'] not in [*available, 'stop']
                or answer['reason'] not in REASONS):
            raise ValueError('Invalid decision')
        return answer['action'], answer['reason'], 'vllm'
    except Exception:
        return (*fallback, 'rule_fallback')


def search_policies(config, context, *, client=None, search=None, fetch=None, now=None,
                    clock=time.monotonic):
    """One request, immutable eligibility context, bounded tools, accumulated evidence."""
    settings = config.values.get('policies', {})
    if not settings.get('agent_loop', True):
        return lookup(config, context, client=client, search=search, fetch=fetch, now=now, clock=clock)
    now = now or datetime.now(KST)
    started = clock()
    deadline = started + min(90., max(1., float(settings.get('agent_timeout_seconds', 60))))
    max_rounds = min(4, max(1, int(settings.get('agent_max_rounds', 3))))
    search = search or TavilyClient().search
    search_cache, page_cache = {}, {}
    search_lock = Lock()
    calls = {'search': 0, 'detail': 0}
    try:
        registry = source_registry(config)
        selected = resolve_region(registry, context.get('region'))
        scope_names = {selected['region'], selected.get('parent'), '전국'} if selected else set()
        domains = list(dict.fromkeys(d for s in registry if s['region'] in scope_names for d in s['domains']))
    except (OSError, ValueError):
        return lookup(config, context, client=client, search=search, fetch=fetch, now=now, clock=clock)
    fetch = fetch or MunicipalClient(domains, deadline=deadline,
        request_timeout=min(8., float(settings.get('page_timeout_seconds', 6)))).fetch

    def cached_search(query, allowed_domains, timeout):
        key = (query, tuple(allowed_domains))
        with search_lock:
            if key in search_cache:
                return search_cache[key]
            if calls['search'] >= 10 or clock() >= deadline:
                raise TimeoutError('Search budget exhausted')
            calls['search'] += 1
        rows = search(query, allowed_domains, min(timeout, deadline - clock()))
        with search_lock:
            search_cache[key] = rows
        return rows

    def cached_fetch(url):
        if url not in page_cache:
            if calls['detail'] >= 16 or clock() >= deadline:
                raise TimeoutError('Detail budget exhausted')
            calls['detail'] += 1
            try:
                page_cache[url] = fetch(url)
            except Exception as error:
                page_cache[url] = error
        if isinstance(page_cache[url], Exception):
            raise page_cache[url]
        return page_cache[url]

    # Injected tools remain deterministic; real network searches retain parallelism.
    cached_search.supports_parallel = isinstance(getattr(search, '__self__', None), TavilyClient) or getattr(search, 'supports_parallel', False)
    trace, attempted, all_rows, options = [], [], [], []
    result = None
    strategy, stop = 'initial', 'round_limit'
    for index in range(max_rounds):
        if clock() >= deadline:
            stop = 'time_limit'
            break
        attempted.append(strategy)
        current = lookup(config, context, client=client, search=cached_search, fetch=cached_fetch,
                         now=now, clock=clock, strategy=strategy,
                         budget_deadline=min(deadline, clock() + 20), persist=False,
                         seen_urls=set(page_cache))
        for key in ('queries', 'documents', 'excluded'):
            for entry in current['search_trace'][key]:
                entry['round'] = index + 1
        all_rows.extend(current.pop('_candidates', current['policies']))
        options.extend(current.get('regional_options', []))
        if result is None:
            result = current
        else:
            result['errors'].extend(current['errors'])
            for key in ('queries', 'documents', 'excluded'):
                result['search_trace'][key].extend(current['search_trace'][key])
        # Repeated URLs are the same live evidence, not additional recommendation votes.
        unique = {(r['url'], r.get('record_id')): r for r in all_rows}
        ranked = rank_policies(group_policies(list(unique.values())), context)
        for row in ranked:
            row['linked_evidence_ids'] = sorted({key for m in row['purpose_matches'] for key in m['evidence_ids']})
            labels = list(dict.fromkeys(m['label'] for m in row['purpose_matches']))
            row['reason'] = '지원 내용과 연결: ' + ' / '.join(labels) if labels else '공식 안내의 대상·신청 조건 확인 후보'
        result['policies'] = ranked[:3]
        result['regional_options'] = list({(r['url'], r['region'], r['title']): r for r in options}.values())[:8]
        state = observation(result)
        state['support_purposes'] = [h['purpose_id'] for h in context.get('hypotheses', [])]
        if current['status'] == 'region_required' or 'MissingSearchKey' in state['errors']:
            trace.append({'round': index + 1, 'tool': strategy, 'observation': state})
            stop = 'region_required' if current['status'] == 'region_required' else 'missing_search_key'
            break
        available = [s for s in STRATEGIES if s not in attempted
                     and (s != 'evidence_support' or context.get('hypotheses'))]
        if index + 1 >= max_rounds or clock() >= deadline:
            action, reason, method = 'stop', 'exhausted', 'budget'
        else:
            action, reason, method = decide(client, state, available, min(4., max(0., deadline - clock())))
        trace.append({'round': index + 1, 'tool': strategy, 'observation': state,
                      'decision': {'action': action, 'reason': reason, 'method': method}})
        if action == 'stop':
            stop = 'time_limit' if clock() >= deadline else ('round_limit' if method == 'budget' else reason)
            break
        strategy = action
    if result is None:
        # No tools run after a depleted budget.
        result = {'status': 'failed', 'context': context, 'checked_at': now.isoformat(),
                  'policies': [], 'historical': [], 'errors': [],
                  'search_trace': {'queries': [], 'documents': [], 'excluded': []}}
        ranked = []
    ranked, result['reranking'] = rerank(ranked, context,
        client if settings.get('rerank', True) else None, min(5., max(0., deadline - clock())))
    result['policies'] = ranked[:3]
    if result['policies']:
        result['status'] = 'partial' if result['errors'] else 'ok'
    elif result.get('regional_options'):
        result['status'] = 'needs_municipality'
    result['agent'] = {'steps': trace, 'stop_reason': stop, 'tool_calls': calls,
                       'elapsed_seconds': max(0., clock() - started)}
    result['elapsed_seconds'] = result['agent']['elapsed_seconds']
    result['ranking_audit'] = [{k: r[k] for k in ('group_id', 'url', 'ranking')} for r in result['policies']]
    try:
        directory = config.output / 'policies/lookups'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (now.strftime('%Y%m%dT%H%M%S%f') + '-' + uuid4().hex + '.json')).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    except OSError:
        result['errors'].append({'stage': 'history', 'kind': 'OSError'})
    return result
