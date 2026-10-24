"""Present request-time official policy evidence alongside analytical evidence."""
import inspect
import json
from functools import wraps

from .search import SESSION, policy_session, question_context
from .matching import enrich_context
from .agent import search_policies as lookup


def recommend(config, region=None, limit=3, now=None, **kwargs):
    """Compatibility list interface; every standalone call searches live sources."""
    context = question_context(config, '', region=region)
    return lookup(config, context, now=now, **kwargs)['policies'][:limit]


def policy_lines(config, region=None, *, context=None, result=None):
    if result is None:
        context = context or question_context(config, '', region=region)
        session = SESSION.get()
        key = json.dumps(context, ensure_ascii=False, sort_keys=True)
        if session is not None and key in session['results']:
            result = session['results'][key]
        else:
            result = lookup(config, context, client=session['client'] if session else None)
            if session is not None:
                session['results'][key] = result
    status = result['status']
    if status == 'region_required':
        return ['정책 검색 지역을 확인할 수 없습니다. 서울·강원의 시도 또는 시군구를 지정하거나 전국으로 검색해 주세요.']
    lines = [f"\n**정책 조회 · {result['checked_at']} · 분석 월과 별개인 현재 기준**"]
    if result.get('context', {}).get('region') == '전국':
        lines.append('- 정책 검색 범위: 전국 공통 · 특정 지역 거주를 추정하지 않습니다.')
    if status in {'failed', 'partial'}:
        lines.append('- 실시간 정책 검색 또는 일부 공식 원문 확인에 실패했습니다.'
                     ' 아래에는 이번 요청에서 확인한 근거만 표시합니다.')
    if any(error.get('kind') == 'MissingSearchKey' for error in result.get('errors', [])):
        lines.append('- 정책 검색을 사용하려면 TAVILY_API_KEY 환경변수를 설정해 주세요.')
    if not result['policies']:
        lines.append('- 이번 검색에서 추천할 공식 정책을 확인하지 못했습니다. 정책이 없다는 뜻은 아닙니다.')
    for row in result['policies']:
        clean = lambda text: str(text).replace('\n', ' / ').replace('<', '&lt;').replace('>', '&gt;').replace('[', '［').replace(']', '］')
        lines.extend([
            f"- 공식 정책 확인 후보: [{clean(row['title'])}](<{row['url']}>) / 적용 지역 {row['applies_to']} / {row['recruitment']}",
            f"  - 원문 대상 조건: {clean(row['eligibility'])}",
            f"  - 원문 신청 기간: {clean(row['period'])}",
            f"  - 원문 신청 방법: {clean(row['application'])}",
            f"  - 추천 근거: {row['reason']} / 확인 시각 {row['checked_at']}",
            f"  - 미확인 항목: {'; '.join(row['unknown'])}. 현재 지원 가능 여부와 최종 자격은 원문에서 확인해 주세요."])
        if row.get('ranking'):
            rank = row['ranking']
            lines.append(f"  - 추천 순위 {rank['rank']} / {'vLLM 목적 재평가' if rank['method'] == 'vllm' else '규칙 기반 정렬'}"
                         ' · 순위 점수는 수급 가능성이나 정책 효과 확률이 아닙니다.')
        for match in row.get('purpose_matches', []):
            if match.get('evidence_ids'):
                lines.append(f"  - 정책 탐색 가설: {clean(match['label'])} / 연결 근거 ID: "
                             + ', '.join(f"`{key}`" for key in match['evidence_ids'])
                             + '. 관측된 활동 감소와 연결한 검토 목적이며 필요·원인 확정이 아닙니다.')
        related = [s for s in row.get('related_sources', []) if s['url'] != row['url']]
        if related:
            lines.append('  - 같은 정책의 공식 자료: ' + ' / '.join(
                f"[{clean(s['title'])}](<{s['url']}>)" for s in related))
    if result.get('reranking', {}).get('status') == 'fallback':
        lines.append('- 목적 재평가가 검증을 통과하지 못해 규칙 기반 순위를 제공합니다.')
    if result.get('regional_options'):
        lines.append('\n**시군구 확인 후 검토할 지역 사업 — 시도 전체 적용 추천과 구분**')
        for option in result['regional_options'][:5]:
            title = str(option['title']).replace('[', '［').replace(']', '］').replace('\n', ' ')
            lines.append(f"- {option['region']}: [{title}](<{option['url']}>) / 해당 지역 거주·이용 자격과 접수 여부 확인 필요")
        lines.append('- 실제 거주·이용 시군구를 지정하면 해당 사업의 대상 조건과 신청 경로를 좁혀 추천합니다.')
    if result.get('historical'):
        lines.append('\n**과거 확인 자료 — 현재 상태 미확인 · 현재 추천 아님**')
        for row in result['historical']:
            title = str(row['title']).replace('[', '［').replace(']', '］').replace('\n', ' ')
            lines.append(f"- [{title}](<{row['url']}>) / 과거 확인 시각 {row['checked_at']}")
    return lines


def with_policy_response(method):
    """Always attach one memoized policy exploration, even without anomalies."""
    signature = inspect.signature(method)

    @wraps(method)
    def wrapped(self, question, *args, **kwargs):
        if not question.strip():
            return '질문을 입력해 주세요.'
        if self.backend:
            return method(self, question, *args, **kwargs)
        config = self.config if hasattr(self, 'config') else self.tools.config
        bound = signature.bind(self, question, *args, **kwargs)
        filters = {**bound.arguments.get('filters', {}), **{
            k: v for k, v in bound.arguments.items() if k in ('region', 'age', 'sex')}}
        try:
            context = question_context(config, question, **{k: filters.get(k) for k in ('region', 'age', 'sex')})
        except (OSError, ValueError):
            return method(self, question, *args, **kwargs) + '\n정책 검색 출처 설정을 확인할 수 없습니다.'
        with policy_session():
            SESSION.get()['observations'] = []
            evidence = method(self, question, *args, **kwargs)
            context = enrich_context(config, context, SESSION.get()['observations'])
            return evidence + '\n' + '\n'.join(policy_lines(config, context=context))
    return wrapped
