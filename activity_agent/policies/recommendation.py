"""Present request-time official policy evidence alongside analytical evidence."""
import inspect
import json
from functools import wraps

from .search import SESSION, POLICY_WORDS, lookup, policy_session, question_context


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
        return ['정책 검색에는 서울·강원의 시도 또는 시군구를 지정해 주세요. 전국 공통 정책도 함께 검색합니다.']
    lines = [f"\n**정책 조회 · {result['checked_at']} · 분석 월과 별개인 현재 기준**"]
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
    if result.get('historical'):
        lines.append('\n**과거 확인 자료 — 현재 상태 미확인 · 현재 추천 아님**')
        for row in result['historical']:
            title = str(row['title']).replace('[', '［').replace(']', '］').replace('\n', ' ')
            lines.append(f"- [{title}](<{row['url']}>) / 과거 확인 시각 {row['checked_at']}")
    return lines


def with_policy_response(method):
    """One policy lookup per answer, including when no anomaly was detected."""
    signature = inspect.signature(method)

    @wraps(method)
    def wrapped(self, question, *args, **kwargs):
        if self.backend or not any(word in question for word in POLICY_WORDS):
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
            evidence = method(self, question, *args, **kwargs)
            return evidence + '\n' + '\n'.join(policy_lines(config, context=context))
    return wrapped
