"""Support-purpose hypotheses from validated group observations, never diagnoses."""
import math

PURPOSES = {
    'participation': {'label': '사회적 관계망·지역 활동 참여',
                      'terms': ['관계망', '사회적 관계', '교류', '모임', '활동 참여', '커뮤니티']},
    'checkin': {'label': '안부 확인·지속적 관계 지원',
                'terms': ['안부', '정기 방문', '지속적 관계', '정기 연락', '주기적 연락']},
}


def observe_candidates(frame, kind):
    from .search import SESSION
    session = SESSION.get()
    if session is None:
        return
    observations = []
    for row in frame.to_dict('records'):
        decline = row.get('activity_decline')
        if not isinstance(decline, (int, float)) or not math.isfinite(decline) or decline <= 0:
            continue
        if kind == 'anomaly':
            if row.get('quality_eligible') is not True or row.get('candidate') is not True:
                continue
            identity = ':'.join(str(row[k]) for k in ('month', 'block_code', 'sex', 'age_band'))
            region, age = row['province'], row['age_band']
            persistent = row.get('persistent_20pct') is True
        else:
            if row.get('signal') not in ('mobility', 'district_mobility') or row.get('quality') != 'ok':
                continue
            if not isinstance(row.get('score'), (float, int)) or not math.isfinite(row['score']):
                continue
            identity, region, age = row['feature_id'], row.get('region'), row.get('age')
            # Use an existing flag only; do not invent an activity threshold.
            persistent = row.get('persistent_20pct') is True
        if row.get('sex') not in ('남성', '여성') or str(age) not in ('10', '20', '30', '40', '50', '60+'):
            continue
        observations.append({'evidence_id': identity, 'region': region, 'age': age,
            'sex': row.get('sex'), 'month': row.get('month'), 'activity_decline': decline,
            'persistent_20pct': persistent, 'kind': kind,
            'unit': '지역·집단 관측: 개인 판정이나 감소 원인 확정 아님'})
    session['observations'] = observations


def enrich_context(config, context, observations):
    from .collector import source_registry
    from .search import resolve_region
    registry = source_registry(config)
    selected = resolve_region(registry, context.get('region'))
    if not selected:
        return context
    matched = []
    for item in observations:
        scope = resolve_region(registry, item.get('region'))
        if not scope or not (scope['region'] == selected['region'] or scope.get('parent') == selected['region']):
            continue
        if any(context.get(key) and str(context[key]) != str(item.get(key)) for key in ('age', 'sex')):
            continue
        matched.append(item)
    if not matched:
        return context
    hypotheses = []
    for key in PURPOSES:
        evidence = [item['evidence_id'] for item in matched if key == 'participation' or item['persistent_20pct']]
        if evidence:
            hypotheses.append({'purpose_id': key, 'label': PURPOSES[key]['label'],
                'evidence_ids': sorted(set(evidence)), 'status': '지원 필요 확정이 아닌 정책 탐색 가설'})
    return {**context, 'observations': matched, 'hypotheses': hypotheses}


def purpose_matches(text, context):
    """Quoted support-content matches; eligibility demographics are not need evidence."""
    matches = []
    explicit = context.get('purpose', '')
    explicit_terms = [t for t in explicit.split() if len(t) >= 2 and t not in {'지원', '예방', '활동', '1인가구'}]
    if context.get('purpose_explicit') and '고립' in explicit:
        explicit_terms = list(dict.fromkeys([*explicit_terms,
            *PURPOSES['participation']['terms'], *PURPOSES['checkin']['terms']]))
    if explicit_terms:
        found = [t for t in explicit_terms if t in text]
        if found:
            matches.append({'purpose_id': 'user', 'label': explicit, 'terms': found,
                            'evidence_ids': [], 'status': '사용자 요청 목적'})
    for hypothesis in context.get('hypotheses', []):
        found = [term for term in PURPOSES[hypothesis['purpose_id']]['terms'] if term in text]
        if found:
            matches.append({**hypothesis, 'terms': found})
    return matches
