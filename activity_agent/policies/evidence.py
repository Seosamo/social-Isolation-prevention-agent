"""Bounded evidence windows with stable offsets in the collected official body."""
import hashlib
import re

LABELS = r'지원대상|참여대상|모집대상|신청자격|지원지역|신청\s*기간|접수\s*기간|모집\s*기간|신청\s*방법|접수\s*방법'


def chunk_text(text, size=1800, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind('\n', start + size // 2, end)
            if boundary > start:
                end = boundary + 1
        content = text[start:end]
        chunks.append({'id': 'chunk-' + str(start), 'start': start, 'end': end, 'text': content,
                       'sha256': hashlib.sha256(content.encode()).hexdigest()})
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def select_chunks(text, terms, limit=4):
    chunks = chunk_text(text)
    # Required condition windows precede optional purpose windows; the full text
    # remains the source of the rule-based eligibility/deadline checks.
    patterns = (r'지원대상|참여대상|모집대상|신청자격|지원지역',
                r'신청\s*기간|접수\s*기간|모집\s*기간', r'신청\s*방법|접수\s*방법')
    selected = []
    for pattern in patterns:
        candidates = [c for c in chunks if re.search(pattern, c['text'])]
        if candidates:
            best = max(candidates, key=lambda c: len(re.findall(LABELS, c['text'])))
            if best not in selected:
                selected.append(best)
    scored = sorted(chunks, key=lambda c: (-sum(term in c['text'] for term in terms), c['start']))
    for chunk in scored:
        if len(selected) >= limit:
            break
        if chunk not in selected:
            selected.append(chunk)
    return sorted(selected[:limit], key=lambda c: c['start'])
