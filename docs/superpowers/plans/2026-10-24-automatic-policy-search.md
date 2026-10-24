# Automatic Policy Search Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement inline.

**Goal:** Always attach verified policy discovery to nonempty analysis questions,
defaulting to nationwide common programs when no region is supplied.

**Architecture:** Keep the shared response decorator and request-scoped cache.
Remove policy intent classification from vLLM. Normalize missing policy scope to
nationwide and retain explicit invalid-scope and eligibility checks.

**Tech Stack:** Python, pytest, existing Tavily/vLLM adapters.

**Spec:** docs/superpowers/specs/2026-10-24-automatic-policy-search-design.md

## Global Constraints

- Test with /opt/venvs/social/bin/python. Never print credentials.
- Preserve the pre-existing CSV newline restoration and analytical artifacts.
- Work in the user's requested checkout; no commits, merges, or new worktrees.
- Keep existing search budgets and selection rules; no unrelated dotenv changes.

## Review Focus

- Missing versus explicitly invalid or ambiguous geographic scope.
- National source pages containing municipality-only eligibility.
- Duplicate searches through outer/inner vLLM answer paths or fallbacks.
- Blank questions and no-candidate answers.
- Observation hypotheses across regions without inferring residence.

### Task 1: Automatic discovery and nationwide scope

Files: policies/recommendation.py, search.py, matching.py, llm/vllm.py;
tests/test_policy_automatic.py, existing affected tests; README.md,
docs/policy-recommendation.md.

- [x] Write regression tests covering local/activity/isolation and vLLM answers
  without policy words, nationwide defaults/domain restrictions, eligibility,
  invalid regions, empty questions, observations, and request reuse/failure.
- [x] Run new tests and confirm failures reflect the missing behavior.
- [x] Remove keyword/plan-flag gates, default only unspecified region to nationwide,
  and enforce common-program scope while retaining demographic validation.
- [x] Update affected old expectations and user documentation.
- [x] Run targeted tests, then the complete pytest suite and git diff --check.
- [x] Obtain independent review; address concrete correctness findings and verify.

## Execution ledger

Ruling: implement in place as explicitly requested by the user; preserve existing
dirty CSV. The chat-approved design is recorded above, without another approval step.

Initial RED: 11 failed, 11 passed (keyword gate, absent national scope).
Initial GREEN: 22 passed. Full suite: 191 passed in 107.01s.
Live CLI: a question without policy words or a region produced one nationwide lookup,
one verified national-service candidate, no local domains, and an attached policy section.

Independent review found ordinary household nouns misread as municipalities and
municipality-only eligibility on national pages escaping scope checks. Reproduced
with regression tests (9 failed, 23 passed); fixing both without weakening demographics.

Ruling: recognizing arbitrary bare place names outside the source registry (e.g.
an unknown city without an administrative suffix) needs a broader geographic lexicon,
which this change does not introduce. Document that limitation and always display
the nationwide scope; explicit region filters retain the existing invalid-scope check.
National candidates must still have positive common-program evidence and no detected
local eligibility. No residence is inferred from an unrecognized word.

Review fixes GREEN: 123 related tests passed; final automatic-policy tests: 34 passed.
A further regression identified the ordinary-word suffix `주시` masking 청주시;
2 failing cases were reproduced and fixed by removing that exception.
Final complete suite: 203 passed in 115.61s using /opt/venvs/social/bin/python.
Final live CLI: a question containing 1인가구 but no policy request or region
completed in 28.69s, attached the nationwide scope and one verified national-service
candidate. Search status was partial due to some detail-fetch failures. Record:
/tmp/activity-agent-artifacts/automatic-policy-smoke.json.
Whitespace checked with core.whitespace=cr-at-eol to preserve the pre-existing,
checksum-required CRLF model reference file. No code credentials or dotenv changes.
