# Response Prompt Integration Plan

User specification: activity_agent_response_prompt.md and the explicit request to
place it appropriately and apply it. Implement in the current checkout without
another approval step; no commits or dotenv/model changes.

Architecture: move the document under activity_agent/llm/prompts/activity_response.md.
Load it through importlib.resources for each final-answer generation, retaining the
existing evidence constraints. Include it in wheel package data. A missing, empty,
or invalid UTF-8 prompt must produce the existing diagnostic/local-evidence fallback.
The local backend has no text-generation model; the prompt applies to vLLM generation.

Correct only integration conflicts: the leading typo, true two-month-mean decline
definition, valid negative decrease rates, unchanged numeric/date/identifier tokens,
actual persistence field, and no invented total counts from limited samples.
Automatic policy discovery remains in the shared answer path.

- [x] Add tests at the actual chat payload boundary and observe failures.
- [x] Move/correct the document, add runtime loader and package-data entry.
- [x] Verify request-time refresh and failure fallback; update README usage.
- [x] Run related tests, full suite, distribution/resource check and code review.

Verification: new tests initially 6 failed; related tests 71 passed; complete suite
209 passed in 107.29s using /opt/venvs/social/bin/python. Built a wheel from a temporary
source copy (large unchanged model artifacts excluded); checked identical prompt bytes
and loaded the resource from the wheel outside the source checkout. Independent
read-only review found no issues and separately confirmed all 6 new tests pass.

Live localhost request confirmed the complete document in the final system message
(7,638 characters including evidence guards), separate from the query-plan prompt.
The model produced the analysis-table/policy sections but changed some long identifiers
and the month notation, so unchanged numeric validation rejected that response and
retained diagnostics plus source evidence. See /tmp/activity-agent-artifacts/response-prompt-live.md.
