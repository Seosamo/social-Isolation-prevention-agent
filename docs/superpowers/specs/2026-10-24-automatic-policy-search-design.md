# Automatic policy search

Approved in conversation: always include policy discovery in activity/isolation answers,
in both local and vLLM modes. Without a requested region, search nationwide common
programs. Keep specified local scopes and national alternatives; do not infer residence.

Use the existing shared response decorator and request session. Remove the keyword
gate and the model's `policies` flag rather than forcing that flag to true or creating
a second search path. Preserve memoization, eligibility checks, budgets, and fallbacks.
Blank questions do not search. Invalid/ambiguous explicit regions retain the existing
region-confirmation response. Nationwide searches must exclude local-only programs,
including local conditions found on central-government pages. Valid observed declines
may supply support-purpose hypotheses without changing the national search scope.

No dotenv changes, API-key changes, estimator changes, or deployment actions.
