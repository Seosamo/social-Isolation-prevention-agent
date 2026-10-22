# LLM agent에 이상탐지 결과 전달

기존 추론 결과를 조회하므로 재학습하거나 예측을 변경하지 않습니다.
현재 설정에서 선택한 운영 모델을 사용합니다. 기본 선택은 IF_F2이며, `isolation.model_id=IF_BASELINE`일 때는 기존 모델을 사용합니다. 모델 변경 후 `detect --task isolation`을 다시 실행해야 합니다.

```powershell
python -m activity_agent agent-evidence --month 202512 --region 11 --limit 10
python -m activity_agent agent-evidence --month 202512 --region 11 --limit 10 --output artifacts/isolation/anomaly/agent_evidence.json
```

```python
from activity_agent.common.config import Config
from activity_agent.isolation.queries import AnomalyQueryTools

tools = AnomalyQueryTools(Config.load())
payload = tools.candidate_evidence(month="202512", region="11", limit=10)
# payload는 JSON 직렬화 가능한 dict입니다. LLM tool 응답으로 전달합니다.
```

`ask --task isolation --backend vllm`도 요약 호출의 `structured_evidence`로
동일 계약을 전달합니다. 사람이 읽는 기존 근거·정책 조회와 실패 시 로컬 응답은 유지합니다.

계약 버전: `activity-anomaly-evidence-v1`.

- `status`: OK / NO_CANDIDATES / NO_DATA / NOT_EVALUABLE.
- `model`: 선택 모델 ID, 버전, `score_reference`(training/calibration), 학습 월, 지역 범위, 점수 임계값, 감소 조건.
- `filters`, `summary`: 실제 적용 월·필터, 관측·평가·후보·반환 건수와 잘림 여부.
- `candidates`: 관측 식별자와 evidence_id, 점수, 집단 활동, 지역 공통 활동,
  품질, 규칙별 조건 충족 여부. 선정 규칙은 모델 내부 기여도 설명이 아닙니다.
- `semantics`, `limitations`, `provenance`: 단위·집계 수준·해석 한계·입력 경로.

감소율은 0.2가 20% 감소이며, 백분위는 0~100입니다. F2는 보정 표본 기준 `calibration_percentile`, Baseline은 학습 표본 기준 `training_percentile`을 반환합니다. 점수는 고립 확률이 아닙니다.
결측·비유한 수치는 JSON null입니다. limit은 1~100으로 제한합니다.
후보는 최신 월(월 미지정 시), 높은 이상 원점수 순으로 반환합니다.
후보가 없거나 관측이 없거나 점수를 산출할 수 없어도 정상임을 의미하지 않습니다.
낮·주말 지표를 해당 성·연령 집단의 독립 관측으로 설명하면 안 됩니다.
이전 두 달 평균 대비 감소와 두 달 모두 20% 감소 여부를 구분해야 합니다.

추론 결과가 없거나 현재 모델 버전과 맞지 않으면 오류가 발생합니다.
먼저 `python -m activity_agent detect --task isolation`을 실행하세요.
