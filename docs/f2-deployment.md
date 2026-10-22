# IF_F2 운영 탐지 실행

현재 `config/default.json`의 고립 의심 과제는 저장된 `IF_F2`를 선택합니다.

```json
"model_id": "IF_F2",
"model_path": "activity_agent/model/experiments/20261021T110042016328Z/IF_F2.pkl",
"flow_cutoffs_path": "/tmp/activity-agent-artifacts/isolation/experiments/20261021T110042016328Z/data/flow_cutoffs.csv"
```

이 설정은 `isolation` 안에 있습니다. 최상위 `detector=rule`은 별도의 활동 위축 과제에 해당합니다. F2를 실행할 때는 `--task isolation`을 지정합니다. 저장 모델과 동명 JSON은 그대로 사용하며 `isolation_forest.pkl`을 덮어쓰지 않습니다.

F2는 기존 6개 활동 변화율에 집단과 지역의 상대 감소율 `relative_decline`, 집단 비중 변화 `share_change`를 추가합니다. 동일 age 원천의 12개 성·연령 집단 합계를 지역 분모로 쓰며 공통 격자·커버리지를 검사합니다. 시간·요일 특징은 소지역 전체의 관측입니다.

저장된 정규화와 2025년 10월 보정 표본의 95% 임계값 `0.5552859724396212`를 고정합니다. 백분위는 보정 표본 대비 위치이며 고립 확률이 아닙니다. 10% 이상 집단 유동 감소와 낮 또는 주말 유동 감소가 함께 있어야 후보입니다.

## 실행 순서

프로젝트 폴더에서 사용자가 만든 uv 가상환경의 Python을 사용합니다.

```bash
cd /workspace/social-Isolation-prevention-agent
uv pip install --python /opt/venvs/social/bin/python -e .
```

정제 파일이 없는 경우에만 실행합니다. 현재 `/tmp/activity-agent-artifacts/clean/`과 `clean_manifest.json`이 있으면 생략할 수 있습니다.

```bash
/opt/venvs/social/bin/python -B -m activity_agent preprocess
```

공간 매핑이 없으면 원래 실험과 같은 공식 SGIS 2025년 2분기 시도 경계의 SHP·SHX·DBF·PRJ·CPG 파일을 준비하고 실행합니다. 다음 SHP 경로는 실제 파일 위치로 바꿔야 합니다.

```bash
/opt/venvs/social/bin/python -B -m activity_agent spatial-audit \
  --boundary-file data/boundaries/sido_2025/bnd_sido_00_2025_2Q.shp
```

현재 `output_dir`은 `/tmp/activity-agent-artifacts`이고 `isolation.block_mapping_file`도 그 아래 `isolation/block_geography_candidates.csv`를 가리킵니다. 따라서 위 명령이 만드는 후보 매핑과 `spatial_audit.json`을 그대로 사용합니다. `output_dir`을 변경하면 매핑 경로도 함께 변경하세요. 기존 감사 결과를 복원한다면 원본 정제 manifest와 후보 매핑의 SHA-256이 일치해야 합니다.

원래 실험 실행 폴더의 `data/flow_cutoffs.csv`도 복원해야 합니다. 현재 설정은 다음 위치를 읽습니다.

```text
/tmp/activity-agent-artifacts/isolation/experiments/20261021T110042016328Z/data/flow_cutoffs.csv
```

실험 자료가 다른 위치에 있다면 `isolation.flow_cutoffs_path`를 그 파일로 바꾸세요. CSV는 `geo_code,sex,age_band,flow_cutoff` 열을 포함하며 원래 24개 집단의 기준을 보존해야 합니다. 임의로 새 파일을 만들거나 현재 자료에서 추정한 값으로 대체하지 않습니다.

그다음 저장 F2로 추론합니다. Baseline 결과나 기존 실험 특징 파일을 미리 만들 필요는 없습니다.

```bash
/opt/venvs/social/bin/python -B -m activity_agent detect --task isolation
/opt/venvs/social/bin/python -B -m activity_agent readiness
/opt/venvs/social/bin/python -B -m activity_agent evaluate --task isolation
/opt/venvs/social/bin/python -B -m activity_agent agent-evidence --month 202512 --region 11 --limit 10
/opt/venvs/social/bin/python -B -m activity_agent ask --task isolation --backend local \
  "2025년 12월 서울 활동 이상 후보 근거"
/opt/venvs/social/bin/python -B -m activity_agent dashboard
```

대시보드에서 **분석 과제 → 고립 의심**을 선택합니다. localhost vLLM 설명은 질문 명령의 `--backend local`을 `--backend vllm`으로 바꿔 사용합니다.

## 원본 자료 확인과 산출물

과거 실험 pickle에는 소량 유동 제외 기준이 별도로 들어 있지 않아 원래 실험의 `flow_cutoffs.csv`를 함께 사용합니다. 첫 적용 시 원본 2025년 10월 자료의 동일 seed 80/20 소지역 분할로 계산한 기준이 이 CSV와 일치하는지도 검사합니다. 저장된 학습 23,979행·보정 5,989행, 정규화, 보정 점수가 재현되는지 검사하며 추정기나 보정 임계값을 다시 학습하지 않습니다. 이 검사를 위해 원본 7~10월의 정제 자료와 동일한 공간 매핑이 필요합니다. 이후 월의 값으로 기준을 조정하지 않습니다.

자료·매핑이 달라 원본 기준을 재현하지 못하면 `reference ... changed` 오류로 중단합니다. 데이터를 임의로 채우거나 다른 모델로 자동 대체하지 않습니다. 제외 기준 CSV가 없으면 `Missing original IF_F2 flow cutoff reference`, 공간 매핑이 없으면 `No accepted province mapping` 오류로 중단합니다. 당시와 같은 매핑을 만들 수 없다면 과거 실행의 매핑·감사 파일을 복원해야 합니다.

산출물은 `/tmp/activity-agent-artifacts/isolation/anomaly/`에 저장합니다.

- `detections.parquet`, `candidates.csv`, `candidates.html`: F2 점수와 후보.
- `evaluation.json`, `deployment.json`: 모델 버전·학습/보정 규모·월별 후보 분포.
- `readiness.json`: 적용 준비 상태.

근거 JSON의 `model.id`는 `IF_F2`, `model.score_reference`는 `calibration`, 점수 필드는 `calibration_percentile`입니다. 기존 Baseline을 선택한 경우에는 기존 `training_percentile` 필드를 유지합니다. 모델이 바뀌면 이전 모델의 탐지 결과를 화면과 조회에 사용하지 않으므로 `detect --task isolation`을 다시 실행해야 합니다.

`train --task isolation`은 F2 선택 상태에서 중단됩니다. `run-isolation`은 구조 통계를 정제한 뒤 저장 F2로 탐지하며, F2를 재학습하지 않습니다. 비교 실험 재실행은 별도 설정 파일에서 `isolation.model_id=IF_BASELINE`을 선택해 진행하세요.
