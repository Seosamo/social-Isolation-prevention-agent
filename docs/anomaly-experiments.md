# 서울·강원 활동 위축 이상 탐지 비교 실험

## 실행과 리포트

```powershell
.venv/Scripts/python.exe -B -m activity_agent experiment --suite all
.venv/Scripts/python.exe -B -m activity_agent experiment --suite injection
.venv/Scripts/python.exe -B -m activity_agent experiment --suite models
.venv/Scripts/python.exe -B -m activity_agent experiment --suite ablation
.venv/Scripts/python.exe -B -m activity_agent experiment --suite regional
.venv/Scripts/python.exe -B -m activity_agent experiment --suite temporal

# 완료된 특징·모델·합성 결과 체크포인트를 재사용
.venv/Scripts/python.exe -B -m activity_agent experiment --suite all --resume RUN_ID

# 저장된 결과만으로 HTML 재생성. 모델 재학습 없음
.venv/Scripts/python.exe -B -m activity_agent experiment-report --run-id RUN_ID
```

실제 실행한 결과의 진입점:

`artifacts/isolation/experiments/20261021T110042016328Z/index.html`

HTML은 서버·인터넷·CDN 없이 직접 열 수 있다. Plotly를 각 페이지에 포함하며 후보 탐색 자료도 압축해 HTML 안에 넣는다. 최신 Chrome/Edge에서는 로컬 파일의 압축 자료를 풀어 필터링할 수 있다. 데이터 CSV·Parquet 및 모델 파일은 같은 실행 폴더에 보관한다. 전체 비교 CSV는 클 수 있으므로 브라우저에서는 모델 하나씩 로드한다.

8개 페이지: 개요, 데이터·특징 진단, 모델 비교, 합성 주입, Ablation, 지역·시간 분석, 후보 탐색, 방법·한계. 후보 검색에는 모델·월·지역·성·연령·백분위·후보 여부·소지역 코드 필터와 필터 결과 CSV 다운로드가 있다.

## 운영 파이프라인 보존

새 코드는 `activity_agent/isolation/experiments/` 아래에 있다.

- `features.py`: 원천 공통 격자 집계, 신규 특징, 집단/지역 단위 구분.
- `models.py`: 소지역 80/20 분리, 학습 전용 정규화, IF·LOF·KNN·Robust Z·Dual IF.
- `injection.py`: 원천 집계량에 대한 순수 함수 개입과 특징 재계산.
- `runner.py`: 고유 실행 폴더, 체크포인트, 비교·주입·안정성·후속 월 진단.
- `report.py`: 저장 수치에서 독립 HTML 8페이지 및 비교 CSV 생성.
- `validation.py`: 저장 결과 보조 진단, Python 추적 할당량 측정, 링크·오프라인 브라우저·CSV 일치 검사.

기존 CLI에는 `experiment`와 `experiment-report`만 추가했다. 운영 `run-isolation`, `detect`, `evaluate`, `ask`, `dashboard` 경로와 운영 모델·점수·라벨·기본 설정은 바꾸지 않는다. 실험 전후 운영 파일 SHA-256을 비교한다.

운영 모델 `IF_BASELINE`의 원본 pickle을 복사하고 저장된 점수·후보를 재현한다. 이번 실행은 점수 최대 절대 오차 0, 후보 불일치 0이었다. 모델·점수 경계를 실험 결과에 따라 운영 경로로 자동 승격하지 않는다.

## 집계 수준과 특징량

연령 유동은 소지역×성×연령×월, 시간·요일 유동은 소지역×월이다. 시간·요일 특징은 성·연령별 직접 관측이 아니며 동일 소지역의 12개 집단에 반복 연결된다. F6와 Dual 지역 단계는 소지역×월 한 행으로 학습한다.

상대 감소와 비중의 분모는 **같은 age 원천의 제공된 성·연령 12개 집단 합계**다. 다른 정의의 시간·요일 유동과 절대 규모를 강제로 맞추지 않는다. 이 합계를 고유 인원 또는 실제 주민 전체 인구라고 해석하지 않는다.

신규 특징은 누적 감소, 월간 감소 가속, 두 연속 월간 감소 여부, 집단-지역 상대 감소, 집단 비중 변화, 시간·요일 분포의 base-2 JSD다. 감소가 양수인 비율과 감소가 음수인 비중 차이를 구분한다. 분모 0·분포 합 0은 결측이다. 각 비교는 그 3개월 창의 공통 격자와 80% 커버리지를 검사한다. 모든 월에서 하나의 격자를 고정했다는 가정은 하지 않는다.

Group C의 시도 1인가구·독거노인 통계와 고독사 지표는 모델 입력에서 제외한다. 특히 1인가구 구성비를 소지역별 독립 정답으로 복제하지 않는다.

## 공정 비교

- 학습 월 2025년 10월, 평가 11·12월.
- 10월 소지역 코드를 시도별 고정 seed 42로 80% Train / 20% Calibration 분리한다. 같은 소지역의 성·연령 행이 양쪽에 섞이지 않는다.
- 소량 유동 제외 기준과 정규화는 Train에서만 적합한다. Calibration에서 각 모델의 신규 관측 점수 95% 경계를 정하고 이후 월에는 고정한다.
- 기존 운영 Baseline은 원래의 학습 점수 95% 경계를 유지해 별도로 표시한다. 운영 Baseline과 보정 비교 IF F0의 차이를 특징 개선으로 해석하지 않는다.
- F0~F5 및 주 비교 모델은 같은 공통 유효 집단 관측을 사용한다. F6는 별도 지역 단위다. `FULL_IF_F0`~`FULL_IF_F6`는 각 특징 집합의 전체 유효 관측 비교다. 이번 자료에서는 집단 모델들의 전체/공통 유효 집합이 동일했다.
- 모든 비교 모델은 동일한 형태의 위축 후처리를 사용한다. 현재 집단 유동(지역 전용 모델은 해당 지역 합계)이 10% 이상 감소하고 낮·주말 중 하나가 감소해야 후보가 된다.
- 정규화 N0는 기존 시도×성×연령, N1는 시도별 통합, N2는 시도별 정규화·별도 모델, N3는 성×연령 통합이다. 작은 집단과 거의 0인 IQR의 대체·floor 사용은 기록한다.
- LOF는 `novelty=True`다. 학습 표본의 LOF 점수로 보정하지 않는다. KNN은 20번째 이웃 거리, Robust Z는 median/MAD 기반 최대 절대 편차이며 0에 가까운 MAD는 표준편차로 대체한다.
- Dual IF는 집단/지역 모델을 독립 학습하고 각각 보정 CDF 백분위로 변환해 기본 0.5/0.5로 결합한다. 0.25/0.75 대안은 민감도 진단이며 평가 결과로 가중치를 최적화하지 않는다.

## 합성 개입

모델·정규화·경계는 고정하고 같은 평가 소지역의 원본/변형 관측을 쌍으로 저장한다. 기본적으로 시도×월별 24개 소지역에서 점수와 무관하게 한 집단을 선택한다. 집단 공통 비교는 같은 96개 쌍을 사용한다. 운영 Baseline과 지역 전용 모델은 별도 유효 모집단·단위임을 표시한다.

- S1: 원본 현재 집단 유동 감소.
- S2: 시간 원천에서 전체 -10%·낮 -40%에 해당하는 선택적 감소. 다른 시간 비중을 조정하되 집단 유동을 임의로 동기화하지 않는다.
- S3: 집단 -35%·같은 age 원천 지역 합 -5%. 남는 집단 유동이 음수가 되는 경우 제외한다.
- S4: 같은 격자에서 직전 유동=(1-d)×두 달 전, 현재=(1-d)×직전으로 연속 월간 감소를 만든다. 겹치는 이전 비교 창에도 같은 직전월 배율을 적용한다.
- S5: 직전 비교를 합성적으로 0으로 안정화한 뒤 현재 비교만 감소시킨다. 이는 실제 정상 라벨을 뜻하지 않는다.
- S6: 원본 현재 집단 유동 증가. 큰 원본 감소는 증가 개입 후에도 감소일 수 있다. 증가 방향 배제율은 변형 후 실제 비교 감소율이 0 이하인 쌍에서 계산한다.
- S7: 같은 소지역의 모든 age 집단에 동일 배율의 공통 충격. 시간·요일 원천을 임의 동기화하지 않는다.
- S8: 시간·요일 합계를 유지하는 분포 이동.
- NC2: 시간·요일 비중을 유지한 전체 규모 변화. JSD의 규모 불변성과 후보 후처리를 점검한다.

10·20·30·40·50% 강도와 S3 명목 35%를 비교한다. 주입 후 분모·유동량 조건을 못 채운 쌍은 보류 수를 별도로 표시한다. 모델 입력이 바뀌지 않는 개입은 후처리 대조로 해석하며 `model_input_changed_rate`를 기록한다. 집단 개입이 적용되지 않는 F6 항목은 SKIPPED다.

## 관찰 결과

아래는 **같은 96개 평가 쌍에서 집단 유동 30% 감소(S1)**를 주입한 최종 후보 비율이다. 실제 사회고립 탐지 Recall이 아니다.

| 모델 | S1 합성 후보 비율 | S7 공통 충격 후보 비율 |
| --- | --- | --- |
| IF F0, 공정 보정 | 15.6% | 15.6% |
| IF F4, 신규 특징 전체 | 62.5% | 17.7% |
| LOF F4 | 65.6% | 69.8% |
| KNN F4 | 68.8% | 66.7% |
| Robust Z F4 | 45.8% | 5.2% |
| Dual IF | 17.7% | 14.6% |

상대 감소·비중 특징(F2)은 S1 61.5%, 공통 충격 10.4%로 반응했다. 시간·요일 JSD만 추가한 F3는 S1 8.3%로 F0보다 낮았다. 신규 특징이나 복잡한 모델이 모든 상황에서 좋아지지는 않았다.

KNN·LOF는 집단 위축에 민감하지만 공통 충격에도 많이 반응했다. 주 비교 추론 59,358행에서 KNN·LOF는 약 12~13초, IF F0/F4는 약 0.7초, Robust Z는 약 0.3초였다. 하드웨어·실행 조건에 따른 값이며 seed·경계·지역 편향과 함께 비교해야 한다.

원본 비교상 실제 증가로 바뀐 S6의 94개 쌍은 최종 위축 후보에서 모두 제외됐다. 이는 모델의 실제 고립 탐지 성능이 아닌 방향 후처리의 검증이다.

## 지속성과 미완료 범위

11월 후보를 12월 동일 격자의 10·11·12월 유동으로 진단한다. 11→12월 10% 이상 감소는 지속 감소, 10월 대비 90% 미만이던 유동이 90% 이상 돌아오면 회복, 그 수준 미만에서 10% 이상 증가하면 반등, 나머지는 정체·부분 회복이다. 후속 유효 관측이 없으면 평가 불가다.

검증된 행사·공휴일의 지역 단위 연계 자료가 없어 해당 Negative Control은 SKIPPED다. 12월 이후 데이터가 없어 장기 지속성도 SKIPPED다. 메모리는 고정 512행 추론의 Python 추적 할당량을 측정했으며 네이티브 전체·학습 peak RSS를 측정했다고 주장하지 않는다.

## 검증과 재현 자료

실행 폴더의 `config/`에는 실제 특징 목록, 설정, 소지역 분할, 운영 파일 해시, 코드 스냅샷을 보관한다. 모델 pickle과 JSON에는 적합된 중앙값·scale 및 보정 경계가 있다. 원천 SHA-256과 기존 특징 재구성 오차, 모델 입력별 표본 수·IQR·상관관계를 별도로 저장한다.

브라우저 검증은 HTTP(S) 리소스를 차단한 상태에서 페이지 8개를 열고, 모델·지역·월 필터 및 후보 다운로드 CSV와 원본 수치가 같은지 확인한다. 결과와 화면은 `data/browser_validation.json`, `data/static_validation.json`, `validation/`에 저장한다.

```powershell
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider
.venv/Scripts/python.exe -B -m activity_agent.isolation.experiments.validation --run-dir artifacts/isolation/experiments/RUN_ID --browser
```

저장된 결과에서 후속 월·가중치 진단을 다시 만들려면 `--finalize`, Python 할당량 보조 측정을 다시 하려면 `--benchmark`를 사용한다. 코드·데이터가 변경됐다면 새 run을 생성한다. 합성 성능, seed 안정성, 후보 규모 중 하나만으로 최적 모델을 강제하지 않는다.
# 모델 저장 경로

운영 모델은 `activity_agent/model/isolation_forest.pkl`, 실험 모델은
`activity_agent/model/experiments/<run_id>/<experiment_id>.pkl`에 저장합니다.
예: `IF_F0.pkl`, `IF_F4.pkl`, `LOF_F4.pkl`, `DUAL_IF.pkl`.
동명 JSON은 특징 목록과 전처리 설정입니다. 결과표의 모델 ID와 파일명이 일치합니다.
실행별 `config/model_directory.json`으로 이동 경로를 해석하므로 실험 재개와
저장 결과의 HTML 재생성에도 같은 모델을 사용합니다. HTML·수치 데이터는 artifacts에 유지합니다.

