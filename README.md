# 지역 활동 위축 탐지·예방 Agent

SKT 통신 유동인구와 신한카드 집계 자료를 정제하고, 지역·집단의 활동 위축 신호를 계산하는 초기 파이프라인입니다. 규칙 탐지, 약한 라벨, 검토 이력, 지도학습 실험, 근거 조회와 로컬 대시보드를 연결합니다.

**결과는 개인의 고립 판정이나 검증된 고립 확률이 아닙니다.** 규칙 라벨을 학습한 모델의 높은 성능은 규칙과의 일치도를 뜻합니다. 실제 고립 탐지 성능을 주장하려면 분석 단위·기간에 대응하는 독립 조사·현장 확인 자료가 필요합니다.

## 설치와 실행

Python 3.11 이상을 사용합니다. 아래는 저장소 루트에서 실행하는 Windows PowerShell 명령입니다.

```powershell
python -m venv .venv
python -m pip --python .venv/Scripts/python.exe install -e ".[dev]"
.venv/Scripts/python.exe -m activity_agent run-all
.venv/Scripts/python.exe -m activity_agent dashboard
```

`venv`가 `ensurepip` 단계에서 실패했지만 `.venv/Scripts/python.exe`는 만들어진 경우, 시스템 Python의 pip로 두 번째 명령을 실행할 수 있습니다. 활성화 스크립트 실행은 필요하지 않습니다. 기본 대시보드 주소는 `http://localhost:8501`입니다.

원자료는 다음 구조를 유지합니다. 원자료와 실행 산출물, 가상환경은 Git에서 제외됩니다.

검증 환경의 전체 패키지 버전은 `requirements.lock.txt`에 고정했습니다. 동일 환경이 필요하면 해당 파일을 먼저 설치한 뒤 `install -e . --no-deps`로 프로젝트를 설치하세요.

```text
data/
  skt_통신인구/flow_{age,time,wkdy}_pop_YYYYMM.csv
  신한카드_카드데이터/*데이터1.txt
  신한카드_카드데이터/*데이터2.txt
```

단계별 실행과 로컬 질의:

```powershell
.venv/Scripts/python.exe -m activity_agent validate
.venv/Scripts/python.exe -m activity_agent preprocess
.venv/Scripts/python.exe -m activity_agent features
.venv/Scripts/python.exe -m activity_agent detect
.venv/Scripts/python.exe -m activity_agent label
.venv/Scripts/python.exe -m activity_agent report
.venv/Scripts/python.exe -m activity_agent ask "이번 달 50대 남성 활동 위축과 예방 자원 추천"
.venv/Scripts/python.exe -m pytest -q
```

`validate`는 원자료를 읽어 `validation_quality.json`을 작성하며 정제 데이터나 정제 시점의 품질 기록을 갱신하지 않습니다. `run-all`은 정제부터 보고서까지 순서대로 실행합니다. 검토 DB와 모델은 삭제하지 않습니다. 정제 파일 목록을 기록하므로 입력에서 빠진 월의 오래된 파일이 분석에 다시 포함되지 않습니다. 명령의 `--config` 옵션은 하위 명령 앞에 둡니다.

## 분석 단위와 품질 처리

| 신호 | 분석 단위 | 제한 |
| --- | --- | --- |
| mobility | 소지역 × 성 × 연령 × 월 | 해당 집단의 유동이며 주민이나 동일인의 이동을 뜻하지 않음 |
| time_pattern / weekday_pattern | 소지역 전체 × 월 | 성·연령별 시간대·주말 교차통계가 아님 |
| consumption | 가맹점 시군구 × 성 × 공통 연령 × 월 | 소지역 소비 또는 거주민 소비로 해석하지 않음 |
| district_mobility | 검증된 매핑의 시군구 × 성 × 연령 × 월 | 매핑이 없으면 생성하지 않음 |

- 원자료 공간 키는 `월·소지역 코드·X·Y`입니다. 소지역 코드만으로 원자료를 결합하지 않습니다.
- 완전 중복은 제거하고, 동일 키에 값이 다른 행과 잘못된 값은 `artifacts/quarantine`에 격리합니다. 합산으로 충돌을 숨기지 않습니다.
- SKT 비교는 당월과 직전 2개월의 공통 좌표에서 계산합니다. 세 달 중 하나라도 공통 좌표 비중이 80% 미만이면 점수를 보류합니다. 전체 좌표 기준 감소율도 함께 보관합니다.
- 시간대는 낮 `[09,18)`, 저녁 `[18,22)`, 심야 `[22,24)∪[00,06)`, 아침 `[06,09)`입니다. 주말 비율은 `토·일 평균 / 월~금 평균`입니다.
- 카드1은 소비 분석의 주자료, 카드2는 이용고객 위치별 소비 구성의 별도 집계입니다. 두 자료의 건수를 합산하지 않습니다. 카드의 날짜를 이용한 일별 기상 보정은 초기 버전에 포함하지 않습니다.
- 법인은 카드 정제 자료에 보존하지만 인구집단 특징에서 제외합니다. 카드 60~90대를 합쳐 SKT의 `60+`와 맞춥니다. 원래 연령별 소비는 `card_native_monthly.parquet`에 남깁니다.
- 카드 엔트로피는 업종별 결제건수 비중으로 계산합니다. 관측되지 않은 업종은 0기여이며, 총건수가 0이면 계산 불가입니다.
- 업종 매핑은 해석을 위한 초기 가정입니다. 외식은 혼밥·필수 식사를 포함하고 여가 소비가 대인 교류를 입증하지 않습니다. `industry_inventory.csv`에서 미분류 업종과 분류 근거를 검토하세요.

## 점수와 라벨

`config/default.json`에서 임계값·경로·활성 엔진을 설정합니다. 초기 설정은 검증된 기준이 아닙니다.

- 감소율은 `(직전 2개월 평균 - 당월) / 직전 2개월 평균`입니다. 음수는 증가, 기준값이 0이거나 자료가 없으면 결측입니다.
- 유동 점수는 유동 감소율, 소비 점수는 결제건수·사회/여가 건수·엔트로피 감소율의 동일 가중 평균입니다. 감소율을 각각 0~1로 제한해 0~100으로 변환합니다. 소비의 필수 지표 중 하나라도 없으면 점수를 보류합니다.
- 낮·주말 위축 점수는 설명용 별도 신호입니다. 심야 비중 증가만으로 라벨을 부여하지 않습니다. 서로 다른 신호나 엔진의 점수를 동일 척도로 비교하지 마세요.
- `1 / 위축 의심`: 두 연속 비교에서 각각 20% 이상 감소. 소비는 총건수와 사회/여가 건수 모두 충족해야 합니다.
- `0 / 위축 근거 미약`: 두 연속 비교에서 변화가 각각 ±10% 이내. 소비는 사회/여가 건수도 안정 범위여야 합니다. 고립이 없다는 뜻이 아닙니다.
- `-1 / 판단 보류`: 기간 부족, 저품질, 중간 범위, 상충, 0인 기준값 등. 학습의 음성 표본으로 사용하지 않습니다.
- 7~8월에는 비교 이력이 부족합니다. 연속 비교 라벨은 빨라도 10월부터 가능합니다.
- 시군구 통합 라벨은 시군구 유동 라벨과 소비 라벨이 모두 1 또는 모두 0인 경우만 확정하고 나머지는 보류합니다. 소지역으로 카드 라벨을 복제하지 않습니다.

## 공간·예방 참조자료 추가

`references/block_regions.csv`에 `block_code,region,source_url,verified`를 입력합니다. `verified=true`, HTTPS 출처가 있는 고유 매핑만 사용합니다. `region`은 카드 값과 동일한 `서울 강남구`, `강원 춘천시` 형식을 사용하세요. 코드 접두부를 행정구역으로 임의 추정하지 않습니다.

지도는 원본 좌표계의 정확한 EPSG를 확인한 뒤 설정의 `source_crs`에 넣고 `verified_crs=true`로 바꾸면 활성화됩니다. 정의서에는 UTM-K만 기재되어 있어 기본 설정은 비활성입니다. 변환 결과가 예상 한국 영역을 벗어나면 표시하지 않습니다. 지도 선택은 상세 근거 대상과 연결됩니다.

정책 자료는 지자체 공식 사이트에서 별도 수집합니다. 다음 명령은 `references/policy_sources.json`에 지정한 강남구·춘천시 게시판의 첫 목록 페이지와 지정 상세 페이지에서 1인가구·고립·안부 관련 자료를 수집합니다.

```powershell
python -B -m activity_agent policies-refresh
python -B -m activity_agent ask --task isolation "2025년 12월 강남구 고립 의심 후보와 지원 정책 추천"
```

수집 결과는 `artifacts/policies/catalog.json`, 실패 내역은 `collection_report.json`에 저장합니다. 추천 문구가 포함된 질문에서 활동·고립 양쪽 agent가 지역에 맞는 공식 출처를 연결합니다. 사이트 전체 검색이나 모든 과거 글 수집은 하지 않습니다. 목록 URL에 검색어나 페이지를 추가해 수집 범위를 확장할 수 있습니다. 새 지자체는 소스 설정과 `policies/collector.py`의 허용 호스트를 함께 추가합니다.

robots.txt·요청 간격·페이지 수 제한을 적용하며 HTML 본문만 수집합니다. 이미지·PDF·HWP에만 있는 조건은 자동 판독하지 않습니다. 정책 본문에 언급된 개인 정보나 집단 특성을 모델 입력으로 사용하지 않습니다. 수집 자료는 **정책 확인 후보**이며, 연령·소득·거주 조건과 현재 모집 여부를 확인한 추천이 아닙니다. 과거 보도자료도 포함되므로 반드시 원문을 확인해야 합니다. 현재 추천 정보는 과거 분석 월 당시 이용 가능한 정책을 재현하지 않습니다.

조회 중 외부 요청은 하지 않으며 기본 7일 이내 수집 자료만 표시합니다. 매일 `policies-refresh`를 실행하도록 운영 환경에서 예약할 수 있습니다. 전체 수집 실패 시 기존 파일을 보존하지만 유효기간은 갱신하지 않습니다. 부분 실패는 보고서에 기록합니다. 신규 의존성·검색 API 키는 필요 없습니다.

`references/resources.csv`는 검토자가 직접 확인한 예방 자원 입력표로 함께 유지됩니다. URL·지역·대상 나이·확인일을 입력하면 활동 조회에 연결됩니다. 수집 후보와 수동 검토 자원은 구분해 표시합니다.

기본 질의 기능은 로컬 라우터이며, vLLM을 선택하면 LLM 질문 해석과 근거 기반 설명을 사용합니다. ‘이번 달’은 자료상 최신 월입니다. 기상 수집·실시간 웹 검색·자율 개입 실행은 포함하지 않습니다.

## vLLM 연결

이미 실행 중인 vLLM의 Chat Completions API에 연결합니다. 기본 연결값은 `http://hayoung:8000/v1`, 모델은 `skt/A.X-4.0`입니다. 이 프로젝트는 클라이언트이므로 vLLM 패키지나 GPU를 로컬에 설치할 필요가 없습니다. 서버에는 해당 모델이 chat template과 함께 제공되어야 하며, `VLLM_MODEL`은 `/v1/models`에 노출되는 이름과 같아야 합니다.

```powershell
$env:VLLM_BASE_URL = "http://hayoung:8000/v1"
$env:VLLM_MODEL = "skt/A.X-4.0"
$env:AGENT_BACKEND = "vllm"
# 서버 인증을 사용하는 경우에만 설정
# $env:VLLM_API_KEY = "서버 API 키"

.venv/Scripts/python.exe -B -m activity_agent llm-check
.venv/Scripts/python.exe -B -m activity_agent ask --backend vllm "이번 달 50대 남성 활동 위축과 지원 정책 추천"
.venv/Scripts/python.exe -B -m activity_agent ask --task isolation --backend vllm "2025년 12월 강남구 고립 의심 근거와 정책 추천"
.venv/Scripts/python.exe -B -m activity_agent dashboard
```

대시보드 사이드바의 **Agent 응답 방식**에서 `로컬` 또는 `vLLM`을 선택합니다. CLI에서는 `--backend local`로 서버 호출을 끌 수 있습니다. 선택 우선순위는 CLI/화면 선택 → `AGENT_BACKEND` → `VLLM_BASE_URL`이 있으면 vLLM → `config/default.json`의 `agent.backend`입니다. 기본 설정은 local입니다. `.env.example`은 환경변수 예시이며 `.env`를 자동으로 읽지는 않습니다.

vLLM 실행 흐름은 `질문 → LLM의 조회 조건 JSON → 허용된 로컬 조회 → LLM 설명 → 조회 근거 원문 첨부`입니다. 모델에 임의 코드·SQL·학습·파일 쓰기 도구를 제공하지 않습니다. 질문과 선택된 집계 근거·정책 안내는 설정한 vLLM 서버로 전송됩니다. 원본 데이터 전체를 전송하지 않습니다. 모델별 tool-calling parser나 JSON 강제 출력 지원에 의존하지 않고 일반 채팅 응답의 JSON을 검증합니다.

명시된 월·성별·연령·강남/춘천 지역은 모델이 조회 범위를 넓히지 못하도록 보존하며, CLI/화면 필터가 우선합니다. 지원하지 않는 세부 연령은 기존 안내를 반환합니다. 최종 설명에서 근거에 없는 숫자·URL을 감지하면 설명을 버리고 로컬 근거를 표시합니다. 이는 모든 의미적 오류를 검증하는 것은 아니므로 원문도 항상 함께 제공합니다. 서버 오류·시간 초과·잘못된 JSON·응답 잘림도 명시하고 로컬 조회로 대체합니다.

`VLLM_TIMEOUT`(기본 60초, 요청별), `VLLM_MAX_TOKENS`(설명 기본 1200)를 환경변수 또는 `agent` 설정으로 조정할 수 있습니다. 한 질문에는 일반적으로 해석·설명 두 번의 요청이 발생합니다. `llm-check`는 추론 없이 모델 목록과 이름을 확인하는 명령이며 실제 생성 가능 여부를 보장하지는 않습니다. hayoung에 접속되지 않으면 서버와 같은 네트워크/VPN, 이름 해석, 포트 접근을 확인하세요.

## 검토와 지도학습

대시보드의 라벨 검토 탭 또는 CLI에서 근거를 남깁니다.

```powershell
.venv/Scripts/python.exe -m activity_agent review FEATURE_ID --label 1 --reviewer "검토자" --note "현장 확인 자료와 해당 기간의 판단 근거"
```

`--independent`는 규칙 점수와 독립적인 외부 확인 자료가 있을 때만 사용합니다. 화면 지표를 보고 붙인 라벨을 독립 정답으로 취급하지 마세요. 변경은 덮어쓰기 대신 이력으로 추가합니다. 최신 결정이 이전 결정을 대체하고, 데이터·특징 버전이 바뀌면 과거 검토를 새 버전에 자동 적용하지 않습니다.

```powershell
.venv/Scripts/python.exe -m activity_agent train --signal mobility --source weak --train-end 202510 --validation-month 202511 --test-month 202512
.venv/Scripts/python.exe -m activity_agent evaluate
```

- 첫 모델은 결측 대체·표준화·로지스틱 회귀 파이프라인입니다. 변환은 학습 기간에만 적합합니다. 특징은 명시적 허용 목록으로 제한하고 라벨·검토 텍스트·사후 정보는 제외합니다.
- 기본 학습은 약한 라벨 실험입니다. `--source reviewed`로 최신 검토 라벨만 사용할 수 있습니다. 양·음성 또는 검증·평가 표본이 부족하면 학습을 중단하고 라벨을 임의로 채우지 않습니다.
- 같은 월의 그룹을 무작위로 나누지 않습니다. `--holdout-region "지역명"`은 해당 지역을 학습·검증에서 제외하고 테스트에서 지역 외삽 성능을 별도로 보고합니다. 유동 분석에는 검증된 매핑이 필요합니다.
- `model_evaluation.json`은 **약한 라벨 일치도**와 **독립 검토 라벨 성능**을 구분합니다. 독립 정답이 없으면 평가 행 수 0으로 보고합니다. 정밀도·재현율·PR-AUC(AP)·상위 20개 적중률과 규칙 기준선을 제공합니다. 단일 클래스에서는 AUC를 산출하지 않습니다.
- 모델은 자동 활성화되지 않습니다. 검토 후 `detector`를 `supervised`로 바꾸고 `detect`와 `report`를 실행하세요. 모델이 지원하지 않는 신호와 학습 종료 월까지의 과거 결과는 규칙 엔진을 유지하며 각 행의 엔진·버전을 표시합니다. 미래 자료로 학습한 모델을 과거 월에 적용하지 않습니다.
- 모델 pickle은 로컬에서 직접 생성한 신뢰할 수 있는 파일만 사용하세요. 규칙 라벨 학습 모델의 출력도 고립 확률이 아닙니다.

## 주요 산출물

| 위치 | 내용 |
| --- | --- |
| `artifacts/quality.json`, `quarantine/` | 원본 SHA256, 행 수·총합·중복·오류·가용 범위 |
| `artifacts/clean/` | 정제 SKT 월별 좌표, 카드1·2 별도 집계 |
| `artifacts/features.parquet` | 분석 키, 공통·전체 좌표 특징, 품질, 버전 |
| `artifacts/detections.parquet` | 점수, 상태, 엔진, 근거 지표 |
| `artifacts/weak_labels.parquet`, `combined_labels.parquet` | 약한 라벨과 시군구 통합 라벨 |
| `artifacts/reviews.sqlite3` | 자동 라벨 스냅샷과 추가형 검토 이력 |
| `artifacts/threshold_sensitivity.parquet` | 임계값 15·20·25·30%에서 후보 수와 일치도 |
| `artifacts/REPORT.md`, `run_manifest.json` | 결과 요약, 한계, 실행 설정 |
| `artifacts/models/`, `model_evaluation.json` | 모델·메타데이터와 평가 결과 |

소스는 역할별 패키지로 구분합니다. 공통 코드가 특정 분석 과제에 의존하지 않으며, 수치 계산은 질의 기능과 분리됩니다.

```text
activity_agent/
  common/       # 설정·저장·평가·표시 공통 기능
  data/         # 원자료 정제·공통 좌표 집계
  activity/     # 활동 특징·탐지·라벨·학습·조회·보고서
  isolation/    # 구조 통계·표본·고립 라벨·학습·조회
  policies/     # 지자체 공식 정책 수집·후보 조회
  llm/          # vLLM 통신·질문 해석·근거 기반 설명
  ui/           # 대시보드 진입점과 과제별 화면
  cli.py        # 명령행 진입점
app.py          # Streamlit 실행용 진입점
```

기존 CLI 명령과 `app.py` 실행법은 동일합니다. Python 내부 모듈 경로는 위 패키지 구조를 사용합니다. `candidate_export.parquet` 같은 전체 결과 복사본은 생성하지 않고, 대시보드에서 필요할 때 선택 결과를 CSV로 다운로드합니다. `__pycache__`, pytest 캐시와 실패한 임시 폴더는 실행 산출물이 아니며 보관하지 않습니다.

## 좌표로 시도 연결 후보 확인

공식 [SGIS 2025년 행정구역 경계](https://www.data.go.kr/data/15129688/fileData.do)의 시도 SHP·SHX·DBF·PRJ·CPG 파일을 같은 폴더에 둔 뒤 실행합니다.

```powershell
.venv/Scripts/python.exe -B -m activity_agent spatial-audit --boundary-file data/boundaries/sido_2025/bnd_sido_00_2025_2Q.shp
```

정제 SKT의 모든 월·자료 유형에서 고유 좌표를 모아 UTM-K 후보 EPSG:5179와 EPSG:5178 각각으로 시도 경계에 대조합니다. 동일 `BLOCK_CD`의 모든 좌표가 두 후보에서 같은 시도로 판정될 때만 연결 후보를 만듭니다. 경계 중첩·시도 불일치·미포함 점이 있는 소지역은 보류합니다. 대표 좌표의 평균이나 코드 앞자리로 확정하지 않습니다. `artifacts/isolation/spatial_audit.json`, `spatial_coordinates.parquet`, `block_geography_candidates.csv`에 결과를 기록합니다. 원본 좌표계의 측지계는 여전히 미확인이므로 후보 CSV는 `verified=false`이며 기존 검증 매핑을 덮어쓰지 않습니다.

현재 서울·강원 실험은 `mapping_mode=crs_consensus`를 명시해 이 후보 중 `candidate_crs_agree`이고 `region_codes=["11","32"]`인 소지역만 사용합니다. 경기도와 판정 보류 소지역은 집계·라벨·학습·평가에서 제외됩니다. 감사 보고서의 후보 파일·정제 목록 해시를 검사하므로 파일이 변경되면 `spatial-audit`부터 다시 실행해야 합니다. `spatial_selection.json`에는 실제 선택·제외 개수가 기록됩니다. 원본 좌표계가 검증됐다고 표시하는 설정은 아닙니다. 기본 검증 매핑을 쓰려면 `mapping_mode=verified`, `block_mapping_file=references/block_geography.csv`로 변경합니다.

## 구조 통계 기반 고립 의심 확장 파이프라인

모델 비교·특징량 Ablation·합성 주입·지역/시간 진단과 독립 오프라인 HTML 리포트는 `python -B -m activity_agent experiment --suite all`로 실행합니다. 기존 운영 모델을 보존하며 `artifacts/isolation/experiments/<run_id>/index.html`이 진입점입니다. [실험 방법·실제 결과·재현 명령](docs/anomaly-experiments.md)을 참고하세요.

현재 기본 방법은 **서울·강원 소지역 활동의 비지도 이상 탐지(`isolation.method=isolation_forest`)**입니다. `run-isolation` 또는 `train --task isolation`으로 Isolation Forest를 학습하고 이후 월의 활동 위축 후보를 산출합니다. 양성·음성 라벨 없이 학습하며, 시도 1인가구 통계는 모델 입력에 넣지 않고 설명용 맥락으로만 연결합니다. `readiness`, `evaluate --task isolation`, `detect --task isolation`, `ask --task isolation`과 대시보드의 고립 의심 화면도 이 방법에 연결됩니다. [비지도 학습 결과·평가 범위·실행법](docs/activity-anomaly-model.md)을 참고하세요.

운영 모델과 메타데이터는 `activity_agent/model/isolation_forest.pkl`, `activity_agent/model/isolation_forest.json`에 저장하며 Git 추적 대상입니다. 실험 비교 모델은 `activity_agent/model/experiments/<run_id>/`에 실험 ID 이름으로 저장합니다. 탐지 결과와 HTML 리포트는 기존 `artifacts/`에 보관합니다.

LLM agent용 구조화 JSON은 `python -m activity_agent agent-evidence --month 202512 --region 11 --limit 10`으로 조회합니다. Python 조회 도구와 vLLM 자동 전달 방식은 [agent 전달 계약](docs/agent-evidence.md)을 참고하세요.


지도학습 실험은 `isolation.method=supervised`로 선택할 수 있습니다. 아래 약한 라벨과 양성·음성 학습 준비 조건 설명은 그 실험에 해당합니다. 비지도 결과는 `artifacts/isolation/anomaly/`에 별도 저장하며 기존 라벨 분류 모델을 덮어쓰지 않습니다.

서울·강원 합계에 가려진 소지역 위축은 `python -B -m activity_agent explore-contraction`으로 탐색합니다. 기존 라벨·모델·임계값을 바꾸지 않고 같은 소지역의 단일·지속 위축 비중, 유동량 가중 비중, 소량 유동 민감도, 분할별 규칙 후보 수를 `artifacts/isolation/exploration/`에 저장합니다. 실행 전 현재 설정의 고립 라벨을 준비해야 합니다. [실제 탐색 결과와 학습 가능성 판단](docs/contraction-exploration.md)을 참고하세요.

새 과제 `regional_isolation_suspicion_v1`은 기존 활동 라벨·모델과 분리됩니다. 코드는 `isolation/structural.py`(외부 통계 정제), `isolation/cohorts.py`(공간·연령·시점 결합), `isolation/labeling.py`(학습 기준·라벨·검토), `isolation/training.py`(준비 상태·학습·평가)에 있습니다. 상세 기준은 [설계 문서](docs/isolation-labeling-training-design.md)를 참고하세요.

```powershell
# 기존 SKT·카드 정제와 특징이 준비되어 있어야 합니다.
.venv/Scripts/python.exe -m activity_agent run-isolation

# 단계별 실행
.venv/Scripts/python.exe -m activity_agent structural
.venv/Scripts/python.exe -m activity_agent cohorts
.venv/Scripts/python.exe -m activity_agent label --task isolation
.venv/Scripts/python.exe -m activity_agent readiness
.venv/Scripts/python.exe -m activity_agent train --task isolation
.venv/Scripts/python.exe -m activity_agent evaluate --task isolation
.venv/Scripts/python.exe -m activity_agent detect --task isolation
.venv/Scripts/python.exe -m activity_agent ask --task isolation "2025년 12월 50대 남성 고립 의심 근거"
```

`run-isolation`은 정제·결합·라벨·준비 상태를 기록하고, 준비 조건이 충족된 경우에만 별도 모델을 학습합니다. 조건 미충족은 보고서에 남기며 임의 라벨을 만들지 않습니다. 직접 실행한 `train --task isolation`은 조건이 부족하면 종료 코드 2로 중단하고 기존 모델을 보존합니다. 학습은 자동 활성화되지 않으며 `detect --task isolation`을 명시적으로 실행하면 학습 종료 월 이후의 적용 가능한 표본에만 점수를 저장합니다. 대시보드 사이드바의 **분석 과제 → 고립 의심**에서 준비 상태·라벨 근거·검토·질의를 확인할 수 있습니다.

추가 참조표는 다음 세 개입니다. 검증되지 않은 지역 코드나 예시 프로그램을 실제 자료로 넣지 않았습니다.

| 참조표 | 필수 내용 |
| --- | --- |
| `references/geography.csv` | `geo_level,geo_code,province,district,dong,parent_code,reference_year,source_url,verified` |
| `references/block_geography.csv` | `block_code,geo_level,geo_code,reference_year,source_url,verified` |
| `references/structural_sources.csv` | `source_id,path,reference_period,available_at,url` |

지리 코드표와 `mapping_mode=verified`의 소지역 매핑은 HTTPS 출처와 `verified=true`가 있는 고유한 코드 연결만 사용합니다. 위의 `crs_consensus`는 명시적으로 선택하는 별도 실험 모드입니다. 행정동은 평가 그룹을 구성할 수 있도록 상위 시군구 코드도 등록해야 합니다. 현재 성·연령별 1인가구 CSV는 시도 수준만 선택되어 있어, 정확히 대응하는 시군구/행정동 표본을 라벨링하려면 해당 지역을 선택한 자료가 필요합니다. 코드가 연결되지 않은 자료는 원본 지역명과 품질을 보존하되 해당 집단의 구조 근거로 사용하지 않습니다.

기본값은 **서울·강원 한정, 시도 단위(`sido`), 2025년 사후 분석(retrospective)**입니다. `config/default.json`의 `isolation` 설정에서 `region_codes`·공간 단위(`sido`/`sigungu`/`dong`)·시간 모드·학습/평가 월을 변경하세요. 서울 `11`과 강원 `32`는 이 SGIS 경계의 코드 체계입니다. 서로 다른 공간 단위는 한 모델에 섞지 않습니다. 원본 참조표의 `reference_period`는 네 자리 기준 연도, 확인된 `available_at`은 `YYYY-MM-DD`로 입력합니다. 공표일을 연도만으로 입력해 연초에 공개된 것처럼 처리하지 않습니다. `as_of`는 구조 통계의 공표일이 판정 시점 이전이고 기준 기간도 종료된 경우에만 사용합니다. 공표일을 모르면 배제합니다. `retrospective`에서 연간 통계를 사용한 월별 평가는 당시 실시간 예측의 성능이 아닙니다.

시도 실행에서는 `geography.csv`에 `geo_level=sido`, 시도 코드·정식 시도명·기준 연도·출처를 등록하고 `district,dong,parent_code`는 비워 둡니다. `block_geography.csv`는 SKT 소지역을 같은 시도 코드에 연결합니다. 소지역 코드 앞자리를 임의로 행정구역 코드로 간주하거나 검증 없이 `verified=true`로 채우지 않습니다. 매핑된 관측 격자를 시도별로 합산하되 두 이전 월과의 공통 격자·커버리지 검사를 유지합니다. 부분 지역만 관측한 경우 시도 전체를 대표한다고 해석할 수 없습니다. 강남·춘천 카드 관측은 시도 전체 소비로 합산하지 않으며, 시도 모델의 B는 해당 시도에 연결된 낮·주말 유동 근거를 사용합니다.

새 라벨은 A(집단 유동), S(구조적 취약성), B(활동 보조)의 상태를 저장합니다. A·B는 두 연속 비교를 사용합니다. S는 같은 공간·성·연령의 **전체 1인가구 중 해당 집단 구성비**를 같은 집단의 다른 지역과 비교합니다. 인구 대비 1인가구 비율이 아닙니다. 학습 자료에서만 75% 분위수 기준을 적합하며, 연간 값을 여러 월에 연결해도 한 번만 기준 분포에 반영합니다. 서울·강원 한정 실험은 최소 참조 지역과 최소 학습 지역을 각각 2개로 설정합니다. S는 두 시도의 상대 비교이며 전국 상위 25%나 통계적 충분성을 의미하지 않습니다. 월·성·연령 행을 늘려 지역 수를 채우지 않습니다. 20% 위축·10% 안정 기준은 유지하고 양성 라벨을 만들기 위해 임의 변경하지 않습니다.

평가는 2025년 10월까지 학습 / 11월 검증 / 12월 테스트로, 같은 두 시도의 이후 월을 평가합니다. 다른 지역으로의 일반화 성능으로 보고하지 않습니다. 지역별 지표와 학습 범위는 모델·평가 파일에 기록되며 범위 밖 지역에는 점수를 산출하지 않습니다. 현재 실제 데이터 실행 결과는 [서울·강원 학습 상태](docs/sido-training-status.md)를 참고하세요.

- A·S·B가 support이면 `1 / 고립 의심`입니다.
- A가 안정적이고 구조 근거를 판정할 수 있으며 관측된 B가 모두 안정적이면 `0 / 의심 근거 미약`입니다. 고립 부재 확인이 아닙니다.
- 나머지는 `-1 / 판단 보류`이며 학습 음성으로 바꾸지 않습니다.
- 카드 부재는 0이 아닙니다. 카드 없는 지역도 같은 지역 전체의 낮·주말 자료로 B를 구성할 수 있습니다.
- 65세 이상 통계는 SKT 60세 이상 집단의 정확한 S가 아니며 `_context` 열에만 남깁니다. `20세 미만`을 10대로 임의 분할하지 않습니다.
- **2024년 고독사 수·발생률은 라벨 규칙과 모델 입력에서 모두 제외**하고 별도 배경 설명으로만 조회합니다. 이 자료를 바꾸어도 라벨·학습 특징이 바뀌지 않도록 검증합니다.

검토도 새 과제의 버전에 연결됩니다.

```powershell
.venv/Scripts/python.exe -m activity_agent review COHORT_ID --task isolation --label 1 --reviewer "검토자" --note "확인 근거"
.venv/Scripts/python.exe -m activity_agent train --task isolation --source reviewed
```

SQLite의 기존 활동 테이블은 유지하고 `label_sets`, `label_decisions`, `label_reviews`를 추가합니다. 라벨·검토는 과제·표본·특징 버전·라벨 버전별로 식별합니다. 독립 외부 확인이 있을 때만 `--independent`를 사용하며, 최신 판단 보류나 비독립 검토가 이전 독립 검토를 대체하면 독립 평가에서도 제외합니다.

새 모델은 명시적 숫자·범주 특징만 사용합니다. 지역 코드, 고독사 지표, 라벨 결과와 검토 텍스트를 입력하지 않습니다. 결측 대체·표준화·범주 인코딩은 학습 자료에서만 적합합니다. 라벨 구성 원지표를 학습에 쓰므로 기본 성능은 규칙 재현 실험이며, 독립 확인 정답의 성능은 따로 보고합니다. `holdout_regions`에 시도 모드에서는 시도 코드, 시군구·행정동 모드에서는 시군구 코드를 지정합니다. 해당 지역을 라벨 기준 적합·모델 학습·검증에서 제외하고 테스트에 별도로 평가합니다. 기존 `holdout_districts` 설정도 읽을 수 있지만 두 설정이 충돌하면 중단합니다. 분할 설정을 바꾸면 라벨부터 다시 생성해야 합니다.

준비·평가 보고서의 `evaluation_region_level`, `independent_evaluation_regions`와 모델의 `training_regions`, `holdout_regions`로 지역 단위를 확인할 수 있습니다. 기존 저장 형식과의 호환을 위한 `district_code`, `independent_districts`, `training_districts`, `holdout_districts`는 시도 모드에서 시도 평가 그룹을 뜻합니다.

산출물은 `artifacts/isolation/`에 저장합니다: `structural.parquet`, `structural_quality.json`, `cohorts.parquet`, `cohort_manifest.json`, `readiness.json`, `labels.parquet`, `label_reference.json`, `label_summary.json`, `models/logistic.pkl`, `models/logistic.json`, `evaluation.json`, `detections.parquet`. 원자료로 학습할 수 없는 경우에도 준비 상태와 부족한 근거를 확인할 수 있습니다.
