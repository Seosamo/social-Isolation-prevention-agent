# 운영 모델

현재 기본 선택은 `experiments/20261021T110042016328Z/IF_F2.pkl`입니다. `config/default.json`의 `isolation.model_id=IF_F2`와 `isolation.model_path`로 연결하며, 입력 8개와 저장 정규화·보정 임계값을 유지합니다. 실행법은 `docs/f2-deployment.md`를 참고하세요.

`isolation_forest.pkl`은 서울·강원 활동 위축 이상탐지 Baseline 모델이며,
`isolation_forest.json`에는 입력 특징·정규화·임계값·학습 범위를 저장합니다.
Baseline을 선택한 학습과 추론은 프로젝트 루트 기준 이 폴더를 사용합니다. 두 파일은 Git에 포함합니다.

실험 모델은 `activity_agent/model/experiments/<run_id>/`에 보관합니다.
파일명은 결과표와 같은 실험 ID를 사용합니다(예: `IF_F0.pkl`, `IF_F4.pkl`,
`LOF_F4.pkl`, `KNN_F4.pkl`, `ROBUST_Z_F4.pkl`, `DUAL_IF.pkl`).
`IF_BASELINE_ORIGINAL.pkl`은 운영 원본, `IF_BASELINE.pkl`은 비교용 래퍼입니다.
실험 결과의 `config/model_directory.json`이 모델 폴더의 상대 경로를 기록합니다.
실험 모델을 운영 모델로 자동 교체하지 않습니다.

IF_F2의 소량 유동 제외 기준은 같은 실험 모델 폴더의 `flow_cutoffs.csv`에 포함합니다. 원본 실험 자료로 검증한 기준이며 `IF_F2.json`에 SHA-256, `flow_cutoffs_provenance.json`에 검증 내역을 기록합니다. 임시 출력 폴더의 파일이나 임의 기준으로 대체하지 않습니다. 새 실험도 기준표를 모델 폴더에 함께 저장합니다.
