# data — 시나리오 형상

각 하위 폴더의 번호 폴더(`01`, `02`, …)가 작업 하나이고, 안에 `config.yaml`과 `target.stl`이 있다.
용량이 커서 git으로는 이 설명 파일만 추적하고, 형상은 아래 명령으로 다시 만든다.

| 폴더 | 개수 | 용도 | 만드는 방법 |
|---|---:|---|---|
| `jobs/` | 6 | 공개 모델 01~06 작업 폴더(초기 실행용) | `WAAM_Validator/tests/01~06` 복사 |
| `bench/` | 10 | 채점기 일관성 확인용 벤치마크(알고리즘 비교용 아님). `bench/jobs_ours`가 원본 형상 | `WAAM_Validator/scripts/generate_benchmark_jobs.py` |
| `train_div/` | 30 | 설정 튜닝·학습용 | `python tools/make_jobs_diverse.py data/train_div 30 101` |
| `valid_div/` | 20 | 설정 선택용 검증 | `python tools/make_jobs_diverse.py data/valid_div 20 202` |
| `test_div/` | 15 | 처음 보는 형상(시드 303) 일반화 확인 | `python tools/make_jobs_diverse.py data/test_div 15 303` |
| `holdout404/` | 15 | 처음 보는 형상(시드 404) 일반화 확인 | `python tools/make_jobs_diverse.py data/holdout404 15 404` |
| `holdout505/` | 15 | 처음 보는 형상(시드 505) 일반화 확인 | `python tools/make_jobs_diverse.py data/holdout505 15 505` |
| `final_test/` | 20 | **마지막 테스트용 봉인 묶음(2026-09-29 생성).** 결과를 보고 설정을 고치는 데 쓰지 않고, 최종 버전에서 한 번만 평가한다 | `python tools/make_jobs_diverse.py data/final_test 20 707` |

`train_div`부터 `holdout505`까지 81개는 설정을 고를 때 결과를 이미 봤으므로 연습용이다.
처음 보는 형상에서의 성능은 `final_test`로만 판단한다.

공개 모델 6개 평가는 교수님 저장소의 `WAAM_Validator/tests`를 그대로 쓴다.
