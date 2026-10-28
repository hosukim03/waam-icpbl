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
| `check808/` | 21 | 21개 계열 확인용(연습용, 결과 확인함) | `python tools/make_jobs_diverse.py data/check808 21 808` |
| `fresh1027/` | 21 | 처음 보는 형상(시드 1027, 21개 계열 하나씩). 80 mm 조각 후보를 넣을지 확인하는 데 씀 | `python tools/make_jobs_diverse.py data/fresh1027 21 1027` |
| `novel1027/` | 21 | **21개 계열과 다른 방식으로 만든 확인용(2026-10-27).** 무작위 윤곽, 연속값 벽 두께, 높이에 따라 층마다 조금씩 변함. 첫 측정에서 떨어진 형상을 보고 경로 점수·가는 틈 문제를 고쳤으므로 이제 본 묶음이다(다음 확인은 새 시드로) | `python tools/make_jobs_novel.py data/novel1027 21 1027` |
| `final_test/` | 20 | 옛 15개 계열로 만든 봉인 묶음. `final_test21`로 대체되어 쓰지 않는다(삭제 대기) | `python tools/make_jobs_diverse.py data/final_test 20 707` |
| `final_test21/` | 21 | **마지막 테스트용 묶음(2026-09-30 생성, 21개 계열 하나씩).** 결과를 보고 설정을 고치는 데 쓰지 않고, 최종 버전에서 한 번만 평가한다 | `python tools/make_jobs_diverse.py data/final_test21 21 909` |

`train_div`부터 `check808`까지는 설정을 고르거나 버그를 찾을 때 결과를 이미 봤으므로 연습용이다.
처음 보는 형상에서의 성능은 `final_test21`로만 판단한다.
다만 `final_test21`과 `fresh1027`은 같은 생성기라 종류는 그대로이고 크기·두께·층수만 바뀐 확인이다(벽 두께도
8·16·24 mm, 즉 비드 1·2·3줄에 딱 맞는 값만 나온다). 21개 계열에 없는 종류는 `novel1027`로 확인한다.

형상 생성기(`tools/make_jobs_diverse.py`)의 계열은 21개다: 기존 15개(막대, 원형 벽, 사각 벽, 속찬 덩어리, 물결선,
벌집, 사각 격자, 삼각 격자, 나선, 삼각 트러스, 떨어진 섬, 별 모양 벽, 판+기둥, 비틀린 꽃병, L자 브래킷)와
2026-09-30 추가한 6개(톱니바퀴, 임펠러, 글자, 플랜지, 슬롯 판, 위로 갈수록 넓어짐/좁아짐).
