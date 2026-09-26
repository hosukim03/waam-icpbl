# WAAM IC-PBL — 멀티로봇 WAAM 경로계획

INE4107 인공지능과알고리즘 IC-PBL 과제. 로봇 3대가 목표 형상(`target.stl`)을 함께 적층하도록
시간이 지정된 TCP 경로(`trajectory.csv`)를 만든다. 결과는 교수님 채점기
[WAAM_Validator](https://github.com/Ha-An/WAAM_Validator)로 검증한다.

평가 기준은 makespan(세 로봇이 모두 끝나는 시간) 최소화, 생성 시간 최소화, 처음 보는 형상에서의
일반화 성능이다. 공개 모델마다 설정을 따로 맞추지 않고 모든 형상에 같은 알고리즘을 쓴다.

## 현재 결과 (v0.01, 알고리즘 `pf_sw`)

| 묶음 | 형상 수 | 판정 | makespan 합계 | 생성 시간 합계 |
|---|---:|---|---:|---:|
| 공개 모델 | 6 | 전부 PASS | 124,380 s (34.55 h) | 305 s |
| 검증(`valid_div`) | 20 | 전부 PASS | 134,275 s | 441 s |
| 벤치마크 | 10 | 전부 PASS | 14,574 s | 136 s |
| 새 형상 시드 303 | 15 | 전부 PASS | 114,705 s | 444 s |
| 새 형상 시드 404 | 15 | 전부 PASS | 136,198 s | 469 s |
| 새 형상 시드 505 | 15 | 전부 PASS | 188,236 s | 536 s |

충돌은 모두 0건이다. 버전별 변화는 [CHANGELOG.md](CHANGELOG.md)에 있다.

## 폴더 구조

```text
README.md         이 문서
CHANGELOG.md      버전별 변경 기록
docs/             설계 문서(DESIGN.md), 용어집
data/             시나리오 형상 (git 미추적, 만드는 방법은 data/README.md)
v0/               알고리즘 코드 (경로 생성·스케줄러·포트폴리오·평가 스크립트)
results/          평가 결과 CSV, results/logs/ 평가 로그
WAAM_Validator/   교수님 채점기 클론 (git 미추적, 가상환경 .venv 포함)
```

주요 코드:

| 파일 | 역할 |
|---|---|
| `v0/toolpath.py` | 층 단면에서 적층 경로(동심원·나선·중심선) 생성 |
| `v0/schedule_v8.py` | 경로를 조각으로 나눠 로봇에 배정하고, 채점기와 같은 규칙으로 충돌을 피하며 시각 부여 |
| `v0/balance.py` | 부하 균형 배정·섬 배분 |
| `v0/portfolio.py` | 설정 12개를 코어별로 동시에 돌려 makespan이 가장 짧은 결과 선택 |
| `v0/export_waam.py` | 제출용 작업 폴더 생성 + 채점기 검증(FAIL이면 다음 설정으로 교체) |
| `v0/eval_set.py` | 형상 묶음 전체를 생성·채점해 `results/<이름>.csv`로 저장 |

## 실행 방법 (Windows)

처음 한 번 채점기와 가상환경을 준비한다.

```powershell
git clone https://github.com/Ha-An/WAAM_Validator.git
cd WAAM_Validator
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.lock
.\.venv\Scripts\python -m pip install -e . --no-deps
cd ..
```

제출물 만들기(공개 모델 01번 예):

```powershell
.\WAAM_Validator\.venv\Scripts\python v0\export_waam.py --scenario WAAM_Validator\tests\01 --output out\01
```

형상 묶음 평가(생성 + 채점):

```powershell
cd v0
python eval_set.py portfolio.py ..\WAAM_Validator\tests pf_public
python eval_set.py portfolio.py ..\data\valid_div pf_valid
```

생성 시간은 충전기를 연결하고 다른 파이썬 작업이 없을 때 잰다(배터리 모드에서는 약 2.5배 느려진다).
