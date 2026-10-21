# WAAM IC-PBL — 멀티로봇 WAAM 경로계획

INE4107 인공지능과알고리즘 IC-PBL 과제. 로봇 3대가 목표 형상(`target.stl`)을 함께 적층하도록
시간이 지정된 TCP 경로(`trajectory.csv`)를 만든다. 결과는 교수님 채점기
[WAAM_Validator](https://github.com/Ha-An/WAAM_Validator)로 검증한다.

평가 기준은 makespan(세 로봇이 모두 끝나는 시간) 최소화, 계산 시간 최소화, 처음 보는 형상에서의
일반화 성능이다. 공개 모델마다 설정을 따로 맞추지 않고 모든 형상에 같은 알고리즘을 쓴다.

## 현재 결과 — WAAM Validator v1.0.2 기준

v1.0.2 형상 기준: 전체·층별 IoU 98% 이상, 전체 Coverage 98% 이상, 전체 Overfill 3% 이하, 실패 층 0개.

| 모델 | 형상 | makespan | 계산 시간 | IoU | 판정 |
|---|---|---:|---:|---:|---|
| 01 | 벌집 | 918.7 s (15분 19초) | 7.8 s | 0.9973 | PASS |
| 02 | 나선 | 1,404.7 s (23분 25초) | 14.0 s | 0.9966 | PASS |
| 03 | 삼각 트러스 | 1,413.4 s (23분 33초) | 10.4 s | 0.9869 | PASS |
| 04 | 모터 내부 | 28,875.9 s (8시간 1분 16초) | 108.1 s | 0.9996 | PASS |
| 05 | 비틀린 삼각 껍데기 | 75,706.8 s (21시간 1분 47초) | 135.7 s | 0.9990 | PASS |
| 06 | 앵글 브래킷 | 5,315.3 s (1시간 28분 35초) | 6.1 s | 0.9961 | PASS |
| **합계** | | **113,634.9 s (31시간 33분 55초)** | **282.2 s** | | **6/6** |

- 충돌 0건, 속도·대기 위반 없음. 가장 낮은 층의 IoU는 98.68%(03)이다.
- makespan은 채점기(`waam-validator run`)가 계산한 값이다.
- 계산 시간은 형상을 넣은 순간부터 `trajectory.csv`가 나올 때까지의 시계 시간이다. 실행 중 학습 단계는
  없고 채점 시간은 넣지 않았다. 측정 환경: Intel Core Ultra 5 338H(12코어, 병렬 실행), RAM 16 GB,
  Windows 11, 전원 연결. 설정 여러 개와 층별 계산을 코어에 나눠 돌리므로 코어가 적으면 더 길어진다.
- 다시 재려면 `tools/report_public.py`를 실행한다(아래 실행 방법). 기록: `results/report_public_20261021_1701.csv`
- 직접 만든 형상 생성기(`tools/make_jobs_diverse.py`, 21개 계열)로 처음 보는 형상에서도 확인했다.
  이 확인은 v1.0.1 기준(IoU 90%)이고, v1.0.2 기준 재확인은 진행 예정이다.

## 알고리즘

형상 하나를 받으면 아래 순서로 계획을 만든다. 핵심은 **규칙 기반 그리디 스케줄러**이고, 실행 중 학습이나
반복 탐색은 없다.

### 1. 층별 경로 후보 만들기 (`src/toolpath.py`, `src/toolpath_cache.py`)

- 형상을 층 높이(2 mm)로 자른 단면마다 적층 경로 후보를 여러 개 만든다.
  - 동심원: 테두리에서 안쪽으로 고리 간격 7.0·7.5·8.0 mm, 모서리는 둥글게 또는 각지게
  - 나선: 동심원 고리를 이어 한 붓으로 만든 것(조각 수와 이동이 준다)
  - 중심선(medial axis): 비드 한두 줄 두께의 얇은 벽용
  - 위 후보에 남은 빈틈을 중심선이나 짧은 선으로 메운 변형
- 채점기와 같은 방식(중심선을 비드 폭 8 mm로 버퍼해 층 가운데 단면과 비교)으로 후보마다 IoU·Coverage·
  Overfill을 직접 계산한다.
- 작업 config의 형상 기준에 여유를 둔 목표(IoU·Coverage는 기준 +0.2%p, Overfill은 기준 −0.5%p)를
  넘는 후보 가운데 **예상 적층 시간이 가장 짧은 것**을 고른다. 층마다 목표를 넘기므로 전체 IoU와
  Coverage도 목표를 넘는다.
- 층별 계산은 코어에 나눠 한 번만 하고, 아래 4단계의 모든 설정이 나눠 쓴다.

### 2. 조각 나누기와 로봇 배정 (`src/schedule_v8.py`, `src/balance.py`)

- 경로를 정해진 길이(80·160·220 mm) 이하의 조각으로 자른다.
- 로봇 3대가 작업대를 120° 간격으로 둘러서 있으므로, 작업대를 부채꼴 셋으로 나눠 **가까운 로봇**에게
  조각을 준다. 각자 자기 앞만 맡으면 팔(베이스→토치 선분)끼리 엇갈리지 않는다.
- 세 로봇의 일이 비슷해지도록 부채꼴 경계 각도를 조정하고(팔이 닿는 범위 안에서), 떨어진 섬이 여럿인
  층은 섬을 통째로 한 로봇에게 준다.
- 로봇마다 자기 조각을 **가까운 것부터** 줄 세운다(반지름 띠 순서 + 최근접 이웃). 앞 조각 끝과 다음 조각
  시작이 가까우면 이어 긋는다.

### 3. 그리디 배차와 충돌 회피 (`src/schedule_v8.py`)

- 매 순간, 각 로봇 줄 앞쪽의 후보(6개 또는 25개) 가운데 **충돌 없이 가장 빨리 시작할 수 있는 조각**을
  골라 확정한다. 한 번 확정하면 되돌리지 않는다.
- 충돌 판정은 채점기와 같은 규칙이다. 위에서 본 베이스→토치 선분(팔 캡슐) 사이 거리가 250 mm에 여유
  (5 또는 20 mm)를 더한 값 이상이어야 하고, 0.1 s 간격으로 보며, 먼저 끝나 서 있는 로봇도 장애물로 본다.
- 모든 후보가 막히면 길을 막은 로봇이 팔을 접거나 주차 위치로 비켜선다.
- 시간 계산은 채점기 기준을 따른다. 채점기가 검사하지 않는 아크 켜기·끄기 대기는 넣지 않고, 속도는 허용
  오차(±10%) 안에서 최대로 쓴다(적층 8.72 mm/s, 이동 163.5 mm/s).

### 4. 설정 고르기와 병렬 실행 (`src/features.py`, `src/portfolio.py`)

- 로봇 수(1·2·3대), 조각 길이, 경로 종류(동심원·나선)가 다른 설정 몇 개로 1–3단계를 돌려 **makespan이
  가장 짧은 계획**을 낸다.
- 어떤 설정을 돌릴지는 형상 특징으로 정한다(계산 1초 안팎).
  - 두 로봇이 형상 위에 동시에 설 수 있는 조합이 없으면(작은 형상) 한 번에 한 대만 쌓을 수 있으므로
    1대 설정 2개만 돌린다. 한 대로 다 닿지 않으면 2대가 번갈아 쌓는 설정 2개와 3대 설정 1개를 돌린다.
  - 그 밖에는 2대·3대 설정 3개와 1대 설정 1개, 모두 4개를 돌린다.
- 설정들은 코어별로 동시에 돌리고, 이미 다른 설정보다 느려진 설정은 중간에 멈춘다(가지치기).

### 5. 제출기 (`src/export_waam.py`)

결과를 채점기 `run`으로 확인하고, FAIL이면 makespan 순으로 다음 설정의 결과로 바꾼다. 마지막에
`waam-validator check`까지 통과한 작업 폴더(`config.yaml`, `target.stl`, `trajectory.csv`)를 만든다.

### 해 봤지만 채택하지 않은 것

- 강화학습(CEM, PPO)으로 조각 고르기: 큰 형상에서만 조금 좋아지고 작은 형상·처음 보는 형상에서는
  나빠져, CEM으로 찾은 설정값 하나(`r3tuned`)만 설정 후보로 남겼다.
- 충돌 여유 5 → 2 mm, 먼저 끝난 로봇이 남의 조각 넘겨받기: 처음 보는 형상에서 FAIL이 나거나 오히려
  느려졌다.
- 분석해 보면 작은 형상(01·06)은 250 mm 규칙 때문에 두 로봇이 동시에 쌓을 자리가 없어 한 대씩 쌓는 것이
  최선이고, 큰 형상(05)은 세 로봇이 시간의 99% 이상을 적층에 쓴다. makespan은 이미 규칙이 허락하는
  한계에 가깝다.

버전별 변화와 실험 기록은 [CHANGELOG.md](CHANGELOG.md)에 있다.

## 폴더 구조

```text
README.md         이 문서
CHANGELOG.md      버전별 변경 기록
docs/             설계 문서(DESIGN.md), 용어집
data/             시나리오 형상 (git 미추적, 만드는 방법은 data/README.md)
src/              제출에 쓰는 코드 (경로 생성·스케줄러·설정 고르기·제출기)
tools/            평가·측정·형상 생성·분석·그림 스크립트
experiments/      RL·튜닝·예전 버전 스케줄러 (기록용, 그대로는 실행되지 않을 수 있음)
archive/          기능을 바꾸기 전 백업과 옛 실행 스크립트 (기록용)
results/          평가 결과 CSV, results/logs/ 평가 로그
WAAM_Validator/   교수님 채점기 클론 (git 미추적, 가상환경 .venv 포함)
```

주요 코드:

| 파일 | 역할 |
|---|---|
| `src/export_waam.py` | 입구. 제출용 작업 폴더 생성, 채점기 확인(FAIL이면 다음 설정으로 교체) |
| `src/portfolio.py` | 설정 고르기, 코어별 동시 실행, makespan 최소 결과 선택 |
| `src/features.py` | 형상 특징(두 로봇이 동시에 설 수 있는 비율, 한 대로 닿는 비율) |
| `src/toolpath.py` | 층 단면에서 적층 경로 후보 생성과 선택 |
| `src/toolpath_cache.py` | 층별 경로를 코어에 나눠 한 번만 계산 |
| `src/schedule_v8.py` | 조각 배정, 줄 세우기, 그리디 배차, 충돌 회피 |
| `src/balance.py` | 부채꼴 경계 조정, 섬 배분 |
| `src/schedule_v1.py`, `src/schedule_v2.py` | 예전 스케줄러. 로봇 동작 기록·조각 자르기 등 기본 부품을 `schedule_v8.py`가 가져다 쓴다 |
| `tools/report_public.py` | 공개 6개의 makespan·계산 시간·채점기 판정 표 |
| `tools/eval_set.py` | 형상 묶음 전체를 생성·채점해 `results/<이름>.csv`로 저장 |
| `tools/cpu_time.py` | 코어 수와 무관한 총 계산량(CPU 시간) 측정 |

## 실행 방법 (Windows)

처음 한 번 채점기와 가상환경을 이 저장소 안의 `WAAM_Validator/`에 준비한다(Python 3.11 이상).

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
.\WAAM_Validator\.venv\Scripts\python src\export_waam.py --scenario WAAM_Validator\tests\01 --output out\01
```

공개 6개 결과 표(makespan·계산 시간·채점기 판정):

```powershell
.\WAAM_Validator\.venv\Scripts\python tools\report_public.py
```

형상 묶음 평가(생성 + 채점):

```powershell
cd tools
..\WAAM_Validator\.venv\Scripts\python eval_set.py portfolio.py ..\WAAM_Validator\tests pf_public
```

- 코드는 채점기 가상환경이 `WAAM_Validator\.venv\Scripts\python.exe`에 있다고 가정한다(Windows).
  다른 운영체제에서는 `src/portfolio.py`, `src/export_waam.py`, `tools/*.py`의 `VPY` 경로를 바꿔야 한다.
- 계산 시간은 충전기를 연결하고 다른 파이썬 작업이 없을 때 잰다. 배터리 모드나 절전 상태에서는 크게 늘어난다.
