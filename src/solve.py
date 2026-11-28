# -*- coding: utf-8 -*-
"""코어 1개용 풀이 — 형상을 한 번만 자르고, 층별 경로를 한 번만 만들고, 설정 후보들을 코어 하나로
차례로 계획해 makespan 이 가장 짧은 것을 낸다. 모든 계산을 한 프로세스에서 한다.

예전 portfolio.py 는 설정 2~4개를 코어별로 동시에 끝까지 돌려 가장 짧은 것을 골랐다. 코어가 많으면 시계
시간은 짧지만 계산량은 설정 수만큼 들고, 코어가 모자라면 설정을 줄여 makespan 까지 나빠졌다(코어 2개 +5.7%).
여기서는 코어 수와 상관없이 같은 계산을 같은 순서로 하므로 어느 컴퓨터에서나 결과가 같고, 고르는 답도
portfolio.py 와 같다. 계산을 줄이는 방법은 세 가지다.
  - 프로그램을 설정마다 따로 띄우지 않는다(형상 읽기·층 자르기·경로 계산을 한 번만)
  - 후보 순서: 층 4개로 짧게 시험해 빨리 끝날 것 같은 후보부터 끝까지 계획하고, 시험에서 10% 넘게 느린
    후보는 끝까지 계획하지 않는다
  - 가지치기: 앞에서 끝난 후보보다 이미 느려진 후보는 그 자리에서 멈춘다

순서
  1) 층 단면 자르기(한 번)
  2) 형상 특징으로 설정 후보 고르기(features.py, portfolio.py 의 SOLO·TURN·MULTI 와 같다)
  3) 후보들이 쓰는 경로 묶음(나선 포함·고리만)을 층마다 한 번 계산
  4) 후보를 차례로 끝까지 계획(best_of_all)하고 가장 짧은 결과로 trajectory.csv 를 쓴다
  5) 고른 설정이 부채꼴 배정을 쓰면 국소 탐색으로 한 번 더 계획하고, 더 짧으면 그것을 쓴다
  4)·5)는 계산량 한도(BUDGET, 충돌 검사 횟수) 안에서만 한다(아래)

usage: python solve.py <job_dir> <out_dir> [--eval] [--ls N] [--budget N] [--chunk N]   (WAAM venv 파이썬)
  --chunk N  (시험용) 2·3대 계획의 조각 길이를 N mm 로(원래 길이는 첫 후보 안전장치 길이, 아래 계산량 한도)
  --eval  분석용: 후보를 전부 끝까지 계획해 짧은 시험의 예상과 실제 makespan 을 함께 적는다(시간이 더 든다)
  --ls N  국소 탐색: 층마다 가장 늦게 끝난 로봇의 부채꼴 경계를 옮겨 최대 N 번 다시 계획해 본다(기본 2, 0=안 함)
  --budget N  계산량 한도(충돌 검사 횟수, 기본 BUDGET, 0=한도 없음)

국소 탐색 횟수(2026-10-28, 여러 로봇이 함께 쌓는 형상 7개·층 297개 합계, 코어 1개 시간은 공개 6개):
  층마다 1번 makespan -1.7%·계산 +79 s, 2번 -2.3%·+113 s, 4번 -2.6%·+164 s. 2번이 이득 대부분을 얻으면서
  4번보다 계산이 약 30% 덜 들어 기본으로 둔다. 로봇 1대 설정은 옮길 경계가 없어 계산도 결과도 그대로다.

계산량 한도(2026-11-25): 최종 테스트는 코랩 무료 CPU 에서 형상마다 5분 안이어야 한다. 코랩은 노트북보다
계획·국소 탐색이 1.9배 느려(경로 계산은 거의 같다) 공개 04 가 342 s 였다. 시간으로 끊으면 컴퓨터마다 답이
달라지므로, 대신 충돌 검사 횟수를 센다(어느 컴퓨터에서나 같은 값). 짧은 시험·끝까지 계획·국소 탐색을 합쳐
BUDGET 번까지만 쓴다(첫 후보 안전장치(2026-11-29): 설정에 --chunk-safe 가 있으면 첫 후보가 끝까지 갔을 때의 검사
횟수가 남은 몫을 넘을 것 같을 때 남은 층을 그 길이로 짠다. 지금 후보 설정에는 없고 --chunk 로 시험할 때만 켜진다):
첫 후보는 계획이 하나는 있어야 하므로 늘 끝까지, 그다음 후보는 남은 만큼만(넘으면 멈추고
안 고름), 국소 탐색은 고른 설정을 다시 계획하는 몫(cost[best])을 뺀 나머지를 층마다 나눠 쓴다. 한도 안에 드는
형상은 결과가 그대로다(공개 01·02·03·06 궤적 바이트 단위로 같음).
  코랩에서 충돌 검사 1번 ≈ 0.64 ms(04: 계획+국소 탐색 296 s / 462,565번), 32만 번 ≈ 205 s.
  무거운 형상 4개(한도 없음 → 32만 / 35만, makespan 변화, 코랩 예상 시간은 같은 때 함께 돌린 04 와의 비율로 환산):
    공개 04                28,048 s(코랩 342 s 실측) → +2.2%(약 250 s) / +1.1%(약 267 s)
    연습 1 21번            32,227 s(약 316 s)        → +2.0%(약 255 s) / +0.9%(약 268 s)
    fresh1027 03번         44,148 s(약 355 s)        → +2.1%(약 158 s, 국소 탐색 못 함) / +2.0%(약 236 s)
    연습 2 20번            25,586 s(약 307 s)        → +0.0%(약 260 s) / +0.0%(약 283 s, 경로 계산이 길어 5분에 가깝다)
  35만은 연습 2 20번이 5분에 너무 가까워 32만으로 둔다. 모두 PASS·충돌 0.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).parent))

import features  # noqa: E402
import schedule_v8 as s8  # noqa: E402
import toolpath  # noqa: E402
from portfolio import MULTI, PAIR_EPS, PRIORITY, SOLO, TURN, VARIANTS  # noqa: E402

SAMPLE_ALL = 8      # 층이 이 수 이하면 짧게 시험하지 않고 후보를 전부 끝까지 계획한다
SAMPLE = 4          # 그보다 많으면 높이 방향으로 고르게 이만큼의 층으로 짧게 시험해 순서를 정한다
KEEP = 0.10         # 짧은 시험에서 가장 빠른 후보보다 이 비율 넘게 느린 후보는 끝까지 계획하지 않는다
SAMPLE2_FRAC = 0.05  # 남은 후보가 둘 이상이면 2차로 전체 층의 이 비율(최소 SAMPLE2_MIN 층)을 더 시험한다
SAMPLE2_MIN = 8     # (05 는 350층 중 106층에서만 설정 차이가 나서 1차 4층으로는 셋이 똑같이 나왔다)
BUDGET = 320_000    # 계산량 한도(충돌 검사 횟수, 0=한도 없음): 후보 계획과 국소 탐색을 합쳐 이만큼만 쓴다(위 설명)
CHUNK = None        # (--chunk, 시험용) 2·3대 계획의 조각 길이(mm). 원래 길이는 첫 후보 안전장치 길이(--chunk-safe)로 쓴다
# 시간 안전장치(2026-11-29): 코랩에서 형상 하나당 5분(프로그램 시작·파일 쓰기 포함). 이 프로그램이 시작한 뒤 DEADLINE 초
# 안에 계획을 끝낸다. 두 번째 후보부터는 남은 시간으로 짤 수 있는 충돌 검사 횟수(지금까지 잰 1번당 시간 × TIME_SAFETY)
# 안에서만 짜고, 그래도 넘을 것 같으면 멈춘다. 첫 후보는 남은 층을 더 빨리 짜는 설정으로 바꾼다(schedule_v8.TIME_RELAX).
# 국소 탐색은 고른 계획을 다시 짜는 데 든 시간만큼 남아 있을 때만 하고, 넘을 것 같으면 멈추고 다듬기 전 계획을 쓴다.
# 빠른 컴퓨터에서는 마감에 닿지 않아 결과가 그대로다. 이 컴퓨터에서 여러 개를 동시에 돌리는 실험은 --deadline 0 으로 끈다.
DEADLINE = 260.0
TIME_SAFETY = 0.85


def candidates(f: dict) -> list[str]:
    """형상 특징으로 후보 설정을 고른다(portfolio.select_variants 와 같은 규칙)."""
    if f["pair_max"] < PAIR_EPS:
        return list(SOLO if f["reach1"] >= 0.999 else TURN)
    return list(MULTI)


REUSE_DIFF = 0.005  # 아래층과 단면 차이(대칭차 면적 ÷ 면적)가 이보다 작으면 아래층 경로를 다시 써 본다
# 경로 만들 때만 단면 단순화(2026-11-29): 곡면을 잘게 나눈 STL 은 층 단면 꼭짓점이 수천 개라(연습 4 01 평균 8,132개, 19
# 5,045개) 안쪽으로 줄이는 계산(buffer)이 경로 시간의 대부분이다. 꼭짓점이 SIMPLIFY_MIN_VERTS 보다 많은 층만 SIMPLIFY_TOL
# 로 단순화한 단면으로 경로를 고른다(0.05 mm 면 꼭짓점이 절반, 넓이 차이 0.005% 이하. 비드 폭 8 mm 에 비해 아주 작다).
# 층 8개씩 잰 경로 시간: 연습 4 01 −51%, 19 −27%, 연습 3 09 −22%(원래 단면으로 채점해도 모두 목표 통과, 쌓는 시간 ±0.1%).
# 꼭짓점이 2,000개 넘는 형상 21개를 끝까지 풀면 21개 모두 PASS, 경로 시간 −73%(대략), makespan 합계 +0.21% 였지만 형상마다
# 흔들림이 컸다(공개 02 +5.3%, 연습 1 13 +13.6%, 연습 3 17 −5.0%). 경로가 조금만 달라져도 계획이 달라지기 때문이다. 그래서
# 기본으로는 끄고(--simplify 0.05 로 켬), 경로 만들기가 마감에 가까울 때만 쓰는 방법을 따로 정한다.
SIMPLIFY_TOL = 0.0
SIMPLIFY_MIN_VERTS = 2000


def for_paths(A):
    """경로를 고를 단면: 꼭짓점이 많으면 단순화한 단면(올바르지 않거나 비면 원래 단면)."""
    if SIMPLIFY_TOL <= 0 or shapely.get_num_coordinates(A) <= SIMPLIFY_MIN_VERTS:
        return A
    G = A.simplify(SIMPLIFY_TOL, preserve_topology=True)
    return G if (not G.is_empty and G.is_valid) else A


def build_paths(layers: dict, n: int, need: set[str], reuse: bool = False) -> dict:
    """{'sp': {층: (경로, IoU)}, 'ring': {...}}. toolpath_cache.py 를 코어 하나로 돌린 것과 같다.
    reuse: 아래층과 단면이 거의 같고(REUSE_DIFF) 아래층 경로가 이 층의 형상 목표도 통과하면 그대로 쓴다
    (toolpath_cache 의 --reuse 와 같은 규칙, 점수 재기는 경로 고르기보다 수십 배 싸다).
    경로는 for_paths 단면(꼭짓점이 많으면 단순화)으로 고른다."""
    out = {t: {} for t in need}
    prev, prevA = None, None                        # prev = {묶음: 경로}
    for k in range(n):
        A = for_paths(layers[k])
        if A.is_empty:
            continue
        got = {}
        if reuse and prev is not None and A.symmetric_difference(prevA).area / max(A.area, 1e-9) < REUSE_DIFF:
            for t in need:
                sc = toolpath.score(A, prev[t])
                if toolpath.meets(sc):
                    got[t] = (prev[t], sc["iou"])
        todo = need - set(got)
        if todo == {"sp", "ring"}:                  # 두 묶음을 한 번에(고리 후보를 두 번 평가하지 않음)
            (p1, s1, _), (p2, s2, _) = toolpath.best_paths(A, spiral=True, both=True)
            got["sp"], got["ring"] = (p1, s1["iou"]), (p2, s2["iou"])
        else:
            for t in todo:
                p, sc, _ = toolpath.best_paths(A, spiral=(t == "sp"))
                got[t] = (p, sc["iou"])
        for t in need:
            out[t][k] = got[t]
        prev, prevA = {t: got[t][0] for t in need}, A
    return out


def sample_layers(layers: dict, n: int, m: int = SAMPLE, skip: set[int] = frozenset()) -> set[int] | None:
    """짧은 시험에 쓸 층(skip 을 뺀 층에서 높이 방향으로 고르게 m 개). None 이면 전부."""
    ks = [k for k in range(n) if not layers[k].is_empty and k not in skip]
    if not skip and len(ks) <= max(SAMPLE_ALL, m):
        return None
    if len(ks) <= m:
        return set(ks)
    return {ks[i] for i in np.linspace(0, len(ks) - 1, m).round().astype(int)}


def race(names: list[str], trial, layers: dict, n: int, m1: int = SAMPLE) -> tuple[str, dict, list]:
    """(분석용, 쓰지 않음) 후보를 짧게 시험만 해서 하나를 고른다. trial(이름, 층 집합 또는 None) → makespan.
    1차: 고르게 뽑은 m1 층으로 전부 시험, 가장 빠른 것보다 KEEP 넘게 느린 후보는 뺀다.
    2차: 남은 후보가 둘 이상이면 다른 층을 더 시험해 1·2차 합으로 고른다.
    공개 6개 + 연습 1(21개)에서 후보 중 최선보다 +0.56%(7개 형상에서 놓침, 05 +1.14%): 층을 떼어 시험하면
    층이 이어질 때 생기는 차이(앞 층이 끝난 자리에서 다음 층을 시작하는 데서 오는 차이)를 못 본다."""
    rank = {nm: (PRIORITY.index(nm) if nm in PRIORITY else len(PRIORITY), i) for i, nm in enumerate(names)}
    only1 = sample_layers(layers, n, m1)
    total = {nm: trial(nm, only1) for nm in names}
    rounds = [only1]
    if only1 is not None:
        lo = min(total.values())
        keep = [nm for nm in names if total[nm] <= lo * (1 + KEEP)]
        if len(keep) > 1:
            m2 = max(SAMPLE2_MIN, int(np.ceil(SAMPLE2_FRAC * n)))
            only2 = sample_layers(layers, n, m2, skip=only1)
            if only2:
                rounds.append(only2)
                for nm in keep:
                    total[nm] += trial(nm, only2)
        names = keep
    best = min(names, key=lambda nm: (round(total[nm], 6), rank[nm]))
    return best, total, rounds


def best_of_all(names: list[str], trial, plan, layers: dict, n: int, m1: int = SAMPLE):
    """후보를 끝까지 계획해 makespan 이 가장 짧은 것을 고른다(같으면 목록 앞쪽, portfolio.py 와 같은 규칙).
    코어 하나로 차례로 돌리므로 순서가 계산 시간을 좌우한다. 고르게 뽑은 층 m1 개로 짧게 시험해 빨리 끝날
    것 같은 후보부터 계획하고(시험에서 KEEP 넘게 느린 후보는 계획하지 않는다), 앞에서 끝난 후보보다 이미
    느려진 후보는 그 자리에서 멈춘다(가지치기 규칙은 portfolio.py 와 같다: 로봇 시각이 최선을 넘었거나,
    20% 이상 진행해 예상 makespan 이 최선의 1.15배를 넘음).
    trial(이름, 층 집합) → 시험 makespan, plan(이름, bound) → (makespan, 로봇들) 또는 멈췄으면 None.
    돌려주는 값: 고른 이름, (makespan, 로봇들), 시험 값, 끝까지 계획한 후보의 makespan, 시험 층."""
    rank = {nm: i for i, nm in enumerate(names)}
    only1 = sample_layers(layers, n, m1)
    trials = {}
    order = list(names)
    if only1 is not None and len(names) > 1:
        trials = {nm: trial(nm, only1) for nm in names}
        prio = {nm: (PRIORITY.index(nm) if nm in PRIORITY else len(PRIORITY)) for nm in names}
        # 짧은 시험에서 가장 빠른 후보보다 KEEP 넘게 느린 후보는 끝까지 계획하지 않는다
        # (공개 6 + 연습 1(21개)에서 끝까지 계획해 가장 짧았던 후보는 모두 이 안에 있었다)
        lo = min(trials.values())
        order = sorted((nm for nm in names if trials[nm] <= lo * (1 + KEEP)),
                       key=lambda nm: (round(trials[nm], 6), prio[nm], rank[nm]))
    done, best, kept = {}, None, None
    for i, nm in enumerate(order):
        # 첫 후보 안전장치: 아직 끝난 계획이 없는데 마지막 후보면, 계획이 하나는 있어야 하므로 멈추지 않고 끝까지 짠다
        r = plan(nm, None if best is None else done[best], best is None and i == len(order) - 1)
        if r is None:
            continue
        done[nm] = r[0]
        if best is None or (round(r[0], 6), rank[nm]) < (round(done[best], 6), rank[best]):
            best, kept = nm, r               # 로봇 궤적은 지금까지 최선 하나만 들고 있는다
    return best, kept, trials, done, only1


def uses_sectors(name: str) -> bool:
    """이 설정이 부채꼴 배정(SectorLabeler)을 쓰는가 = 국소 탐색으로 경계를 옮겨 볼 수 있는가.
    1대 설정(한 대로 다 닿는 층)과 --no-balance(작업대 중심 고정 부채꼴)는 옮길 경계가 없다."""
    args = VARIANTS[name][1]
    one = any(a == "--robots" and b == "1" for a, b in zip(args, args[1:]))
    return "--no-balance" not in args and not one


def options(name: str, job: Path, out: Path):
    o = s8.parser().parse_args([str(job), str(out), *VARIANTS[name][1]])
    if CHUNK and o.robots != 1:
        # (--chunk) 2·3대 계획만 조각 길이를 바꾼다(1대는 막힐 일이 거의 없어 그대로). 원래 길이가 더 길면 첫 후보
        # 안전장치 길이로 둔다: 충돌 검사가 남은 몫을 넘을 것 같으면 남은 층은 원래 길이로 짠다
        orig, o.chunk = float(o.chunk), float(CHUNK)
        if orig > o.chunk:
            o.chunk_safe = orig
    return o


def main() -> int:
    global CHUNK, SIMPLIFY_TOL
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="거의 같은 아래층의 경로를 다시 쓴다(경로 계산 단축)")
    ap.add_argument("--slow-voronoi", action="store_true", help="중심선 Voronoi 를 예전 방식으로(비교용)")
    ap.add_argument("--sample", type=int, default=SAMPLE, help="짧은 시험에 쓸 층 수")
    ap.add_argument("--variants", default=None, help="후보 설정을 직접 지정(쉼표로 구분). 생략하면 형상 특징으로 고른다")
    ap.add_argument("--ls", type=int, default=2,
                    help="국소 탐색: 고른 설정을 층마다 최대 이만큼 다시 계획해 보며 한 번 더 계획한다(0=안 함)")
    ap.add_argument("--ls-warm", dest="ls_warm", action="store_true",
                    help="국소 탐색에서 아래층이 받아들인 경계 이동을 먼저 시험한다")
    ap.add_argument("--budget", type=int, default=BUDGET,
                    help="계산량 한도(충돌 검사 횟수): 후보 계획과 국소 탐색을 합쳐 이만큼만 쓴다. 국소 탐색은 남은 "
                         f"몫만 쓴다(기본 {BUDGET:,}, 0=한도 없음)")
    ap.add_argument("--extra", default=None,
                    help="(실험용) 두 대 이상이 동시에 쌓을 수 있는 형상의 후보 끝에 이 설정들을 더한다(쉼표로 구분)")
    ap.add_argument("--chunk", type=float, default=None,
                    help="(시험용) 2·3대 계획의 조각 길이(mm). 원래 길이(2대 160·3대 220)는 첫 후보 안전장치 길이로 쓴다")
    ap.add_argument("--deadline", type=float, default=DEADLINE,
                    help=f"시간 안전장치: 이 프로그램이 시작한 뒤 이 초 안에 계획을 끝낸다(기본 {DEADLINE:g}, 0=끔. "
                         "이 컴퓨터에서 여러 개를 동시에 돌리는 실험은 0)")
    ap.add_argument("--simplify", type=float, default=SIMPLIFY_TOL,
                    help=f"(시험용) 경로 만들 때 꼭짓점이 {SIMPLIFY_MIN_VERTS:,}개 넘는 단면을 이 길이(mm)로 단순화(기본 0 = 끔)")
    a = ap.parse_args()
    CHUNK = a.chunk
    SIMPLIFY_TOL = a.simplify
    toolpath.FAST_VORONOI = not a.slow_voronoi
    job, out = Path(a.job).resolve(), Path(a.out).resolve()
    T = {}
    t0 = time.perf_counter()

    pre = s8.prepare(job)
    cfg, n, layers = pre
    s8.apply_config(cfg, False)                     # 비드 폭·속도·형상 목표를 config 로 맞춘다
    T["slice"] = time.perf_counter() - t0

    t = time.perf_counter()
    feat = features.shape_features_from(cfg, n, layers)
    names = [v for v in a.variants.split(",") if v in VARIANTS] if a.variants else candidates(feat)
    if a.extra and not a.variants and names != SOLO:
        names += [v for v in a.extra.split(",") if v in VARIANTS and v not in names]
    T["features"] = time.perf_counter() - t

    t = time.perf_counter()
    caches = build_paths(layers, n, {VARIANTS[nm][2] for nm in names}, reuse=a.reuse)
    T["paths"] = time.perf_counter() - t

    t = time.perf_counter()

    def trial_run(nm, only):
        return s8.run(job, options(nm, job, out), pre=pre, path_cache=caches[VARIANTS[nm][2]], only=only)[1]

    cost = {}                                       # 후보별로 끝까지 계획하는 데 든 충돌 검사 횟수
    switched = {}                                   # 첫 후보 안전장치로 남은 층을 긴 조각으로 바꾼 층(후보별)
    stopped = {}                                    # 첫 후보 안전장치로 멈추고 다음 후보로 넘어간 까닭(후보별)
    late, relaxed, tcost = {}, {}, {}               # 시간 안전장치: 멈춘 까닭, 첫 후보 설정 바꿈, 후보별 계획 시간
    dl = t0 + a.deadline if a.deadline > 0 else None   # 시간 안전장치 마감 시각

    def plan(nm, bound, must_finish=False):
        c0 = s8.CHECKS
        tp = time.perf_counter()
        cap = soft = None
        o = options(nm, job, out)
        if a.budget > 0 and cost:
            # 계산량 한도: 첫 후보는 계획이 하나는 있어야 하므로 끝까지 계획하고, 그다음 후보부터는 한도에서
            # 남은 만큼만 계획한다(넘으면 가지치기처럼 멈추고 고르지 않는다)
            cap = a.budget - (c0 - c_plan)
            if cap <= 0:
                return None
        elif a.budget > 0:
            # 첫 후보 안전장치: 끝까지 짜되, 남은 몫을 넘을 것 같으면 남은 층은 그 설정의 --chunk-safe 길이로 짠다.
            # 그래도 크게 넘을 것 같으면 멈추고 다음 후보로 넘어간다(must_finish = 마지막 후보면 끝까지)
            soft = a.budget - (c0 - c_plan)
            if soft <= 0 and not must_finish and getattr(o, "chunk_safe", None):
                stopped[nm] = "남은 몫 없음"
                return None
        if dl is not None and cost:
            # 시간 안전장치(두 번째 후보부터): 남은 시간으로 짤 수 있는 횟수(지금까지 계획에 든 시간 ÷ 충돌 검사 횟수로
            # 1번당 시간을 잼) 안에서만 짠다
            per = (tp - t_plan) / max(c0 - c_plan, 2000)
            cap_t = (dl - tp) / per * TIME_SAFETY
            cap = cap_t if cap is None else min(cap, cap_t)
            if cap <= 0:
                late[nm] = "남은 시간 없음"
                return None
        try:
            robots, mk, *_ = s8.run(job, o, pre=pre, path_cache=caches[VARIANTS[nm][2]],
                                    bound=bound, work_cap=cap, soft_cap=soft, soft_abort=not must_finish,
                                    deadline=dl, time_mode="stop" if cost else "relax")
        except s8.OverBudget as ex:
            stopped[nm] = str(ex)
            return None
        except s8.OverTime as ex:
            late[nm] = str(ex)
            return None
        except s8.Pruned:
            return None
        cost[nm] = s8.CHECKS - c0
        tcost[nm] = time.perf_counter() - tp
        if s8.SWITCHED_AT is not None:
            switched[nm] = s8.SWITCHED_AT
        if s8.TIME_EVENTS:
            relaxed[nm] = list(s8.TIME_EVENTS)
        return mk, robots

    c_plan, t_plan = s8.CHECKS, time.perf_counter()
    best, (mk, robots), trial, full, only = best_of_all(names, trial_run, plan, layers, n, a.sample)
    T["plan"] = time.perf_counter() - t
    used = s8.CHECKS - c_plan                       # 짧은 시험 + 끝까지 계획(멈춘 후보 포함)에 쓴 충돌 검사 횟수
    ls_line, ls_checks = None, 0
    if a.ls > 0 and uses_sectors(best):
        # 국소 탐색은 층마다 다시 계획하는 만큼 시간이 들어 고른 설정 하나에만 쓴다. 층 하나씩 보고 고르므로
        # 다음 층에 미치는 영향 때문에 전체가 길어질 수도 있어, 더 짧을 때만 바꾼다.
        # 계산량 한도: 국소 탐색은 고른 설정을 처음부터 한 번 더 계획하므로(그 몫 = cost[best]) 그것과 앞에서 쓴
        # 횟수를 한도에서 빼고, 남은 만큼만 다시 계획해 본다. 남은 몫은 아래층부터 쓰지 않고 층마다 나눠 준다
        # (층 몫 = 그 층 기준 계획에 든 검사 횟수 × 남은 몫 / cost[best], 안 쓴 몫은 위층으로 넘어감). 04 에서
        # 같은 횟수일 때 아래층부터 쓰는 것보다 늘 더 줄었다(한도 25·50·75% 에서 −0.56·−1.96·−2.01% 대
        # −0.17·−1.22·−1.52%). 남은 것이 없으면 국소 탐색을 하지 않는다.
        left = None if a.budget <= 0 else a.budget - used - cost[best]
        if left is not None and left <= 0:
            ls_line = f"ls\tskip budget={a.budget} plan={used} base={cost[best]}"
        elif dl is not None and (best in relaxed or time.perf_counter() + 1.1 * tcost.get(best, 0.0) > dl):
            # 시간 안전장치: 고른 계획을 다시 짜는 시간(처음 짤 때 든 시간)만큼도 남지 않았으면 국소 탐색을 하지 않는다
            ls_line = f"ls\tskip time left={dl - time.perf_counter():.0f}s base={tcost.get(best, 0.0):.0f}s"
        else:
            t = time.perf_counter()
            o = options(best, job, out)
            o.ls, o.ls_warm, o.ls_budget = a.ls, a.ls_warm, left
            o.switch_at = switched.get(best)        # 안전장치로 바꾼 계획이면 같은 층부터 긴 조각으로(같은 기준 계획)
            o.ls_rate = None if left is None else left / max(cost[best], 1)
            c_ls = s8.CHECKS
            try:
                r2, mk2, *_ = s8.run(job, o, pre=pre, path_cache=caches[VARIANTS[best][2]], deadline=dl,
                                     time_mode="stop")
            except s8.OverTime as ex:                # 시간 안전장치: 다듬기 전 계획을 쓴다
                r2, ls_line = None, f"ls\tstop time {ex}"
            ls_checks = s8.CHECKS - c_ls
            T["ls"] = time.perf_counter() - t
            if r2 is not None:
                ls_line = f"ls\tbefore={mk:.3f} after={mk2:.3f} used={int(mk2 < mk)}"
                if left is not None:
                    ls_line += f" budget={a.budget} left={left}"
                if mk2 < mk:
                    mk, robots = mk2, r2
    s8.write(job, out, robots)
    T["total"] = time.perf_counter() - t0

    if a.eval:                                      # 분석용: 멈춘 후보도 끝까지
        for nm in names:
            if nm not in full:
                full[nm] = s8.run(job, options(nm, job, out), pre=pre, path_cache=caches[VARIANTS[nm][2]])[1]

    # 후보별: 이름, 짧은 시험 makespan, 끝까지 계획한 makespan(가지치기로 멈춘 후보는 빈칸, --eval 이면 채움)
    lines = [f"features\t{' '.join(f'{k}={v:.3g}' for k, v in feat.items())}",
             "sample\t" + ("all" if only is None else ",".join(map(str, sorted(only))))]
    for nm in names:
        tr = f"{trial[nm]:.3f}" if nm in trial else ""
        fu = f"{full[nm]:.3f}" if nm in full else ""
        lines.append(f"{nm}\t{tr}\t{fu}")
    lines.append(f"best\t{best}")
    if CHUNK:
        lines.append(f"chunk\t{CHUNK:g}")              # (--chunk) 2·3대 계획의 조각 길이
    for nm, k in switched.items():
        lines.append(f"safe\t{nm} layer={k}")       # 첫 후보 안전장치: 이 층부터 긴 조각으로 짰다
    for nm, why in stopped.items():
        lines.append(f"safe\t{nm} stop {why}")      # 첫 후보 안전장치: 긴 조각으로도 크게 넘을 것 같아 멈춤
    for nm, evs in relaxed.items():                 # 시간 안전장치: 첫 후보의 남은 층을 더 빨리 짜는 설정으로
        for k, what in evs:
            lines.append(f"deadline\t{nm} layer={k} {what}")
    for nm, why in late.items():                    # 시간 안전장치: 두 번째 후보부터 멈춤
        lines.append(f"deadline\t{nm} stop {why}")
    if ls_line:
        lines.append(ls_line)
    lines.append(f"checks\tplan={used} ls={ls_checks} total={used + ls_checks}")
    lines.append("time\t" + " ".join(f"{k}={v:.1f}s" for k, v in T.items()))
    (out / "solve.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{job.name}: 후보 {len(names)}개 → {best} makespan={mk:.1f}s 계산 {T['total']:.1f}s "
          f"(자르기 {T['slice']:.1f} 특징 {T['features']:.1f} 경로 {T['paths']:.1f} 시험·계획 {T['plan']:.1f}"
          + (f" 국소 탐색 {T['ls']:.1f}" if "ls" in T else "") + f") 충돌 검사 {used + ls_checks:,}번"
          + (" (한도로 국소 탐색 건너뜀)" if ls_line and ls_line.startswith("ls\tskip budget") else "")
          + (" (시간으로 국소 탐색 건너뜀)" if ls_line and ls_line.startswith("ls\tskip time") else "")
          + (" (시간으로 국소 탐색 멈춤)" if ls_line and ls_line.startswith("ls\tstop time") else "")
          + "".join(f" (안전장치: {nm} 층 {k}부터 긴 조각)" for nm, k in switched.items())
          + "".join(f" (안전장치: {nm} 멈춤)" for nm in stopped)
          + "".join(f" (시간 안전장치: {nm} {len(evs)}단계까지 빠르게)" for nm, evs in relaxed.items())
          + "".join(f" (시간 안전장치: {nm} 멈춤)" for nm in late),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
