"""V8 스케줄러 = V7 + '채점 기준 그대로' 충돌 판정

V7 까지의 충돌 판정: 작업 하나마다 볼록 영역(베이스·주차점·현재 위치·조각)을 통째로 예약하고,
                    두 영역이 250mm 안이면 시간이 겹치지 않게 했다. 실제 팔이 지나가지 않는 곳까지 막아
                    동시에 일할 기회를 크게 놓친다 (04번: 로봇당 적층 4~5h 인데 makespan 12h).
V8 의 충돌 판정: 채점기와 같은 규칙을 시간축에서 직접 검사한다.
  - 팔 = XY 평면의 선분 (베이스 → 현재 토치 위치)
  - 모든 시각에 두 팔 선분의 최단거리 ≥ 250mm (팔 반경 100+100 + 여유 50) + 안전 여유
  - 각 로봇의 이미 확정된 궤적(rows)을 시간 보간해서, 후보 작업의 계획 궤적과 0.1초 간격으로 비교
  - 확정된 궤적 뒤로는 '마지막 위치에 그대로 서 있다'고 본다 (실제로도 다음 명령 전까지 그 자리에 있음)
  - 모두 막혀 아무도 움직일 수 없으면, 서 있는 로봇 하나를 주차점으로 물린다(양보)
조각 분할·순서 규칙과 후보 선택(가장 일찍 시작 가능한 것)은 V7 과 같다.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

from schedule_v1 import Robot, LIFT, TRAVEL_V, order_pieces, plen, split
import schedule_v1 as v1
from schedule_v2 import chunkify
from toolpath import best_paths
from balance import SectorLabeler

DT = 0.1            # 검사 간격(s) — 채점기 max_time_step_s 와 같음
MARGIN = 20.0       # 샘플 사이 이동(최대 150mm/s × 0.1s × 2대 / 2)을 덮는 안전 여유(mm)
MAX_TRIES = 25      # 시작 시각 후보를 몇 개까지 시험할지
RETRACT_R = 500.0
REACH_OK = 1490.0
PRUNE_MIN_FRAC = 0.2   # 예상 완료 시각으로 가지치기할 때 최소 진행률
PRUNE_MARGIN = 1.15    # 예상 makespan 이 최선의 이 배수를 넘어야 멈춘다   # 팔 길이 1500mm 에서 여유 10mm   # 비켜 설 때 먼저 팔을 이 길이(베이스로부터)까지 당긴다


# ---------------------------------------------------------------- 기하
def _pt_seg2(px, py, sx, sy, dx, dy):
    """점 (px,py) 에서 선분 s + t·d (0≤t≤1) 까지 거리의 제곱 (모두 1차원 배열)."""
    dd = dx * dx + dy * dy
    wx, wy = px - sx, py - sy
    t = (wx * dx + wy * dy) / np.where(dd > 0, dd, 1.0)
    t = np.minimum(np.maximum(t, 0.0), 1.0)
    ex, ey = wx - t * dx, wy - t * dy
    return ex * ex + ey * ey


def seg_dist(a0, a1, b0, b1):
    """XY 선분 쌍들의 최단거리. a0,b0: (2,) 또는 (N,2), a1,b1: (N,2)."""
    n = len(a1)
    a0 = np.broadcast_to(np.asarray(a0, float), (n, 2))
    b0 = np.broadcast_to(np.asarray(b0, float), (n, 2))
    ax, ay, bx, by = a0[:, 0], a0[:, 1], b0[:, 0], b0[:, 1]
    dax, day = a1[:, 0] - ax, a1[:, 1] - ay
    dbx, dby = b1[:, 0] - bx, b1[:, 1] - by
    d2 = np.minimum.reduce([
        _pt_seg2(ax, ay, bx, by, dbx, dby),
        _pt_seg2(a1[:, 0], a1[:, 1], bx, by, dbx, dby),
        _pt_seg2(bx, by, ax, ay, dax, day),
        _pt_seg2(b1[:, 0], b1[:, 1], ax, ay, dax, day),
    ])
    dist = np.sqrt(d2)
    den = dax * dby - day * dbx                           # 교차하면 0
    ox, oy = bx - ax, by - ay
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (ox * dby - oy * dbx) / den
        tb = (ox * day - oy * dax) / den
    hit = (np.abs(den) > 1e-12) & (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1)
    dist[hit] = 0.0
    return dist


class Timeline:
    """로봇의 확정 궤적을 시간으로 조회 (rows 를 따라가며 목록을 늘린다)."""

    def __init__(self, rb):
        self.rb, self.T, self.X, self.Y, self.n = rb, [], [], [], 0

    def sync(self):
        rows = self.rb.rows
        if self.n and rows[self.n - 1][0] != self.T[-1]:
            # 마지막 행이 덮어써졌을 수 있음 → 마지막 행부터 다시
            self.n -= 1
            self.T.pop(); self.X.pop(); self.Y.pop()
        for r in rows[self.n:]:
            if self.T and r[0] <= self.T[-1]:
                self.X[-1], self.Y[-1] = r[1], r[2]
                continue
            self.T.append(r[0]); self.X.append(r[1]); self.Y.append(r[2])
        self.n = len(rows)

    def xy(self, ts, lo=None, hi=None):
        lo = float(ts[0]) if lo is None else lo
        hi = float(ts[-1]) if hi is None else hi
        i0 = max(bisect.bisect_right(self.T, lo) - 1, 0)
        i1 = min(bisect.bisect_left(self.T, hi) + 1, len(self.T))
        T = self.T[i0:i1]
        return np.column_stack([np.interp(ts, T, self.X[i0:i1]), np.interp(ts, T, self.Y[i0:i1])])

    def breakpoints(self, t0, t1):
        i0 = bisect.bisect_right(self.T, t0)
        i1 = bisect.bisect_right(self.T, t1)
        return self.T[i0:i1]

    @property
    def end(self):
        return self.T[-1]


def _densify(Q, step):
    """폴리라인을 step(mm) 간격 이하로 촘촘히 나눈 점들."""
    Q = np.asarray(Q, float)[:, :2]
    out = [Q[:1]]
    for a, b in zip(Q[:-1], Q[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.vstack(out)


# ---------------------------------------------------------------- 같은 방향 쓸기
def _ang(p, c):
    return float(np.arctan2(p[1] - c[1], p[0] - c[0]))


def _winding(pc, c):
    """조각이 무게중심 c 둘레를 도는 부호 있는 각도 합(+: 반시계)."""
    a = np.arctan2(pc[:, 1] - c[1], pc[:, 0] - c[0])
    d = np.diff(a)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(d.sum())


def sweep_order(pieces, cur, c, sense):
    """띠 안의 조각들을 같은 회전 방향(sense=+1 반시계/−1 시계)으로 훑도록 방향을 맞추고 각도순으로 늘어놓는다.

    세 로봇이 모두 같은 방향으로 자기 부채꼴을 쓸면 팔 사이 각도가 120° 로 유지되어(시계 바늘 셋이
    같은 속도로 돌 듯) 경계선 근처에서 마주치는 일이 준다. 닫힌 고리는 cur 에 가장 가까운 점에서 시작한다.
    """
    out = []
    for pc in pieces:
        pc = np.asarray(pc, float)
        closed = len(pc) > 3 and np.hypot(*(pc[0] - pc[-1])) < 1e-6
        if closed:
            j = int(np.argmin(np.hypot(pc[:, 0] - cur[0], pc[:, 1] - cur[1])))
            pc = np.vstack([pc[j:-1], pc[: j + 1]])
        if _winding(pc, c) * sense < 0:
            pc = pc[::-1]
        out.append(pc)
    ref = _ang(cur, c)
    key = (lambda pc: (_ang(pc[0], c) - ref) % (2 * np.pi)) if sense > 0 else           (lambda pc: (ref - _ang(pc[0], c)) % (2 * np.pi))
    out.sort(key=key)
    return out


# ---------------------------------------------------------------- 계획 궤적
def plan(pos, t0, ch, z, vd, arc_on, via_park=None, cont=False):
    """(시각, x, y) 키포인트 목록과 끝나는 시각. XY 만 필요 (들어올림·내림은 제자리)."""
    pts = [(t0, float(pos[0]), float(pos[1]))]
    t = t0
    zl = z + LIFT
    cur = np.asarray(pos, float)
    if not cont:
        if via_park is not None:
            t += abs(max(cur[2], zl) - cur[2]) / TRAVEL_V
            pts.append((t, cur[0], cur[1]))
            t += float(np.hypot(*(via_park - cur[:2]))) / TRAVEL_V
            cur = np.array([via_park[0], via_park[1], zl])
            pts.append((t, cur[0], cur[1]))
        t += abs(max(cur[2], zl) - cur[2]) / TRAVEL_V
        pts.append((t, cur[0], cur[1]))
        t += float(np.hypot(ch[0][0] - cur[0], ch[0][1] - cur[1])) / TRAVEL_V
        pts.append((t, float(ch[0][0]), float(ch[0][1])))
        t += LIFT / TRAVEL_V + arc_on
        pts.append((t, float(ch[0][0]), float(ch[0][1])))
    for p, q in zip(ch[:-1], ch[1:]):
        d = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        if d < 0.01:
            continue
        t += d / vd
        pts.append((t, float(q[0]), float(q[1])))
    return pts, t


def sample_plan(pts):
    T = np.array([p[0] for p in pts])
    ts = np.arange(T[0], T[-1] + 1e-9, DT)
    if ts[-1] < T[-1]:
        ts = np.append(ts, T[-1])
    ts = np.union1d(ts, T)
    xy = np.column_stack([np.interp(ts, T, [p[1] for p in pts]), np.interp(ts, T, [p[2] for p in pts])])
    return ts, xy


class Checker:
    def __init__(self, robots, required, max_tries=MAX_TRIES, margin=MARGIN):
        self.max_tries = max_tries
        self.robots = robots
        self.tl = [Timeline(rb) for rb in robots]
        self.req = required + margin

    def sync(self, i=None):
        for j, tl in enumerate(self.tl):
            if i is None or j == i:
                tl.sync()

    def _check(self, i, rel_ts, rel_xy, starts):
        """계획(상대 시각 rel_ts, 위치 rel_xy)을 여러 시작 시각 starts 로 옮겼을 때
        후보마다 (충돌 없음 여부, 충돌한 로봇 집합) 을 한 번의 행렬 계산으로 구한다.
        계획 구간 + 계획 뒤 '그 자리에 서 있는' 꼬리 구간을 모두 본다."""
        starts = np.asarray(starts, float)
        K, M = len(starts), len(rel_ts)
        dur, last = rel_ts[-1], rel_xy[-1]
        base_i = self.robots[i].base
        ok = np.ones(K, bool)
        hit = [set() for _ in range(K)]
        times = (starts[:, None] + rel_ts[None, :]).ravel()
        t_ends = starts + dur
        t0 = float(t_ends.min())
        lo_t, hi_t = float(times.min()), float(times.max())
        # 다른 로봇 두 대의 '계획 구간'과 '서 있는 꼬리 구간'을 한 번의 거리 계산으로 묶는다
        others = [j for j in range(len(self.robots)) if j != i]
        A1, B0, B1, spans = [], [], [], []
        for j in others:
            tl, rb = self.tl[j], self.robots[j]
            A1.append(np.tile(rel_xy, (K, 1)))
            B1.append(tl.xy(times, lo_t, hi_t))
            B0.append(np.broadcast_to(rb.base, (K * M, 2)))
            spans.append(("main", j, K * M, None))
            if tl.end > t0:                               # 꼬리 구간: 나는 서 있고 상대는 움직임
                tt = np.arange(t0, tl.end + DT, DT)
                A1.append(np.broadcast_to(last, (len(tt), 2)))
                B1.append(tl.xy(tt, t0, float(tt[-1])))
                B0.append(np.broadcast_to(rb.base, (len(tt), 2)))
                spans.append(("tail", j, len(tt), tt))
        dist = seg_dist(base_i, np.concatenate(A1), np.concatenate(B0), np.concatenate(B1))
        pos = 0
        bad = {j: np.zeros(K, bool) for j in others}
        for kind, j, n, tt in spans:
            d = dist[pos:pos + n]
            pos += n
            if kind == "main":
                bad[j] |= d.reshape(K, M).min(axis=1) < self.req
            else:
                suf = np.minimum.accumulate(d[::-1])[::-1]   # 뒤쪽 최솟값
                m = np.searchsorted(tt, t_ends)
                inside = m < len(tt)
                tb = np.zeros(K, bool)
                tb[inside] = suf[m[inside]] < self.req
                bad[j] |= tb
        for j in others:
            ok &= ~bad[j]
            for k in np.nonzero(bad[j])[0]:
                hit[k].add(j)
        return ok, hit

    @staticmethod
    def _relative(pts):
        s0 = pts[0][0]
        ts, xy = sample_plan([(p[0] - s0, p[1], p[2]) for p in pts])
        return ts, xy

    def conflict(self, i, pts):
        """충돌하는 다른 로봇 번호 목록 (없으면 빈 목록)."""
        ts, xy = self._relative(pts)
        ok, hit = self._check(i, ts, xy, [pts[0][0]])
        return sorted(hit[0])

    def earliest(self, i, make_pts, s0, horizon=900.0):
        """make_pts(s) 계획이 충돌 없는 가장 이른 시작 시각. 못 찾으면 (None, 막은 로봇들).
        계획은 시작 시각만 달라질 뿐 모양이 같으므로 후보들을 한꺼번에 검사한다."""
        pts = make_pts(s0)
        ts, xy = self._relative(pts)
        ok, hit = self._check(i, ts, xy, [s0])
        if ok[0]:
            return s0, []
        cands = set()
        for j in range(len(self.robots)):
            if j != i:
                cands.update(self.tl[j].breakpoints(s0, s0 + horizon))
        cands = sorted(cands)
        full, drop = cands, None
        if len(cands) > self.max_tries:                 # 고르게 추려서 시험
            idx = np.linspace(0, len(cands) - 1, self.max_tries).astype(int)
            cands = [cands[k] for k in idx]
            drop = idx
        # 다른 로봇이 모두 멈춘 뒤(궤적 끝)에도 막히면 무한 대기 → 양보 필요
        s_end = max(max(self.tl[j].end for j in range(len(self.robots)) if j != i), s0)
        starts = cands + [s_end]
        ok2, hit2 = self._check(i, ts, xy, starts)
        k = int(np.argmax(ok2)) if ok2.any() else -1
        if k >= 0:
            if drop is not None and k < len(cands):
                # 성글게 훑으면 실제 가능한 시각보다 늦게 잡힌다 → 직전 후보와의 사이를 촘촘히 다시 검사
                lo = full[drop[k - 1]] if k > 0 else s0
                fine = [c for c in full if lo < c < starts[k]]
                if fine:
                    ok3, _ = self._check(i, ts, xy, fine)
                    if ok3.any():
                        return fine[int(np.argmax(ok3))], []
            return starts[k], []
        blockers = set(hit[0])
        for h in hit2:
            blockers |= h
        return None, sorted(blockers)


# ---------------------------------------------------------------- 본체


# 층 단위로 시험해 볼 설정 묶음: 조각 길이가 층마다 유불리가 갈리므로 층마다 고르게 한다.
ROLLOUTS = {
    "c3": [{"chunk": 80.0}, {"chunk": 160.0}, {"chunk": 220.0}],
    "c2": [{"chunk": 80.0}, {"chunk": 220.0}],
    # 층마다 로봇 수까지 바꾼다: 위쪽 작은 층은 3대가 서로 비키느라 오히려 느리다.
    "mix": [{"chunk": 160.0}, {"chunk": 80.0}, {"chunk": 220.0},
            {"chunk": 160.0, "robots": 1}, {"chunk": 160.0, "robots": 2}],
    "rob": [{"chunk": 160.0}, {"chunk": 160.0, "robots": 1}, {"chunk": 160.0, "robots": 2}],
}


def _place_layer(ctx, k, paths, z, zl, P, policy=None):
    """한 층을 스케줄한다. ctx(로봇·검사기·주차 상태)를 그 자리에서 바꾸고 비켜선 횟수를 돌려준다.

    층 단위로 떼어 둔 이유: 같은 층을 설정 몇 가지로 시험해 보고 가장 빨리 끝난 것만 남기기 위해서다.
    """
    robots, chk, parked, loads = ctx["robots"], ctx["chk"], ctx["parked"], ctx["loads"]
    first, yields = ctx["first"], 0
    opts, arc_on, arc_off, vd = P["opts"], P["arc_on"], P["arc_off"], P["vd"]
    band_dir, nrob, w_over, retract = P["band_dir"], P["nrob"], P["w_over"], P["retract"]
    order2 = P.get("order2", False)
    sweep = P.get("sweep", False)            # 같은 회전 방향으로 쓸기
    pair = P.get("pair")                     # 2대 설정에서 쓸 로봇 두 대의 번호(0-based)
    link_gap = P.get("link_gap", 0.0)        # 조각 사이를 적층으로 이을 최대 거리(0=안 이음)
    park_end = P.get("park_end", True)       # 층 끝에 주차점까지 물릴지          # 아래에서 P 는 경로 변수로 재사용되므로 먼저 꺼내 둔다

    def go_park(j, zl):
        """로봇 j 를 주차점으로 물린다 (충돌 없는 시각을 찾아서)."""
        rb = robots[j]
        if np.hypot(*(rb.pos[:2] - rb.park_xy)) < 1e-6:
            parked[j] = True
            return True
        # 먼저 자기 팔 선분을 따라 베이스 쪽으로 당긴다(팔이 기존 선분의 일부가 되므로 새 충돌이 없음)
        v = rb.pos[:2] - rb.base
        L = float(np.hypot(*v))
        mid = rb.base + v * (retract / L) if L > retract else rb.pos[:2].copy()

        def mk(s):
            p = [(s, float(rb.pos[0]), float(rb.pos[1]))]
            t = s + abs(max(rb.pos[2], zl) - rb.pos[2]) / TRAVEL_V
            p.append((t, float(rb.pos[0]), float(rb.pos[1])))
            t += float(np.hypot(*(mid - rb.pos[:2]))) / TRAVEL_V
            p.append((t, float(mid[0]), float(mid[1])))
            t += float(np.hypot(*(rb.park_xy - mid))) / TRAVEL_V
            p.append((t, float(rb.park_xy[0]), float(rb.park_xy[1])))
            return p
        s, _ = chk.earliest(j, mk, rb.t)
        if s is None:
            return False
        rb.idle_until(s)
        rb.travel([rb.pos[0], rb.pos[1], max(rb.pos[2], zl)])
        rb.travel([mid[0], mid[1], max(rb.pos[2], zl)])
        rb.travel([rb.park_xy[0], rb.park_xy[1], zl])
        chk.sync(j)
        parked[j] = True
        return True

    if first:
        for rb in robots:
            rb.travel([rb.park_xy[0], rb.park_xy[1], zl])
        first = False
        chk.sync()
        parked = [True] * 3
    clock = max(rb.t for rb in robots)
    for rb in robots:
        rb.idle_until(clock)
        if abs(rb.pos[2] - zl) > 1e-6:
            rb.travel([rb.pos[0], rb.pos[1], zl])
    clock = max(rb.t for rb in robots)
    for rb in robots:
        rb.idle_until(clock)
    chk.sync()

    # 일감 나누기: 부하 균형 회전 부채꼴(기본) 또는 기존 작업대 중심 부채꼴
    # 좁은 형상은 팔이 250mm 떨어져야 해서 3대가 동시에 못 들어간다. 그럴 때는 로봇 수를 줄이면
    # 서로 비켜 주는 시간이 사라져 오히려 빠르다(--robots 1/2, 안 쓰는 로봇은 주차한 채 둔다).
    labeler = None
    if nrob < 3:
        # 팔 길이(1500mm) 안에 그 층 전체가 들어오는 조합만 쓴다. 안 되면 그 층은 3대로 돌린다.
        P = np.vstack([np.asarray(Q, float)[:, :2] for Q in paths])
        if nrob == 1:
            far = [float(np.max(np.hypot(*(P - rb.base).T))) for rb in robots]
            j = int(np.argmin(far))
            if far[j] <= REACH_OK:
                labeler = lambda p, _j=j: ("S", _j)
        else:
            # 어느 두 대를 쓸지(--pair). 형상이 한쪽으로 치우치면 조합에 따라 유불리가 갈린다.
            act = pair if pair else [0, 1]
            lb = SectorLabeler(paths, [robots[i].base for i in act])
            if getattr(lb, "unreach", 1.0) <= 0.0:
                # SectorLabeler 는 선분 가운데 점만 도달 거리를 보므로 긴 선분의 끝점은 팔 밖일 수 있다
                # (새 형상 08: 1503.8mm 로 FAIL). 경로를 2mm 로 촘촘히 나눠 모든 점이 담당 로봇
                # 팔 안(1490mm)에 드는지 확인하고, 하나라도 벗어나면 이 층은 3대로 돌린다.
                dense = np.vstack([_densify(Q, 2.0) for Q in paths])
                own = np.array([act[lb.owner(q)] for q in dense])
                bases = np.array([robots[i].base for i in range(3)])
                if np.all(np.hypot(*(dense - bases[own]).T) <= REACH_OK):
                    labeler = lambda p, _lb=lb, _act=act: ("S", _act[_lb.owner(p)])
    if labeler is None:
        labeler = (SectorLabeler(paths, [rb.base for rb in robots])
                   if getattr(opts, "balance", True) else v1.seq_label)
    groups = {}
    for P in paths:
        for (_, o), pc in split(P, labeler):
            groups.setdefault(o, []).append(pc)
    queues = {i: [] for i in range(3)}
    for o, pcs in groups.items():
        bands = {}
        for pc in pcs:
            r = float(np.max(np.hypot(pc[:, 0], pc[:, 1])))
            bands.setdefault(int(r // opts.band), []).append(pc)
        # 층 끝에 주차하지 않으면 다음 층은 지금 서 있는 자리에서 이어서 시작한다
        ordered, cur = [], (robots[o].park_xy if park_end else robots[o].pos[:2])
        # 진행 방향: 기본은 모두 바깥 띠 → 안쪽. alt 는 로봇마다 반대로 가게 해 같은 반지름에서
        # 만나는 일을 줄인다(팔이 가까워지면 서로 비켜서느라 시간을 쓴다). in 은 모두 안쪽 → 바깥.
        outward = {"out": True, "in": False}.get(band_dir, o % 2 == 0)
        if not park_end:
            outward = (k % 2 == 0)           # 층마다 방향을 뒤집어 다음 층 첫 조각이 발밑에 오게
        c_layer = np.mean(np.vstack([np.asarray(Q, float)[:, :2] for Q in paths]), axis=0)
        for bi, b in enumerate(sorted(bands, reverse=outward)):
            if sweep:                                    # 띠마다 방향을 번갈아(지그재그) 같은 회전 방향으로
                seg = sweep_order(bands[b], cur, c_layer, +1 if bi % 2 == 0 else -1)
            else:
                seg = order_pieces(bands[b], cur)
            if order2:                                   # 2-opt 로 조각 순서 다듬기
                seg = v1.refine_order(seg, cur)
            ordered += seg
            cur = seg[-1][-1]
        if link_gap > 0 and len(ordered) > 1:
            # 앞 조각 끝과 다음 조각 시작이 가까우면 아크를 끄지 않고 그대로 이어 긋는다.
            # 경로 생성 단계의 나선(link_rings)은 한 각도에서만 이어져, 부채꼴로 나눈 뒤에는
            # 로봇 하나만 이어진 채 남는다. 배정 후에 이으면 세 로봇 모두 아크 재점화가 준다.
            # 이어 붙인 구간(≤9mm)은 양옆 고리 비드(반폭 4mm)에 이미 덮이는 자리다.
            merged = [ordered[0]]
            for pc in ordered[1:]:
                if float(np.hypot(*(pc[0] - merged[-1][-1]))) <= link_gap:
                    merged[-1] = np.vstack([merged[-1], pc[1:]])
                else:
                    merged.append(pc)
            ordered = merged
        for pid, pc in enumerate(ordered):
            for ci, ch in enumerate(chunkify(pc, opts.chunk)):
                queues[o].append((pid, ci, ch))

    if policy is not None:                           # 층 단위 보상용 기록
        policy.layer = k
        policy.layer_t0[k] = max(rb.t for rb in robots)
    arc_open = [False] * 3
    last_key = [None] * 3
    while any(queues[i] for i in range(3)):
        best, bkey, blocked = None, (float("inf"), 0), set()
        cands = []                                   # 정책(PPO)용: 충돌 검사를 통과한 모든 후보
        for i in range(3):
            if not queues[i]:
                continue
            rb = robots[i]
            pos = rb.pos.copy()
            for qi in range(min(opts.lookahead, len(queues[i]))):
                pid, ci, ch = queues[i][qi]
                cont = (qi == 0 and arc_open[i] and last_key[i] == (pid, ci - 1)
                        and np.hypot(*(pos[:2] - ch[0])) < 1e-6)
                if cont:
                    pts, _ = plan(pos, rb.t, ch, z, vd, arc_on, cont=True)
                    if not chk.conflict(i, pts):
                        cand = (rb.t, qi, i, "cont")
                        cands.append(cand)
                        key = (rb.t, qi)                 # 이어 붙이면 준비 손해 0
                        if best is None or key < bkey:
                            best, bkey = cand, key
                        break
                off = arc_off if arc_open[i] else 0.0
                s0 = rb.t + off
                s, bl = chk.earliest(i, lambda s_, ch=ch: plan(pos, s_, ch, z, vd, arc_on)[0], s0)
                if s is None:
                    blocked.update(bl)
                    continue
                cand = (s, qi, i, "direct")
                cands.append(cand)
                # 준비 손해 = 아크 끄기 + 들어올림·이동 + 내림 + 아크 켜기 (적층에 쓰이지 않는 시간)
                over = (off + abs(max(pos[2], zl) - pos[2]) / TRAVEL_V
                        + float(np.hypot(ch[0][0] - pos[0], ch[0][1] - pos[1])) / TRAVEL_V
                        + LIFT / TRAVEL_V + arc_on)
                key = (s + w_over * over, qi)
                if best is None or key < bkey:
                    best, bkey = cand, key
                if abs(s - s0) < 1e-9:               # 큐 순서는 그대로 지킨다(아크 이어가기)
                    break

        if best is None:
            # 모두 막힘 → 막고 서 있는 로봇 중 하나를 주차점으로 물린다
            moved = False
            for j in sorted(blocked | {0, 1, 2}, key=lambda j: robots[j].t):
                if parked[j]:
                    continue
                if arc_open[j]:
                    robots[j].wait(arc_off)
                    arc_open[j] = False
                if go_park(j, zl):
                    yields += 1
                    moved = True
                    break
            if not moved:
                chk.sync()
                for a_ in range(3):
                    for b_ in range(a_ + 1, 3):
                        d_ = seg_dist(robots[a_].base, robots[a_].pos[:2][None, :], robots[b_].base,
                                      robots[b_].pos[:2][None, :])[0]
                        print(f"  [진단] 로봇{a_ + 1}-{b_ + 1} 현재 팔 거리 {d_:.1f}mm  "
                              f"pos{a_ + 1}={robots[a_].pos[:2].round(1)} pos{b_ + 1}={robots[b_].pos[:2].round(1)}")
                info = [(j + 1, parked[j], round(robots[j].t, 1), len(queues[j])) for j in range(3)]
                raise RuntimeError(f"layer {k}: 교착 상태 — (로봇, 주차, 시각, 남은조각)={info}")
            continue

        if policy is not None and len(cands) > 1:     # 학습한 정책이 후보 중에서 고른다
            best = cands[policy.pick(cands, robots, queues, loads, vd)]
        s, qi, i, kind = best
        rb = robots[i]
        pid, ci, ch = queues[i].pop(qi)
        loads[i] += float(np.sum(np.hypot(*np.diff(ch, axis=0).T))) / vd
        if kind != "cont":
            if arc_open[i]:
                rb.wait(arc_off)
                arc_open[i] = False
            rb.idle_until(s)
            rb.travel([rb.pos[0], rb.pos[1], max(rb.pos[2], zl)])
            rb.travel([ch[0][0], ch[0][1], zl])
            rb.travel([ch[0][0], ch[0][1], z])
            rb.wait(arc_on)
        t_before_commit = s if kind != "cont" else rb.t
        rb.deposit(ch, z, vd)
        arc_open[i] = True
        parked[i] = False
        last_key[i] = (pid, ci)
        chk.sync(i)
        if opts.debug:                                   # 확정한 실제 궤적을 다시 검사
            tl = chk.tl[i]
            ts = np.arange(t_before_commit, rb.t, DT)
            if len(ts):
                mine = tl.xy(ts)
                for j in range(3):
                    if j == i:
                        continue
                    d_ = seg_dist(rb.base, mine, robots[j].base, chk.tl[j].xy(ts))
                    if d_.min() < required:
                        m_ = int(d_.argmin())
                        print(f"  [위반] 층{k} 로봇{i + 1}({kind}) vs 로봇{j + 1}: {d_.min():.1f}mm "
                              f"t={ts[m_]:.1f} 계획시작={s:.1f} 로봇{j + 1}궤적끝={chk.tl[j].end:.1f}")

    # 층 끝: 아크 끄고 주차점으로
    for i, rb in enumerate(robots):
        if arc_open[i]:
            rb.wait(arc_off)
            arc_open[i] = False
        chk.sync(i)
    for i in (sorted(range(3), key=lambda i: robots[i].t) if park_end else ()):
        if not go_park(i, zl):
            raise RuntimeError(f"layer {k}: 층 끝 주차 실패 (로봇 {i + 1})")
    if policy is not None:
        policy.layer_t1[k] = max(rb.t for rb in robots)
    ctx["first"] = first
    return yields

class Pruned(Exception):
    """이미 다른 설정보다 느려져 더 계산할 필요가 없다."""


def run(job: Path, opts, policy=None):
    cfg = load_config(job / "config.yaml")
    mesh = load_target_mesh(job / "target.stl", cfg)
    pr = cfg.process
    arc_on = (pr.arc_on_time_s or 0.0) if opts.arc else 0.0
    arc_off = (pr.arc_off_time_s or 0.0) if opts.arc else 0.0
    vd = pr.deposition_speed_mm_s
    required = max(r.arm_envelope_radius_mm for r in cfg.robots) * 2 + cfg.collision.arm_clearance_mm

    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    layers = slice_target_layers(mesh, list(range(n)), cfg)
    robots = [Robot(r.id, r.base_xyz_mm, r.home_xyz_mm) for r in sorted(cfg.robots, key=lambda r: r.id)]
    v1.DIRS = [v1.ang_of(rb.base) for rb in robots]
    chk = Checker(robots, required, getattr(opts, "max_tries", MAX_TRIES),
                  float(getattr(opts, "margin", MARGIN)))
    retract = getattr(opts, "retract", RETRACT_R)
    band_dir = getattr(opts, "band_dir", "out") or "out"       # 띠 진행 방향(out/in/alt)
    nrob = int(getattr(opts, "robots", 3) or 3)              # 일감을 받는 로봇 수
    w_over = float(getattr(opts, "overhead_w", 0.0) or 0.0)   # 준비 손해에 주는 가중치(0=기존 그리디)
    parked = [False] * 3
    min_iou, first, yields = 1.0, True, 0
    roll = ROLLOUTS.get(getattr(opts, "rollout", "") or "", None)   # 층마다 시험해 볼 설정들
    loads = np.zeros(3)                                  # 로봇별 누적 적층 시간(정책 특징)

    ctx = {"robots": robots, "chk": chk, "parked": parked, "loads": loads, "first": True}
    P = {"opts": opts, "arc_on": arc_on, "arc_off": arc_off, "vd": vd,
         "band_dir": band_dir, "nrob": nrob, "w_over": w_over, "retract": retract,
         "order2": bool(getattr(opts, "order2", False)),
         "park_end": bool(getattr(opts, "park_end", True)),
         "link_gap": float(getattr(opts, "link_gap", 0.0) or 0.0),
         "sweep": bool(getattr(opts, "sweep", False)),
         "pair": ([int(x) - 1 for x in str(opts.pair).split(",")] if getattr(opts, "pair", None) else None)}

    path_cache = None
    if getattr(opts, "paths", None):
        import pickle
        with open(opts.paths, "rb") as _f:
            path_cache = pickle.load(_f)
    bound_file = getattr(opts, "bound_file", None)
    # 예상 완료 시각으로 가지치기하려면 전체 적층 길이가 필요하다(경로 캐시가 있을 때만)
    lens = {}
    if bound_file and path_cache is not None:
        lens = {k: sum(plen(np.asarray(Q)) for Q in path_cache[k][0])
                for k in path_cache if path_cache[k][0]}
    total_len, done_len = float(sum(lens.values())), 0.0
    for k in range(n):
        A = layers[k]
        if A.is_empty:
            continue
        if bound_file:
            # 가지치기: 포트폴리오에서 먼저 끝난 설정의 makespan 이 여기 적힌다. 지금 로봇 시각이 이미
            # 그보다 크면 이 설정은 절대 이길 수 없으므로(시각은 줄지 않음) 바로 멈춘다. 결과는 그대로.
            try:
                bnd = float(Path(bound_file).read_text().strip() or "inf")
            except (OSError, ValueError):
                bnd = float("inf")
            clock = max(rb.t for rb in ctx["robots"])
            if clock > bnd:
                raise Pruned(f"layer {k}: {clock:.0f}s > 최선 {bnd:.0f}s")
            # 20% 이상 진행했으면 지금 속도로 끝까지 갔을 때의 예상 makespan 이 최선보다 15% 이상 크면 멈춘다
            frac = done_len / total_len if total_len > 0 else 0.0
            if frac >= PRUNE_MIN_FRAC and clock / frac > bnd * PRUNE_MARGIN:
                raise Pruned(f"layer {k}: 예상 {clock / frac:.0f}s > 최선 {bnd:.0f}s×{PRUNE_MARGIN}")
        z = pr.build_plane_z_mm + (k + 0.5) * pr.layer_height_mm
        zl = z + LIFT
        if path_cache is not None and k in path_cache:     # 포트폴리오가 미리 계산한 층별 경로
            paths, iou_k = path_cache[k]
        else:
            paths, sc, _ = best_paths(A)
            iou_k = sc["iou"]
        min_iou = min(min_iou, iou_k)

        done_len += lens.get(k, 0.0)
        if not roll:
            yields += _place_layer(ctx, k, paths, z, zl, P, policy)
        else:
            # 층마다 설정 몇 가지를 시험해 그 층을 가장 빨리 끝낸 것만 남긴다(층 단위 되돌리기).
            bestc = None
            for cf in roll:
                c2 = copy.deepcopy(ctx)
                P2 = dict(P, opts=SimpleNamespace(**{**vars(opts), **cf}),
                          nrob=cf.get("robots", P["nrob"]), band_dir=cf.get("band_dir", P["band_dir"]))
                y2 = _place_layer(c2, k, paths, z, zl, P2, policy)
                end = max(rb.t for rb in c2["robots"])
                if bestc is None or end < bestc[0]:
                    bestc = (end, c2, y2)
            yields += bestc[2]
            ctx = bestc[1]
            robots = ctx["robots"]

    for rb in robots:
        last = rb.rows[-1]
        rb.rows[-1] = (last[0], last[1], last[2], last[3], "W")
        if len(rb.rows) < 2:
            rb.rows.append((last[0] + 1.0, last[1], last[2], last[3], "W"))
    return robots, max(rb.rows[-1][0] for rb in robots), min_iou, n, yields


def write(job, out, robots):
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(job / "config.yaml", out / "config.yaml")
    shutil.copyfile(job / "target.stl", out / "target.stl")
    nrows = 0
    with open(out / "trajectory.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("robot_id,time_s,x_mm,y_mm,z_mm,mode\n")
        for rb in robots:
            prev = -1.0
            for t_, x, y, zz, m in rb.rows:
                tr = round(t_, 6)
                if tr <= prev:
                    tr = round(prev + 1e-6, 6)
                prev = tr
                f.write(f"{rb.rid},{tr:.6f},{x:.6f},{y:.6f},{zz:.6f},{m}\n")
                nrows += 1
    return nrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--no-arc", dest="arc", action="store_false")
    ap.add_argument("--chunk", type=float, default=40.0)
    ap.add_argument("--band", type=float, default=60.0)
    ap.add_argument("--paths", default=None, help="toolpath_cache.py 로 미리 계산한 층별 경로 파일")
    ap.add_argument("--lookahead", type=int, default=6)
    ap.add_argument("--no-balance", dest="balance", action="store_false",
                    help="부하 균형 배정 끄기(=기존 작업대 중심 부채꼴)")
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help="충돌 판정 안전 여유(mm). 샘플 0.1s 사이 이동을 덮는 값(기본 20)")
    ap.add_argument("--max-tries", dest="max_tries", type=int, default=MAX_TRIES)
    ap.add_argument("--retract", type=float, default=RETRACT_R)
    ap.add_argument("--rollout", choices=sorted(ROLLOUTS), default=None,
                    help="층마다 여러 설정을 시험해 가장 빨리 끝난 것을 채택(계산시간이 배로 든다)")
    ap.add_argument("--link-gap", dest="link_gap", type=float, default=0.0,
                    help="배정된 조각 사이가 이 거리(mm) 안이면 아크를 끄지 않고 이어 긋는다")
    ap.add_argument("--no-park-layer", dest="park_end", action="store_false",
                    help="층 끝마다 주차점으로 물리지 않고 그 자리에서 다음 층을 시작(이동 시간 감소)")
    ap.add_argument("--bound-file", dest="bound_file", default=None,
                    help="포트폴리오가 현재 최선 makespan 을 적는 파일. 로봇 시각이 이를 넘으면 중단")
    ap.add_argument("--pair", default=None, help="--robots 2 일 때 쓸 로봇 두 대, 예: 1,3 (기본 1,2)")
    ap.add_argument("--sweep", action="store_true",
                    help="세 로봇이 같은 회전 방향으로 부채꼴을 쓸도록 조각 순서·방향을 맞춘다")
    ap.add_argument("--order2", action="store_true",
                    help="조각 순서를 2-opt 로 다듬는다(이동 시간 감소 시도)")
    ap.add_argument("--band-dir", dest="band_dir", choices=["out", "in", "alt"], default="out",
                    help="띠 진행 방향: out=모두 바깥→안, in=모두 안→바깥, alt=로봇마다 반대")
    ap.add_argument("--robots", type=int, default=3, choices=[1, 2, 3],
                    help="일감을 받는 로봇 수. 좁은 형상은 1~2대가 더 빠를 수 있다")
    ap.add_argument("--overhead-w", dest="overhead_w", type=float, default=0.0,
                    help="후보를 고를 때 준비 손해(아크 재점화·이동)에 주는 가중치. 0이면 기존 그리디")
    ap.add_argument("--policy", default=None, help="ppo_policy.py 로 학습한 정책(.npz)")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    job, out = Path(a.job), Path(a.out)
    pol = None
    if a.policy:
        from ppo_policy import Policy
        pol = Policy.load(a.policy)
    try:
        robots, mk, iou, n, yields = run(job, a, pol)
    except Pruned as e:
        print(f"{job.name}: PRUNED {e} gen={time.time() - t0:.1f}s")
        sys.exit(3)
    nrows = write(job, out, robots)
    print(f"{job.name}: layers={n} rows={nrows} makespan={mk:.1f}s ({mk / 3600:.2f}h) "
          f"self_min_IoU={iou:.3f} yields={yields} gen={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
