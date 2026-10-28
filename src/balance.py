# -*- coding: utf-8 -*-
"""층마다 로봇 3대의 일감을 고르게 나누는 '회전 부채꼴' 배정.

기존 규칙(schedule_v1.seq_label): 작업대 중심(0,0)에서 본 각도로 가장 가까운 베이스 방향 로봇에게 준다.
  → 형상이 한쪽으로 치우치거나 모양이 비대칭이면 한 로봇에 일이 몰린다
    (검증 15번: 로봇3 적층 12.5h, 로봇1·2 는 6.6h·7.9h → makespan 이 로봇3 혼자 일한 시간에 묶임)

새 규칙:
  1) 그 층 경로의 무게중심 c 에서 각도를 잰다.
  2) 세 로봇 사이 경계 각도 3개를 움직여, '가장 많이 맡은 로봇의 적층 길이'가 최소가 되게 한다.
  3) 각 로봇은 자기 베이스에서 도달 반경(1500mm, 여유 포함) 안의 점만 맡을 수 있다(제약).
경계는 항상 두 로봇의 베이스 방향 사이에 있으므로 각 로봇의 몫은 자기 쪽 부채꼴이 된다
(로봇끼리 서로 반대편으로 팔을 뻗지 않음).
"""
from __future__ import annotations

import math

import numpy as np

REACH = 1500.0          # 스케줄러가 config 의 xy_reach_radius_mm 로 덮어쓴다
REACH_MARGIN = 10.0


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SectorLabeler:
    def __init__(self, paths, bases, reach=None, steps=48, passes=3, shift=None):
        reach = REACH - REACH_MARGIN if reach is None else reach   # REACH 는 스케줄러가 config 로 맞춘다
        self.bases = [np.asarray(b[:2], float) for b in bases]
        pts, wts = [], []
        for P in paths:
            P = np.asarray(P, float)[:, :2]
            if len(P) < 2:
                continue
            mid = (P[1:] + P[:-1]) / 2
            ln = np.hypot(*np.diff(P, axis=0).T)
            pts.append(mid)
            wts.append(ln)
        if not pts:
            self.c = np.zeros(2)
            self.bounds = None
            return
        X = np.vstack(pts)
        W = np.concatenate(wts)
        self.c = (X * W[:, None]).sum(0) / max(W.sum(), 1e-9)
        # 로봇을 무게중심 기준 베이스 방향 각도 순으로 정렬
        self.dir = np.array([math.atan2(*(b - self.c)[::-1]) for b in self.bases])
        self.order = list(np.argsort(self.dir))
        ang = np.arctan2(X[:, 1] - self.c[1], X[:, 0] - self.c[0])
        reach_ok = np.stack([np.hypot(*(X - b).T) <= reach for b in self.bases])   # (3, N)
        # 경계 k: order[k] 와 order[k+1] 사이. 두 베이스 방향 사이의 비율 t∈(0,1) 로 표현
        n = len(self.order)
        spans = []
        for k in range(n):
            a0 = self.dir[self.order[k]]
            a1 = self.dir[self.order[(k + 1) % n]]
            d = (a1 - a0) % (2 * np.pi)
            spans.append((a0, d))
        self.spans = spans

        def owners(ts):
            bnd = [spans[k][0] + spans[k][1] * ts[k] for k in range(n)]
            own = np.full(len(ang), self.order[0], int)   # 경계선 위의 점도 누군가에게 배정
            for k in range(n):
                lo = bnd[k - 1]
                width = (bnd[k] - lo) % (2 * np.pi)
                inside = ((ang - lo) % (2 * np.pi)) < width
                own[inside] = self.order[k]
            return own

        def cost(ts):
            own = owners(ts)
            load = np.array([W[own == i].sum() for i in range(n)])
            bad = W[~reach_ok[own, np.arange(len(own))]].sum()
            return load.max() + 50.0 * bad, load, bad

        grid = np.linspace(0.08, 0.92, steps)
        ts = np.full(n, 0.5)
        best = cost(ts)
        for _ in range(passes):
            improved = False
            for k in range(n):
                for g in grid:
                    trial = ts.copy()
                    trial[k] = g
                    c = cost(trial)
                    if c[0] < best[0] - 1e-6:
                        best, ts, improved = c, trial, True
            if not improved:
                break
        self.ts = ts
        self.load = best[1]
        self.unreach = best[2]
        if shift:
            # 국소 탐색(schedule_v8._place_ls)이 경계를 옮겨 본다: {경계 번호: 비율 변화}. 적층 길이만 보고 맞춘
            # 경계가 실제로는 서로 비키느라 한 로봇만 늦게 끝날 때가 있다. 팔이 닿는지는 owner 와 스케줄러가 다시 본다.
            ts = np.clip(ts + np.array([shift.get(k, 0.0) for k in range(n)]), 0.02, 0.98)
            self.ts = ts
        self.bounds = [spans[k][0] + spans[k][1] * ts[k] for k in range(n)]
        self.reach = reach

    def owner(self, p):
        if self.bounds is None:
            return 0
        a = math.atan2(p[1] - self.c[1], p[0] - self.c[0])
        n = len(self.order)
        for k in range(n):
            lo = self.bounds[k - 1]
            width = (self.bounds[k] - lo) % (2 * math.pi)
            if (a - lo) % (2 * math.pi) < width:
                o = self.order[k]
                break
        else:
            o = self.order[0]
        if math.hypot(p[0] - self.bases[o][0], p[1] - self.bases[o][1]) > self.reach:
            # 경계 근처에서 도달 불가면 가장 가까운 베이스 로봇으로 (안전장치)
            o = int(np.argmin([math.hypot(p[0] - b[0], p[1] - b[1]) for b in self.bases]))
        return o

    def __call__(self, p):
        return ("S", self.owner(p))


class IslandLabeler:
    """층에 떨어진 섬(연결 조각)이 여럿이면 섬을 통째로 한 로봇에게 준다.

    부채꼴 배정은 각도로 자르기 때문에, 기둥처럼 떨어진 섬이 있는 층에서는 섬이 한 로봇에게 몰리거나
    섬 하나가 두 로봇 사이에서 잘린다(505-04 기둥 층: 로봇3 적층 67s·대기 227s, 로봇2 적층 277s).
    여기서는 섬마다 무게(적층 길이)를 재고, 무거운 섬부터 '닿을 수 있는 로봇 중 지금까지 일감이 가장
    적은 로봇'에게 준다(LPT). 동점이면 베이스가 가까운 로봇. 섬이 하나뿐이면 쓰지 않는다(valid=False).
    """

    # gap: 경로를 이만큼 부풀려 붙는 것끼리 한 섬으로 본다. 6mm 로는 한 벽 안의 떨어진 경로 조각까지
    # 섬으로 잘못 봐서 물결 벽(404-10)이 +13.6% 나빠졌다. 진짜 섬(기둥·떨어진 조각)은 수십 mm 이상
    # 떨어져 있으므로 12mm(=24mm 이내면 같은 섬)로 둔다.
    def __init__(self, paths, bases, reach=None, gap=12.0):
        reach = REACH - REACH_MARGIN if reach is None else reach
        from shapely import LineString, unary_union, STRtree
        self.bases = [np.asarray(b[:2], float) for b in bases]
        self.valid = False
        segs = [np.asarray(P, float)[:, :2] for P in paths if len(P) >= 2]
        if len(segs) < 2:
            return
        blob = unary_union([LineString(s).buffer(gap) for s in segs])
        comps = list(getattr(blob, "geoms", [blob]))
        if len(comps) < 2:
            return
        tree = STRtree(comps)
        w = np.zeros(len(comps))
        pts = [[] for _ in comps]
        for s in segs:
            from shapely import Point
            ci = int(tree.nearest(Point(*s[len(s) // 2])))
            w[ci] += float(np.sum(np.hypot(*np.diff(s, axis=0).T)))
            pts[ci].append(s)
        # 섬들을 층 무게중심 둘레 각도 순으로 늘어놓고, 이웃한 섬끼리 묶어 로봇 수만큼 연속 구간으로 자른다.
        # (무거운 섬부터 한가한 로봇에 주는 LPT 는 로봇이 띄엄띄엄 떨어진 섬을 맡아 팔이 교차해 오히려 느렸다)
        n, R = len(comps), len(self.bases)
        if n > 30:
            return
        # 작고 동그스름한 섬(기둥·떨어진 조각)일 때만 쓴다. 긴 평행 띠(물결 404-10: 섬 지름 360mm,
        # 이웃 섬 중심거리 59mm)는 여러 로봇이 한 띠를 나눠 맡는 부채꼴이 더 빠르다.
        # 기준: 가장 큰 섬 지름 ÷ 이웃 섬 중심거리 중앙값 ≤ 1.5 (실제 섬 모양 0.78~1.06, 물결 6.1)
        cen_c = np.array([c.centroid.coords[0] for c in comps])
        diam = [float(np.hypot(c.bounds[2] - c.bounds[0], c.bounds[3] - c.bounds[1])) - 2 * gap for c in comps]
        nn = [float(np.min(np.delete(np.hypot(*(cen_c - cen_c[i]).T), i))) for i in range(n)]
        if max(diam) > 1.5 * max(float(np.median(nn)), 1e-9):
            return
        allp = np.vstack([np.vstack(p) for p in pts if p])
        cen = allp.mean(axis=0)
        P = [np.vstack(p) if p else np.asarray(comps[i].exterior.coords)[:, :2] for i, p in enumerate(pts)]
        ang = np.array([math.atan2(*(q.mean(axis=0) - cen)[::-1]) for q in P])
        idx = list(np.argsort(ang))
        bang = np.array([math.atan2(*(b - cen)[::-1]) for b in self.bases])
        rorder = list(np.argsort(bang))                       # 로봇도 같은 방향(각도) 순서로
        reach_ok = np.array([[np.max(np.hypot(*(P[i] - b).T)) <= reach for b in self.bases] for i in range(n)])
        best = None
        for st in range(n):                                   # 원형으로 늘어선 섬 중 시작점
            seq = idx[st:] + idx[:st]
            for c1 in range(0, n + 1):
                for c2 in range(c1, n + 1):
                    arcs = [seq[:c1], seq[c1:c2], seq[c2:]]
                    for rot in range(R):                      # 어느 로봇부터 시작할지
                        own = [rorder[(q + rot) % R] for q in range(R)]
                        if any(not reach_ok[i, own[q]] for q in range(R) for i in arcs[q]):
                            continue
                        loads = [sum(w[i] for i in arcs[q]) for q in range(R)]
                        key = max(loads)
                        if best is None or key < best[0] - 1e-9:
                            best = (key, arcs, own, loads)
        if best is None:
            return
        _, arcs, own, loads = best
        self.owner_of = [0] * n
        for q in range(R):
            for i in arcs[q]:
                self.owner_of[i] = own[q]
        self.comps, self.tree, self.load = comps, tree, loads
        self.valid = True

    def owner(self, p):
        from shapely import Point
        return self.owner_of[int(self.tree.nearest(Point(float(p[0]), float(p[1]))))]

    def __call__(self, p):
        return ("S", self.owner(p))
