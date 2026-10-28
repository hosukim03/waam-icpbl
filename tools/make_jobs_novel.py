# -*- coding: utf-8 -*-
"""21개 계열과 다른 방식으로 만든 확인용 형상 생성기.

사용자 지적(2026-10-27): "시드가 달라도 비슷비슷한 겉모습이 문제가 될 것 같은데" → make_jobs_diverse.py 는 시드를
바꿔도 종류 21가지가 그대로이고 벽 두께가 8·16·24 mm(비드 1·2·3줄) 뿐이다. 여기서는 정해진 계열 없이 무작위
윤곽을 만들고, 두께는 연속값, 높이 방향 변화는 구간으로 끊지 않고 층마다 조금씩 바꾼다.

윤곽 (매개변수는 모두 무작위)
  blob     : 둥근 덩어리(속참). 반지름에 여러 주기의 물결을 더한 매끈한 윤곽
  shell    : 무작위 윤곽을 따라가는 벽. 군데군데 끊긴 곳이 있을 수 있다
  tree     : 나뭇가지처럼 갈라지는 벽. 가지 끝으로 갈수록 얇아진다(끝은 6–9 mm)
  cells    : 불규칙한 칸막이(무작위 점의 Voronoi 칸 경계) + 바깥 벽
  polywall : 여러 번 꺾인 벽 1–3개(서로 겹칠 수 있음)
  bars     : 막대·판을 무작위로 겹친 부품 모양 + 무작위 구멍
  twins    : 떨어진 덩어리 2–3개가 위로 가며 합쳐지거나 갈라진다
  holey    : 속찬 덩어리에 구멍이 높이에 따라 생겼다 커졌다 사라진다
벽 두께는 8–40 mm 사이 아무 값이다(대부분 8–24 mm). 8 mm 보다 얇은 벽만으로 된 층은 비드가 반드시 넘쳐 누구도
통과할 수 없으므로 얇은 부분은 가지 끝에만 둔다.

높이 방향 변화 (층마다 조금씩. 한 층에 옆으로 2 mm 넘게 움직이지 않게 줄인다)
  taper(서서히 커지거나 작아짐), twist(비틀림), drift(서서히 옆으로 밀림), thicken(벽 두께가 서서히 바뀜, 벽만)
  아무것도 안 붙으면 위아래 같음

usage: python make_jobs_novel.py <출력폴더> <개수> <시드>
"""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent))
from make_jobs_diverse import RMAX, SRC_CFG, stack, validator_accepts  # noqa: E402

KINDS = ["blob", "shell", "tree", "cells", "polywall", "bars", "twins", "holey"]
KO = {"blob": "둥근 덩어리", "shell": "무작위 윤곽 벽", "tree": "갈라지는 벽", "cells": "불규칙 칸막이",
      "polywall": "꺾인 벽", "bars": "막대 조합 부품", "twins": "합쳐지거나 갈라지는 덩어리", "holey": "구멍이 생겼다 사라짐"}
SOLID = {"blob", "bars", "holey"}       # 속이 찬 형상은 넓이가 커지므로 작게 만든다
VCAP = 4.5e6                            # 전체 부피 상한(mm³). 공개 05(가장 큼)가 약 9e6
STEP = 2.0                              # 한 층에 옆으로 움직여도 되는 최대 거리(mm)


def blob_ring(r, S, n=256):
    """반지름에 2–5 주기 물결을 더한 매끈한 닫힌 곡선(점 배열)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = np.ones_like(th)
    for k in range(2, 6):
        rad += r.uniform(0, 0.28 / k * 2) * np.cos(k * th + r.uniform(0, 2 * np.pi))
    rad = S * np.clip(rad, 0.35, None) / rad.max()
    ax = r.uniform(0.55, 1.0)                                   # 한쪽으로 납작하게
    return np.column_stack([rad * np.cos(th), ax * rad * np.sin(th)])


def thick(line, t):
    return line.buffer(t / 2, 8)


def wall_t(r, cap):
    """벽 두께: 대부분 8–24 mm, 가끔 24–40 mm(연속값). 형상 크기에 비해 너무 두꺼우면 속이 차 버리므로 cap 으로 줄인다."""
    t = r.uniform(8.0, 24.0) if r.random() < 0.8 else r.uniform(24.0, 40.0)
    return float(min(t, max(8.0, cap)))


def k_blob(r, S, t):
    P0 = Polygon(blob_ring(r, S)).buffer(0)
    return lambda u, t, s: affinity.scale(P0, s, s, origin=(0, 0))


def k_shell(r, S, t):
    pts = blob_ring(r, S)
    keep = np.ones(len(pts), bool)
    for _ in range(int(r.integers(0, 3))):                    # 끊긴 곳 0–2 군데
        a, w = int(r.integers(0, len(pts))), int(len(pts) * r.uniform(0.03, 0.12))
        keep[[(a + i) % len(pts) for i in range(w)]] = False
    if keep.all():
        lines = [LineString(np.vstack([pts, pts[:1]]))]
    else:
        start = int(np.argmin(keep))                           # 끊긴 자리에서 시작해 이어진 조각을 모은다
        order = np.roll(np.arange(len(pts)), -start)
        lines, cur = [], []
        for i in order:
            if keep[i]:
                cur.append(pts[i])
            elif len(cur) > 1:
                lines.append(LineString(cur))
                cur = []
            else:
                cur = []
        if len(cur) > 1:
            lines.append(LineString(cur))
    return lambda u, t, s: unary_union([thick(affinity.scale(L, s, s, origin=(0, 0)), t) for L in lines])


def k_tree(r, S, t):
    segs = []                                                  # (선, 깊이)
    depth = int(r.integers(2, 4))

    def grow(p, ang, L, d):
        q = (p[0] + L * math.cos(ang), p[1] + L * math.sin(ang))
        segs.append((LineString([p, q]), d))
        if d < depth:
            for _ in range(int(r.integers(2, 4))):
                grow(q, ang + r.uniform(-0.9, 0.9), L * r.uniform(0.5, 0.8), d + 1)
    trunks = int(r.integers(1, 4))
    for i in range(trunks):                                    # 줄기 1–3개를 가운데에서 여러 방향으로
        grow((0.0, 0.0), 2 * np.pi * i / trunks + r.uniform(-0.4, 0.4), S * r.uniform(0.35, 0.5), 0)
    decay, tip = r.uniform(0.6, 0.85), r.uniform(6.0, 9.0)
    return lambda u, t, s: unary_union([thick(affinity.scale(L, s, s, origin=(0, 0)), max(t * decay ** d, tip))
                                        for L, d in segs])


def k_cells(r, S, t):
    P0 = Polygon(blob_ring(r, S)).buffer(0)
    pts = []
    while len(pts) < int(r.integers(4, 12)):
        q = Point(r.uniform(-S, S), r.uniform(-S, S))
        if P0.buffer(-S * 0.1).contains(q):
            pts.append(q)
    edges = shapely.voronoi_polygons(MultiPoint(pts), extend_to=P0.buffer(S), only_edges=True).intersection(P0)
    skel = unary_union([edges, P0.exterior])

    def f(u, t, s):
        sk = affinity.scale(skel, s, s, origin=(0, 0))
        return thick(sk, t).intersection(affinity.scale(P0, s, s, origin=(0, 0)).buffer(t / 2))
    return f


def k_polywall(r, S, t):
    lines = []
    for _ in range(int(r.integers(1, 4))):
        p, ang, pts = np.array([r.uniform(-S, S) * 0.5, r.uniform(-S, S) * 0.5]), r.uniform(0, 2 * np.pi), []
        pts.append(p.copy())
        for _ in range(int(r.integers(2, 7))):
            ang += math.radians(r.uniform(-110, 110))
            p = np.clip(p + S * r.uniform(0.3, 0.8) * np.array([math.cos(ang), math.sin(ang)]), -S, S)
            pts.append(p.copy())
        lines.append(LineString(pts))
    return lambda u, t, s: unary_union([thick(affinity.scale(L, s, s, origin=(0, 0)), t) for L in lines])


def k_bars(r, S, t):
    bars = []
    for _ in range(int(r.integers(2, 7))):                    # 중심선 + 폭(폭은 크기를 바꿔도 그대로)
        c, L, ang = r.uniform(-S, S, 2) * 0.5, S * r.uniform(0.4, 1.6), r.uniform(0, np.pi)
        d = 0.5 * L * np.array([math.cos(ang), math.sin(ang)])
        bars.append((LineString([c - d, c + d]), r.uniform(8.0, 60.0)))
    body0 = unary_union([L.buffer(w / 2, cap_style="flat") for L, w in bars])
    holes = []
    for _ in range(int(r.integers(0, 5))):
        hr = r.uniform(5.0, 25.0)
        inner = body0.buffer(-(hr + 8.0))                      # 구멍 둘레에 8 mm 이상 남긴다
        if inner.is_empty:
            continue
        x0, y0, x1, y1 = inner.bounds
        for _ in range(50):
            q = Point(r.uniform(x0, x1), r.uniform(y0, y1))
            if inner.contains(q):
                holes.append((q, hr))
                break

    def f(u, t, s):
        body = unary_union([affinity.scale(L, s, s, origin=(0, 0)).buffer(w / 2, cap_style="flat") for L, w in bars])
        cut = [affinity.scale(q, s, s, origin=(0, 0)).buffer(hr, 32) for q, hr in holes]
        return body.difference(unary_union(cut)) if cut else body
    return f


def k_twins(r, S, t, n, solid):
    m = int(r.integers(2, 4))
    parts = []
    for _ in range(m):
        Si = S * r.uniform(0.3, 0.55)
        ring = blob_ring(r, Si)
        parts.append(Polygon(ring).buffer(0) if solid else thick(LineString(np.vstack([ring, ring[:1]])), t))
    merge = r.random() < 0.6                                   # 합쳐짐(아니면 갈라짐)
    ang0 = r.uniform(0, 2 * np.pi)
    far = np.array([S * 0.55 * np.array([math.cos(ang0 + 2 * np.pi * i / m), math.sin(ang0 + 2 * np.pi * i / m)])
                    for i in range(m)])
    near = far * r.uniform(0.0, 0.25)
    move = np.linalg.norm(far - near, axis=1).max()
    if move > STEP * max(n - 1, 1):                            # 한 층에 STEP 보다 많이 움직이지 않게
        near = far - (far - near) * STEP * max(n - 1, 1) / move

    def f(u, t, s):
        a, b = (far, near) if merge else (near, far)
        c = a + (b - a) * u
        return unary_union([affinity.translate(affinity.scale(p, s, s, origin=(0, 0)), *ci) for p, ci in zip(parts, c)])
    return f, ("합쳐짐" if merge else "갈라짐") + (" 속참" if solid else " 벽")


def k_holey(r, S, t):
    P0 = Polygon(blob_ring(r, S)).buffer(0)
    holes = []
    for _ in range(int(r.integers(1, 5))):
        hmax = r.uniform(8.0, S * 0.35)
        inner = P0.buffer(-(hmax + 10.0))
        if inner.is_empty:
            continue
        x0, y0, x1, y1 = inner.bounds
        for _ in range(50):
            q = Point(r.uniform(x0, x1), r.uniform(y0, y1))
            if inner.contains(q):
                a = r.uniform(0.0, 0.6)
                holes.append((q, hmax, a, min(1.0, a + r.uniform(0.3, 0.8))))
                break

    def f(u, t, s):
        g = affinity.scale(P0, s, s, origin=(0, 0))
        cut = []
        for q, hmax, a, b in holes:
            if a < u < b:
                hr = hmax * math.sin(math.pi * (u - a) / (b - a))
                if hr > 2.0:
                    cut.append(affinity.scale(q, s, s, origin=(0, 0)).buffer(hr, 32))
        return g.difference(unary_union(cut)) if cut else g
    return f


CAP = {"shell": 0.2, "tree": 0.15, "cells": 0.12, "polywall": 0.2, "twins": 0.15}   # 벽 두께 ≤ 이 비율 × 크기


def make_one(r, kind):
    solid = kind in SOLID or (kind == "twins" and r.random() < 0.5)
    if solid:
        S = r.uniform(40, 160) if kind != "twins" else r.uniform(60, 160)
    elif kind in ("cells", "twins"):                           # 칸·덩어리가 보이게 조금 크게
        S = r.uniform(120, 380) if kind == "cells" else r.uniform(100, 300)
    else:
        S = float(r.choice([r.uniform(50, 120), r.uniform(120, 230), r.uniform(230, 380)]))
    n = int(r.integers(4, 61))
    cap = CAP.get(kind, 1.0) * S
    t0 = wall_t(r, cap)
    extra = ""
    if kind == "twins":
        f, extra = k_twins(r, S, t0, n, solid)
    else:
        f = {"blob": k_blob, "shell": k_shell, "tree": k_tree, "cells": k_cells, "polywall": k_polywall,
             "bars": k_bars, "holey": k_holey}[kind](r, S, t0)
    pool = ["taper", "twist", "drift"] + ([] if kind in SOLID else ["thicken"])
    modes = [m for m in pool if r.random() < 0.35]
    span = max(n - 1, 1)
    s0 = s1 = 1.0
    if "taper" in modes:                                       # 바깥 끝이 한 층에 STEP 넘게 움직이지 않게
        lim = STEP * span / S
        s0, s1 = r.uniform(0.7, 1.0), r.uniform(0.7, 1.3)
        if abs(s1 - s0) > lim:
            s1 = s0 + math.copysign(lim, s1 - s0)
    tw = 0.0
    if "twist" in modes:
        tw = math.copysign(min(r.uniform(20, 120), math.degrees(STEP * span / (S * max(s0, s1)))), r.uniform(-1, 1))
    dr = np.zeros(2)
    if "drift" in modes:
        a = r.uniform(0, 2 * np.pi)
        dr = min(r.uniform(10, 60), 0.75 * STEP * span) * np.array([math.cos(a), math.sin(a)])
    t1 = wall_t(r, cap) if "thicken" in modes else t0
    secs = []
    for k in range(n):
        u = k / span
        g = f(u, t0 + (t1 - t0) * u, s0 + (s1 - s0) * u)
        if tw:
            g = affinity.rotate(g, tw * u, origin=(0, 0))
        g = affinity.translate(g, *(dr * u)).buffer(0)
        secs.append(g)
    area = np.array([g.area for g in secs])
    if area.min() < 100:
        return None, ""
    if area.sum() * 2.0 > VCAP:                                # 너무 크면 층 수를 줄인다(위층을 버림)
        n = max(4, int(VCAP / (area.mean() * 2.0)))
        secs = secs[:n]
    b = unary_union(secs).bounds
    w = max(b[2] - b[0], b[3] - b[1])
    if w > 2 * RMAX * 0.95:
        return None, ""
    lim = max(0.0, RMAX * 0.9 - w / 2)
    cx, cy = r.uniform(-1, 1) * lim * 0.5, r.uniform(-1, 1) * lim * 0.5
    off = (cx - (b[0] + b[2]) / 2, cy - (b[1] + b[3]) / 2)
    mesh = stack(secs, [1] * len(secs), [off] * len(secs))
    th = f"두께{t0:.1f}" + (f"→{t1:.1f}" if t1 != t0 else "")
    desc = f"{kind}{('/' + extra) if extra else ''} {'+'.join(modes) or 'same'} 폭~{w:.0f} {th} {len(secs)}층"
    return mesh, desc


def make(out: Path, n: int, seed: int):
    r = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    order = [KINDS[i % len(KINDS)] for i in range(n)]          # 종류가 골고루 나오게 돌아가며
    r.shuffle(order)
    for i, kind in enumerate(order):
        mesh, desc = None, ""
        d = out / f"{i + 1:02d}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC_CFG, d / "config.yaml")
        for _ in range(30):
            mesh, desc = make_one(r, kind)
            if mesh is None or mesh.bounds[1, 2] < 4:
                continue
            mesh.export(d / "target.stl")
            if validator_accepts(d):                           # 채점기가 실제로 읽고 모든 층이 비어 있지 않아야 채택
                break
        (d / "desc.txt").write_text(desc, encoding="utf-8")
        print(f"{d.name}: [{KO[kind]}] {desc}", flush=True)


if __name__ == "__main__":
    make(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
