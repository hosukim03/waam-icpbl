# -*- coding: utf-8 -*-
"""학습용 무작위 형상 생성 (공개 6개·벤치마크 10개와 겹치지 않게 따로 만든다).

v2 (2026-09-16): 층마다 단면이 변하는 형상을 넣었다.
  공개 04·05·06 은 층마다 단면이 35~86% 변하고 조각도 1~4개로 갈라지는데,
  v1 학습 형상은 전부 기둥 모양(변화 0%)이라 그 상황을 배우지 못했다.

형상 종류
  - 기둥형(prism): 단면이 그대로 (둥근 벽, 격자, 살+테두리, 이중 링, 별, 나선, 막대)
  - 원뿔형(taper): 위로 갈수록 좁아지거나 넓어짐
  - 분리형(split): 위로 가면서 한 덩어리가 여러 조각으로 갈라짐
  - 구멍형(hole): 중간 높이에서 구멍이 생겼다 사라짐
  - 비틀림(twist): 층마다 조금씩 회전

usage: python make_train_jobs.py <출력폴더> [개수] [시드]
"""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

SRC_CFG = Path(__file__).resolve().parents[1] / "WAAM_Validator" / "tests" / "01" / "config.yaml"
WALL = 8.0
LAYER_H = 2.0
RMAX = 690.0          # 작업대 반경 700 - 비드 절반 여유


# ---------------------------------------------------------------- 단면 모양들
def ring(R, w=WALL):
    return Point(0, 0).buffer(R, 96).difference(Point(0, 0).buffer(max(R - w, 1.0), 96))


def blob(rng, R):
    m = int(rng.integers(3, 7))
    th = np.linspace(0, 2 * math.pi, 240)
    r = R * (1 + 0.22 * np.sin(m * th + rng.uniform(0, 6.28)))
    return LineString(np.column_stack([r * np.cos(th), r * np.sin(th)])).buffer(WALL / 2, 16)


def grid_walls(rng, R):
    n = int(rng.integers(2, 4))
    xs = np.linspace(-R, R, n + 1)
    lines = [LineString([(x, -R), (x, R)]) for x in xs] + [LineString([(-R, y), (R, y)]) for y in xs]
    return unary_union([ln.buffer(WALL / 2, 8) for ln in lines])


def spokes(rng, R):
    k = int(rng.integers(3, 7))
    lines = [LineString([(0, 0), (R * math.cos(a), R * math.sin(a))])
             for a in np.linspace(0, 2 * math.pi, k, endpoint=False)]
    return unary_union([ln.buffer(WALL / 2, 8) for ln in lines] + [ring(R)])


def star(rng, R):
    k = int(rng.integers(5, 9))
    th = np.linspace(0, 2 * math.pi, 2 * k + 1)
    r = np.where(np.arange(2 * k + 1) % 2 == 0, R, R * 0.45)
    return LineString(np.column_stack([r * np.cos(th), r * np.sin(th)])).buffer(WALL / 2, 8)


def spiral(rng, R):
    turns = float(rng.uniform(2.0, 3.5))
    th = np.linspace(0, 2 * math.pi * turns, 600)
    r = np.linspace(R * 0.12, R, th.size)
    return LineString(np.column_stack([r * np.cos(th), r * np.sin(th)])).buffer(WALL / 2, 8)


def bars(rng, R):
    n = int(rng.integers(3, 6))
    ys = np.linspace(-R, R, n)
    return unary_union([LineString([(-R, y), (R, y)]).buffer(WALL / 2, 8) for y in ys])


def double_ring(rng, R):
    return unary_union([ring(R), ring(R * 0.55)])


SHAPES = [blob, grid_walls, spokes, star, spiral, bars, double_ring]



# ---------------------------------------------------------------- 층을 옆면으로 이어 붙이기(로프트)
def _rings(poly):
    out = [np.asarray(poly.exterior.coords)[:-1]]
    out += [np.asarray(r.coords)[:-1] for r in poly.interiors]
    return out


def loft(polys, z0=0.0, h=LAYER_H):
    """층별 단면(구조가 같아야 함)을 옆면으로 이어 방수 메시를 만든다."""
    comps = [list(getattr(p, "geoms", [p])) for p in polys]
    if len({len(c) for c in comps}) != 1:
        return None
    meshes = []
    for ci in range(len(comps[0])):
        rings = [_rings(c[ci]) for c in comps]
        if len({len(r) for r in rings}) != 1:
            return None
        if any(len(rings[0][j]) != len(rings[k][j]) for k in range(len(rings)) for j in range(len(rings[0]))):
            return None
        V, F = [], []
        nz = len(rings)
        offs = []
        for k, rs in enumerate(rings):                      # 단면마다 모든 링의 점을 쌓는다
            o = []
            for r in rs:
                o.append(len(V))
                V += [(float(x), float(y), z0 + k * h) for x, y in r]
            offs.append(o)
        for k in range(nz - 1):
            for j in range(len(rings[0])):
                n = len(rings[k][j])
                a, b = offs[k][j], offs[k + 1][j]
                for t in range(n):
                    t2 = (t + 1) % n
                    F += [[a + t, a + t2, b + t2], [a + t, b + t2, b + t]]
        for k, flip in ((0, True), (nz - 1, False)):        # 아래·위 뚜껑
            cap = trimesh.creation.triangulate_polygon(comps[k][ci], engine="earcut")
            cv, cf = cap
            base = len(V)
            z = z0 + k * h
            V += [(float(x), float(y), z) for x, y in cv]
            cf = np.asarray(cf)
            F += [list(f[::-1] + base) if flip else list(f + base) for f in cf]
        m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F, int), process=True)
        if m.volume < 0:                                    # 면 방향이 뒤집혔으면 되돌린다
            m.invert()
        meshes.append(m)
    return trimesh.util.concatenate(meshes)

# ---------------------------------------------------------------- 층 변화 방식
def prism(rng, fn, R, k, n):
    return fn(rng, R)


def taper(rng, fn, R, k, n):
    """위로 갈수록 좁아지거나(또는 넓어지거나) 한다."""
    s = 1.0 - 0.55 * k / max(n - 1, 1)
    if getattr(taper, "grow", False):
        s = 0.45 + 0.55 * k / max(n - 1, 1)
    return affinity.scale(fn(rng, R), s, s, origin=(0, 0))


def split(rng, fn, R, k, n):
    """아래는 한 덩어리, 위로 가면서 여러 조각으로 갈라진다."""
    t = k / max(n - 1, 1)
    d = R * 0.9 * t
    base = affinity.scale(fn(rng, R), 0.55, 0.55, origin=(0, 0))
    parts = [affinity.translate(base, d * math.cos(a), d * math.sin(a))
             for a in (0.0, 2.094, 4.189)]
    return unary_union(parts)


def hole(rng, fn, R, k, n):
    """얇은 링의 반지름이 층마다 커졌다 작아진다 (가운데가 비었다 찼다 하는 상황)."""
    t = k / max(n - 1, 1)
    r = R * (0.3 + 0.7 * math.sin(math.pi * t) ** 2)
    return ring(max(r, WALL * 2))


def twist(rng, fn, R, k, n):
    return affinity.rotate(fn(rng, R), 90.0 * k / max(n - 1, 1), origin=(0, 0))


MODES = [prism, taper, split, hole, twist]


def make(out: Path, n_jobs: int, seed: int):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n_jobs):
        fn = SHAPES[i % len(SHAPES)]
        mode = MODES[(i // len(SHAPES) + i) % len(MODES)]
        R = float(rng.choice([110, 160, 240, 330, 430, 560]))
        n = int(rng.integers(6, 13))                        # 층 수 6~12
        taper.grow = bool(rng.integers(0, 2))
        sub = rng.integers(0, 2 ** 31)
        polys = []
        for k in range(n + 1):                              # 단면 n+1 개 → 층 n 개
            poly = mode(np.random.default_rng(sub), fn, R, k, n + 1)
            poly = poly.intersection(Point(0, 0).buffer(RMAX, 128))
            polys.append(poly)
        mesh = loft(polys)
        if mesh is None or not mesh.is_watertight:          # 층 구조가 달라 못 이으면 기둥형으로
            mode = prism
            poly = fn(np.random.default_rng(sub), R).intersection(Point(0, 0).buffer(RMAX, 128))
            mesh = trimesh.util.concatenate(
                [trimesh.creation.extrude_polygon(g, n * LAYER_H) for g in getattr(poly, "geoms", [poly])])
        d = out / f"{i + 1:02d}"
        d.mkdir(parents=True, exist_ok=True)
        mesh.export(d / "target.stl")
        shutil.copyfile(SRC_CFG, d / "config.yaml")
        print(f"{d.name}: {fn.__name__:11s} {mode.__name__:6s} R={R:4.0f} 층={n:2d} watertight={mesh.is_watertight}")


if __name__ == "__main__":
    make(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 14,
         int(sys.argv[3]) if len(sys.argv) > 3 else 11)
