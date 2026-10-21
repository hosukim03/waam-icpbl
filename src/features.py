# -*- coding: utf-8 -*-
"""형상 특징 — 포트폴리오가 어떤 설정을 돌릴지 고를 때 쓰는 값(계산 1초 안팎).

핵심 특징은 '로봇 두 대가 동시에 설 수 있는 비율'이다. 채점기 규칙상 두 팔(베이스→토치 선분)은
required(기본 250mm) 이상 떨어져야 하므로, 형상이 작으면 두 토치가 형상 위에 동시에 있을 수 없다.
단면 위의 점을 고르게 뽑아 두 로봇에 하나씩 놓는 모든 조합 가운데 규칙을 지키는 조합의 비율을 구한다.
  0 이면 한 번에 한 대만 적층할 수 있는 형상(1대 설정이 정답), 클수록 여러 대가 나눠 일할 수 있다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely

from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

N_LAYERS = 5        # 특징 계산에 쓰는 층 수(바닥~꼭대기 고르게)
N_POINTS = 250      # 층마다 뽑는 표본점 수


def seg_dist(P1, Q1, P2, Q2):
    """2D 선분 P1-Q1 과 P2-Q2 사이 최단거리(배열 브로드캐스트)."""
    d1, d2, r = Q1 - P1, Q2 - P2, P1 - P2
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.where(den > 1e-12, den, 1.0), 0, 1), 0.0)
    t = (b * s + f) / np.maximum(e, 1e-12)
    s = np.where(t < 0, np.clip(-c / np.maximum(a, 1e-12), 0, 1),
                 np.where(t > 1, np.clip((b - c) / np.maximum(a, 1e-12), 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm((P1 + d1 * s[..., None]) - (P2 + d2 * t[..., None]), axis=-1)


def _points(poly, n: int) -> np.ndarray:
    """단면 안의 점을 격자로 고르게 뽑는다(결정적)."""
    minx, miny, maxx, maxy = poly.bounds
    area = max(poly.area, 1.0)
    step = max(2.0, float(np.sqrt(area / n)))
    g = np.mgrid[minx + step / 2:maxx:step, miny + step / 2:maxy:step].reshape(2, -1).T
    pts = g[shapely.contains_xy(poly, g[:, 0], g[:, 1])] if len(g) else g
    if len(pts) < 8:                                   # 얇은 벽: 격자가 빗나가면 경계 위 점을 쓴다
        b = poly.boundary
        pts = np.array([b.interpolate(d).coords[0] for d in np.linspace(0, b.length, n, endpoint=False)])
    if len(pts) > n:
        pts = pts[np.linspace(0, len(pts) - 1, n).astype(int)]
    return pts


def shape_features(job: Path) -> dict:
    cfg = load_config(Path(job) / "config.yaml")
    mesh = load_target_mesh(Path(job) / "target.stl", cfg)
    pr = cfg.process
    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    ks = sorted({int(round(x)) for x in np.linspace(0, n - 1, min(N_LAYERS, n))})
    layers = slice_target_layers(mesh, ks, cfg)
    robots = sorted(cfg.robots, key=lambda r: r.id)
    bases = [np.array(r.base_xyz_mm[:2], float) for r in robots]
    reach = [float(r.xy_reach_radius_mm) for r in robots]
    need = max(r.arm_envelope_radius_mm for r in robots) * 2 + cfg.collision.arm_clearance_mm
    need = max(need, max(r.tcp_radius_mm for r in robots) * 2)

    pair, single, span = [], [], []
    for k in ks:
        g = layers[k]
        if g.is_empty:
            continue
        minx, miny, maxx, maxy = g.bounds
        span.append(max(maxx - minx, maxy - miny))
        pts = _points(g, N_POINTS)
        ok = [np.linalg.norm(pts - b, axis=1) <= rr - 10.0 for b, rr in zip(bases, reach)]
        single.append(max(float(o.mean()) for o in ok))           # 한 대가 닿는 비율(가장 잘 닿는 로봇)
        best = 0.0
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                P, Q = pts[ok[i]], pts[ok[j]]
                if len(P) == 0 or len(Q) == 0:
                    continue
                d = seg_dist(bases[i][None, None], P[:, None], bases[j][None, None], Q[None, :])
                best = max(best, float((d >= need).mean()))
        pair.append(best)
    return {"layers": n,
            "span": float(max(span)) if span else 0.0,           # 단면의 가로·세로 중 큰 값(최대)
            "pair": float(np.mean(pair)) if pair else 0.0,       # 두 대가 동시에 설 수 있는 조합 비율(층 평균)
            "pair_max": float(max(pair)) if pair else 0.0,
            "reach1": float(min(single)) if single else 0.0}     # 한 대로 닿는 비율(가장 불리한 층)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(shape_features(Path(sys.argv[1]))))
