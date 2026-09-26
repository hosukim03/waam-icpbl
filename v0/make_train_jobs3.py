# -*- coding: utf-8 -*-
"""학습용 형상 v3 — 공개 05번·04번을 닮은 '비선형·무작위' 형상 생성기.

v2 의 한계(사용자 지적): 크기·회전이 층 번호에 비례(선형)해서 변해 너무 쉬웠다.
v3 는 공개 모델에서 시간이 가장 오래 걸리는 두 종류를 무작위로 흉내 낸다.

1) vase (05번 계열): 두꺼운 벽의 꽃병
   단면 = 꽃잎 m개(2~5)인 닫힌 곡선, 벽 두께 8/16/24mm
   높이에 따라 ① 반지름(허리가 들어가거나 불룩), ② 꽃잎 깊이, ③ 회전, ④ 타원 비율이
   각각 무작위 푸리에 곡선으로 변한다 → 층별 변화가 비선형이고 형상마다 다르다.
2) plate (04번 계열): 판 + 원통 기둥들
   무작위 윤곽의 판(가운데 구멍 가능) 위에 원통 기둥 2~6개. 기둥 높이가 제각각이라
   올라갈수록 조각 수가 바뀐다(한 덩어리 → 여러 조각 → 점점 줄어듦).

학습 시간 때문에 층 수는 30~90층으로 줄였다(05번은 350층). 대신 변화 모양은 같은 성격.
usage: python make_train_jobs3.py <출력폴더> [개수] [시드]
"""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon

SRC_CFG = Path(__file__).resolve().parents[1] / "WAAM_Validator" / "tests" / "01" / "config.yaml"
LAYER_H = 2.0
RMAX = 690.0
NTH = 240                       # 단면 둘레 점 개수(모든 층 동일 → 옆면으로 이어 붙일 수 있음)


def fourier(rng, n_terms=3, amp=1.0):
    """t∈[0,1] 에서 부드럽고 무작위인 비선형 곡선 f(t) (평균 0 근처)."""
    ks = np.arange(1, n_terms + 1)
    a = rng.normal(0, amp, n_terms) / ks
    p = rng.uniform(0, 2 * math.pi, n_terms)
    return lambda t: float(np.sum(a * np.sin(math.pi * ks * t + p)))


# ---------------------------------------------------------------- 옆면 잇기
def loft_rings(outer, inner, h=LAYER_H):
    """outer[k], inner[k]: (NTH,2) 점열. 벽(바깥-안쪽) 단면을 층마다 이어 방수 메시로."""
    nz = len(outer)
    V, F = [], []
    for k in range(nz):
        z = k * h
        V += [(x, y, z) for x, y in outer[k]]
        V += [(x, y, z) for x, y in inner[k]]
    m = NTH

    def idx(k, ring, t):
        return k * 2 * m + ring * m + (t % m)

    for k in range(nz - 1):
        for t in range(m):
            for ring in (0, 1):
                a, b = idx(k, ring, t), idx(k, ring, t + 1)
                c, d = idx(k + 1, ring, t + 1), idx(k + 1, ring, t)
                if ring == 0:
                    F += [[a, b, c], [a, c, d]]
                else:
                    F += [[a, c, b], [a, d, c]]
    for k, top in ((0, False), (nz - 1, True)):        # 아래·위 고리 뚜껑
        for t in range(m):
            o0, o1 = idx(k, 0, t), idx(k, 0, t + 1)
            i0, i1 = idx(k, 1, t), idx(k, 1, t + 1)
            if top:
                F += [[o0, o1, i1], [o0, i1, i0]]
            else:
                F += [[o0, i1, o1], [o0, i0, i1]]
    mesh = trimesh.Trimesh(np.asarray(V, float), np.asarray(F, int), process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


# ---------------------------------------------------------------- 05번 계열
MAX_SHIFT = 3.0                 # 층 사이 벽 이동 한계(mm) — 위층 비드가 아래층에 걸치도록


def vase(rng):
    n = int(rng.integers(60, 151))
    m = int(rng.choice([2, 3, 3, 4, 5]))
    wall = float(rng.choice([8.0, 16.0, 24.0]))
    R0 = float(rng.uniform(180, 380))
    waist = float(rng.uniform(-0.35, 0.25))            # 음수 = 허리가 들어감, 양수 = 불룩
    fR = fourier(rng, 3, 0.08)
    fA = fourier(rng, 2, 0.05)
    fT = fourier(rng, 3, 0.25)
    fE = fourier(rng, 2, 0.06)
    lobe0 = float(rng.uniform(0.04, 0.16))
    twist = float(rng.uniform(-0.9, 0.9))              # 라디안 (05번은 완만한 회전)
    th = np.linspace(0, 2 * math.pi, NTH, endpoint=False)
    outer, inner = [], []
    for k in range(n + 1):
        t = k / n
        R = R0 * (1 + waist * math.sin(math.pi * t) ** 1.5 + fR(t))
        lobe = max(0.0, lobe0 + fA(t))
        phi = twist * t + fT(t)
        ell = 1 + fE(t)
        r = R * (1 + lobe * np.cos(m * (th - phi)))
        x, y = r * np.cos(th) * ell, r * np.sin(th) / ell
        rr = np.hypot(x, y)
        s = min(1.0, (RMAX - wall / 2) / rr.max())      # 작업대 밖으로 나가지 않게
        x, y = x * s, y * s
        # 곡선의 바깥·안쪽 오프셋(법선 방향)
        dx, dy = np.gradient(x), np.gradient(y)
        nl = np.hypot(dx, dy) + 1e-9
        nx, ny = dy / nl, -dx / nl
        if np.mean(nx * x + ny * y) < 0:
            nx, ny = -nx, -ny
        outer.append(np.column_stack([x + nx * wall / 2, y + ny * wall / 2]))
        inner.append(np.column_stack([x - nx * wall / 2, y - ny * wall / 2]))
    shift = max(float(np.max(np.hypot(*(outer[k + 1] - outer[k]).T))) for k in range(n))
    if shift > MAX_SHIFT:                              # 층 사이 이동이 크면 버리고 다시 뽑는다
        return None, f"이동 {shift:.1f}mm 초과"
    mesh = loft_rings(outer, inner)
    return mesh, f"층간이동≤{shift:.1f}mm vase  꽃잎{m} 벽{wall:.0f} R{R0:.0f} 허리{waist:+.2f} 비틀림{math.degrees(twist):+.0f}° {n}층"


# ---------------------------------------------------------------- 04번 계열
def plate(rng):
    R0 = float(rng.uniform(160, 330))
    th = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    rr = R0 * (1 + 0.12 * np.sin(int(rng.integers(2, 5)) * th + rng.uniform(0, 6)))
    body = Polygon(np.column_stack([rr * np.cos(th), rr * np.sin(th)]))
    if rng.random() < 0.7:
        body = body.difference(Point(0, 0).buffer(R0 * rng.uniform(0.2, 0.45), 64))
    pl = int(rng.integers(3, 11))                      # 판 두께(층)
    parts = [trimesh.creation.extrude_polygon(body, pl * LAYER_H)]
    k = int(rng.integers(2, 7))
    placed = []
    for _ in range(200):                               # 기둥 자리를 여러 번 뽑아 k개 채운다
        if len(placed) >= k:
            break
        a = float(rng.uniform(0, 2 * math.pi))
        ro = float(rng.uniform(25, 70))
        ri = float(rng.uniform(0.3, 0.7)) * ro
        rad = float(rng.uniform(0.2, 0.9)) * R0
        c = (rad * math.cos(a), rad * math.sin(a))
        if any(math.hypot(c[0] - p[0], c[1] - p[1]) < ro + p[2] + 12 for p in placed):
            continue
        if not body.buffer(-2).contains(Point(c).buffer(ro)):
            continue
        placed.append((c[0], c[1], ro))
        hk = int(rng.integers(5, 45))                  # 기둥마다 높이가 다름
        tube = Point(c).buffer(ro, 64).difference(Point(c).buffer(ri, 64))
        m = trimesh.creation.extrude_polygon(tube, hk * LAYER_H - 0.01)   # 전체 높이는 그대로
        m.apply_translation([0, 0, pl * LAYER_H + 0.01])   # 판과 맞닿지 않게 틈
        parts.append(m)
    if len(placed) < 2:
        return None, "기둥 부족"
    mesh = trimesh.util.concatenate(parts)
    return mesh, f"plate 판{pl}층 R{R0:.0f} 기둥{len(placed)}개"


def make(out: Path, n_jobs: int, seed: int):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n_jobs):
        fam = vase if i % 3 != 2 else plate            # 꽃병 2 : 판 1
        for _ in range(400):
            mesh, desc = fam(rng)
            if mesh is not None and mesh.is_watertight:
                break
        d = out / f"{i + 1:02d}"
        d.mkdir(parents=True, exist_ok=True)
        mesh.export(d / "target.stl")
        shutil.copyfile(SRC_CFG, d / "config.yaml")
        print(f"{d.name}: {desc}  watertight={mesh.is_watertight}  높이 {mesh.bounds[1, 2]:.0f}mm")


if __name__ == "__main__":
    make(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 15,
         int(sys.argv[3]) if len(sys.argv) > 3 else 23)
