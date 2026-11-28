# -*- coding: utf-8 -*-
"""확인용 형상 생성기(2026-11-28): 연습 1·2·3 생성기에 없는 방식으로, 수식으로 정한 3D 모양을 층마다 잘라 만든다.
사용자 지적(2026-10-27, 11-28): 시드만 바꾼 같은 생성기 묶음은 겉모습이 비슷해서 일반화 확인이 안 된다 → 다른 방식.
연습 1(make_jobs_diverse.py)은 이름을 정한 21종류, 연습 2(make_jobs_novel.py)는 무작위 2D 윤곽을 밀어 올림,
연습 3(make_jobs_parts.py)은 실제 부품 11종류다. 여기서는 3D 공간에서 정한 모양을 층마다 자른다.
방식(첫 낱말이 desc.txt 의 첫 낱말, KO 는 보이는 이름)
  lattice : 수식 3D 격자(자이로이드, sinX·cosY + sinY·cosZ + sinZ·cosX = 0 곡면을 두께 있는 벽으로). 바깥 윤곽 안을
            채우고 바깥 껍데기 벽이 있을 수도 없을 수도 있다. 층마다 무늬가 조금씩 계속 바뀐다(벌집·격자는 층마다 같음).
            처마를 줄이려고 세로로 2~3배 늘인다.
  blobs   : 3D 덩어리를 녹여 붙인 모양. 위아래로 긴 타원체 여러 개를 흩어 합치고 이음매를 둥글린다. 높이에 따라
            덩어리가 합쳐지고 갈라져 층마다 섬 개수·위치가 바뀐다. 속찬 것과 껍데기만 있는 것.
  helix   : 휘감아 올라가는 관. 관 1~4가닥이 원을 그리며 올라간다(반지름이 위로 가며 바뀌거나 가운데 기둥이 있을 수
            있음). 단면이 층마다 자리를 옮긴다.
모든 방식에 공통
  - 처마: 층마다 아래층을 STEP(2 mm) 부풀린 범위 안만 남긴다(한 층에 옆으로 2 mm 넘게 튀어나가지 않음, 공중에 뜬
    섬 없음). 연습 2·3 과 같은 한도다.
  - 비드(폭 8 mm)가 못 들어가는 가는 곳은 뺀다(단면을 비드 반지름만큼 깎았다 되살림). 그래도 채울 수 있는 넓이가
    99% 미만인 층이 있으면 버린다(make_jobs_parts.fill_ratio).
  - 부피 상한·작업대 안에 들어감·채점기가 읽고 모든 층이 비어 있지 않음은 연습 3 과 같다.
usage: python make_jobs_formula.py <출력폴더> <개수> <시드>   (WAAM venv 파이썬: 채점기로 읽어 보고 넣는다)
"""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely import affinity, constrained_delaunay_triangles
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent))
from make_jobs_diverse import GAP, H, RMAX, SRC_CFG, validator_accepts  # noqa: E402
from make_jobs_parts import FILL_MIN, RB, STEP, VCAP, disc, rbox  # noqa: E402

KINDS = ["lattice", "blobs", "helix"]
KO = {"lattice": "수식 3D 격자", "blobs": "녹여 붙인 덩어리", "helix": "휘감아 올라가는 관"}
RES = 2.0             # 격자 곡면을 잴 칸 크기(mm). 1.5 는 층마다 0.2 s 더 들고 결과 모양은 거의 같다


def fill_ok(g) -> bool:
    """비드로 넘치지 않게 채울 수 있는 넓이가 99% 이상인가(make_jobs_parts.fill_ratio 와 같은 계산, 원은 8등분 근사)."""
    return not g.is_empty and g.buffer(-RB * 0.999, 8).buffer(RB * 0.999, 8).area >= FILL_MIN * g.area


def closed(m) -> bool:
    """채점기가 STL 을 읽을 때처럼 점을 합치고 납작한·겹친 면을 뺀 뒤에도 닫힌 덩어리인가
    (waam_validator.shape.target.load_target_mesh. 닫혀 있지 않으면 채점기가 형상을 받지 않는다)."""
    m = m.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    return m.is_watertight


def prism(poly, h):
    """다각형을 높이 h 기둥으로. 윗면·아랫면은 구멍을 지키는 삼각형 나누기(constrained Delaunay)로 만든다.
    trimesh 의 extrude_polygon(earcut)은 구멍이 많은 격자 단면에서 구멍을 잇는 폭 0 다리를 만들어, 채점기가 점을
    합치고 납작한 면을 빼면 덩어리가 열린다(2026-11-28 격자 44층 중 3층)."""
    poly = orient(poly, 1.0)                                     # 바깥 고리 반시계, 구멍 시계 → 옆면이 바깥을 본다
    idx, V, Fc = {}, [], []

    def vid(x, y, z):
        if (x, y, z) not in idx:
            idx[(x, y, z)] = len(V)
            V.append((x, y, z))
        return idx[(x, y, z)]
    for t in constrained_delaunay_triangles(poly).geoms:
        (x0, y0), (x1, y1), (x2, y2) = list(t.exterior.coords)[:3]
        if (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0) < 0:
            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        Fc.append((vid(x0, y0, h), vid(x1, y1, h), vid(x2, y2, h)))
        Fc.append((vid(x0, y0, 0.0), vid(x2, y2, 0.0), vid(x1, y1, 0.0)))
    for ring in [poly.exterior, *poly.interiors]:
        c = list(ring.coords)[:-1]
        for (ax, ay), (bx, by) in zip(c, c[1:] + c[:1]):
            a0, b0, a1, b1 = vid(ax, ay, 0.0), vid(bx, by, 0.0), vid(ax, ay, h), vid(bx, by, h)
            Fc += [(a0, b0, b1), (a0, b1, a1)]
    return trimesh.Trimesh(np.array(V, float), np.array(Fc), process=False)


def extrude_safe(p, h):
    """다각형 하나를 높이 h 로 밀어 올린다. 채점기 기준으로 닫혀 있지 않으면 윤곽을 조금씩 더 단순하게 해 다시."""
    for tol in (0.05, 0.3, 0.8):
        q = p.simplify(tol, preserve_topology=True)
        if q.is_empty or q.geom_type != "Polygon" or q.area < 30:
            return None
        try:
            m = prism(q, h)
        except Exception:                                        # noqa: BLE001
            continue
        if m.volume > 0 and closed(m):
            return m
    return None


def stack_safe(secs):
    """층마다 단면을 밀어 올려 쌓는다(층마다 따로 닫힌 덩어리, 위층과는 GAP 틈). 점 하나로 맞닿은 조각은
    0.05 mm 깎아 떼어 놓는다(합친 점에 면이 셋 이상 걸리면 닫힌 덩어리가 아니다). 한 층이라도 비면 None."""
    parts, z = [], 0.0
    for g in secs:
        g = g.buffer(-0.05, 4).buffer(0)
        n0 = len(parts)
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon" or p.area < 30:
                continue
            m = extrude_safe(p, H - GAP)
            if m is not None:
                m.apply_translation([0, 0, z])
                parts.append(m)
        if len(parts) == n0:
            return None
        z += H
    return trimesh.util.concatenate(parts)


def clean(g):
    """면만, 30 mm² 미만 조각은 뺀다."""
    g = g.buffer(0)
    ps = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area >= 30]
    return unary_union(ps) if ps else Polygon()


def opening(g):
    """비드(지름 8 mm)가 못 들어가는 가는 곳 빼기 = 비드 반지름만큼 깎았다 되살리기."""
    return clean(g.buffer(-RB, 8).buffer(RB, 8)) if not g.is_empty else g


def finish(secs):
    """가는 곳 빼기 + 처마 자르기(아래층을 STEP 만큼 부풀린 범위 안만 남김)."""
    out = []
    for g in secs:
        g = opening(g)
        if out and not g.is_empty:
            g = opening(g.intersection(out[-1].buffer(STEP, 8)))
        out.append(g)
    return out


def raster_poly(mask, xs, ys):
    """참인 칸 → 다각형. 행마다 이어진 칸을 사각형 하나로 묶어 합치고, 계단은 살짝 둥글린다."""
    rects = []
    for j in range(mask.shape[0]):
        d = np.diff(np.concatenate([[0], mask[j].astype(np.int8), [0]]))
        y = ys[j]
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            rects.append(box(xs[s] - RES / 2, y - RES / 2, xs[e - 1] + RES / 2, y + RES / 2))
    if not rects:
        return Polygon()
    g = unary_union(rects)
    return g.buffer(RES * 0.6, 8).buffer(-RES * 0.6, 8).simplify(RES * 0.3).buffer(0)


def outline(r):
    """바깥 윤곽(원·둥근 사각·타원)과 그 이름."""
    kind = str(r.choice(["원", "둥근 사각", "타원"]))
    if kind == "원":
        return disc(r.uniform(110, 230), res=96), kind
    if kind == "둥근 사각":
        return rbox(r.uniform(200, 460), r.uniform(160, 380), r.uniform(10, 60)), kind
    return affinity.scale(disc(1.0, res=96), r.uniform(120, 240), r.uniform(80, 200)), kind


# ---------------------------------------------------------------- 방식 (층별 단면 목록, 설명)
def k_lattice(r):
    B, oname = outline(r)
    P = r.uniform(90, 170)                    # 가로 주기(mm)
    s = r.uniform(2.0, 3.0)                   # 세로로 늘인 배수(곡면이 눕는 곳을 줄여 처마를 줄인다)
    w = r.uniform(10, 18)                     # 벽 두께 목표(mm)
    t = w * 2 * math.pi / P * 0.6             # |f| < t 의 수평 두께가 대략 w 가 되게(경험값, 검사로 거른다)
    skin = r.random() < 0.6
    ws = r.uniform(9, 16) if skin else 0.0
    n = int(r.integers(30, 121))
    ph = r.uniform(0, 2 * math.pi, 3)
    x0, y0, x1, y1 = B.bounds
    xs, ys = np.arange(x0 - RES, x1 + RES, RES), np.arange(y0 - RES, y1 + RES, RES)
    X, Y = np.meshgrid(2 * math.pi * xs / P + ph[0], 2 * math.pi * ys / P + ph[1])
    sxcy, sy, cx = np.sin(X) * np.cos(Y), np.sin(Y), np.cos(X)
    inner = B.buffer(-0.5)
    shell = B.difference(B.buffer(-ws, 16)) if skin else Polygon()
    secs = []
    for k in range(n):
        Z = 2 * math.pi * (k + 0.5) * H / (P * s) + ph[2]
        f = sxcy + sy * math.cos(Z) + math.sin(Z) * cx
        g = raster_poly(np.abs(f) < t, xs, ys).intersection(inner)
        secs.append(g.union(shell) if skin else g)
    return secs, (f"lattice/자이로이드 윤곽:{oname} 주기{P:.0f} 세로{s:.1f}배 벽~{w:.0f}"
                  + (f" 껍데기{ws:.1f}" if skin else " 껍데기없음"))


def k_blobs(r):
    m = int(r.integers(3, 8))
    Ht = r.uniform(120, 400)
    n = int(Ht / H)
    nb = max(2, m // 2)                       # 바닥에서 시작하는 덩어리 수(나머지는 위에서 붙거나 사라짐)
    balls = []
    for i in range(m):
        rxy = r.uniform(40, 120)
        rz = rxy * r.uniform(1.5, 3.0)
        a, rad = r.uniform(0, 2 * math.pi), r.uniform(0, 200)
        cz = r.uniform(0.1, 0.6) * rz if i < nb else r.uniform(0.2, 1.0) * Ht
        balls.append((rad * math.cos(a), rad * math.sin(a), cz, rxy, rz, r.uniform(0.6, 1.0), r.uniform(0, 180)))
    bl = r.uniform(10, 30)                    # 이음매를 둥글리는 반지름(녹여 붙임)
    hollow = r.random() < 0.5
    ws = r.uniform(9, 18) if hollow else 0.0
    secs = []
    for k in range(n):
        z = (k + 0.5) * H
        parts = []
        for cx, cy, cz, rxy, rz, asp, rot in balls:
            u = (z - cz) / rz
            if abs(u) >= 1:
                continue
            rr = rxy * math.sqrt(1 - u * u)
            if rr >= 5:
                e = affinity.rotate(affinity.scale(disc(rr, res=48), 1.0, asp), rot, origin=(0, 0))
                parts.append(affinity.translate(e, cx, cy))
        g = unary_union(parts).buffer(bl, 16).buffer(-bl, 16) if parts else Polygon()
        if hollow and not g.is_empty:
            g = g.difference(g.buffer(-ws, 16))
        secs.append(g)
    return secs, f"blobs/{'껍데기' if hollow else '속참'} 덩어리{m}개 높이{Ht:.0f}" + (f" 벽{ws:.1f}" if hollow else "")


def k_helix(r):
    m = int(r.choice([1, 2, 2, 3, 3, 4]))
    R0 = r.uniform(40, 170)
    R1 = R0 * r.uniform(0.6, 1.4)
    rho = r.uniform(16, 45)                   # 관 반지름
    hollow = rho >= 26 and r.random() < 0.45
    wt = r.uniform(9, 14) if hollow else 0.0
    Ht = r.uniform(140, 520)
    n = int(Ht / H)
    p_min = 2 * math.pi * max(R0, R1) * H / (STEP * 0.8)   # 한 층에 옆으로 1.6 mm 넘게 옮겨가지 않는 가장 촘촘한 감기
    p = p_min * r.uniform(1.0, 1.8)
    hand = float(r.choice([-1.0, 1.0]))
    post = r.random() < 0.4
    pr = r.uniform(20, max(21.0, min(R0, R1) - rho - 5)) if post else 0.0
    post_g = disc(pr, res=64) if post else Polygon()
    if post and pr >= 30 and r.random() < 0.5:
        post_g = post_g.difference(disc(pr - r.uniform(9, 14), res=64))
    th0 = r.uniform(0, 2 * math.pi)
    secs = []
    for k in range(n):
        z = (k + 0.5) * H
        R = R0 + (R1 - R0) * z / Ht
        el = math.sqrt(1 + (2 * math.pi * R / p) ** 2)   # 기운 관을 수평으로 자르면 진행 방향으로 이만큼 길다
        parts = [post_g] if post else []
        for i in range(m):
            th = th0 + hand * 2 * math.pi * z / p + 2 * math.pi * i / m
            e = affinity.scale(disc(rho, res=48), el, 1.0)
            if hollow:
                e = e.difference(e.buffer(-wt, 16))
            e = affinity.rotate(e, math.degrees(th) + 90, origin=(0, 0))   # 긴 쪽을 진행(접선) 방향으로
            parts.append(affinity.translate(e, R * math.cos(th), R * math.sin(th)))
        secs.append(unary_union(parts))
    turns = Ht / p
    return secs, (f"helix/관{m}가닥 {'속빈' if hollow else '속찬'} 관반지름{rho:.0f}"
                  + (f" 벽{wt:.1f}" if hollow else "") + f" 감는반지름{R0:.0f}→{R1:.0f} {turns:.2f}바퀴"
                  + (" 가운데기둥" if post else ""))


FN = {"lattice": k_lattice, "blobs": k_blobs, "helix": k_helix}


def make_one(r, kind):
    """형상 하나: 층별 단면 → 가는 곳 빼기·처마 자르기 → (부피 상한) → 채울 수 있는지 → 돌리고 옮기기 → 층마다 쌓기."""
    secs, desc = FN[kind](r)
    secs = finish(secs)
    while secs and secs[-1].is_empty:
        secs.pop()
    if len(secs) < 4 or any(g.is_empty or g.area < 100 for g in secs):
        return None, ""
    area = np.array([g.area for g in secs])
    if area.sum() * H > VCAP:                                    # 너무 크면 위층을 버린다
        secs = secs[:max(4, int(VCAP / (area.mean() * H)))]
    if not all(fill_ok(g) for g in secs):
        return None, ""
    ang = r.uniform(0, 360)
    rot = [affinity.rotate(g, ang, origin=(0, 0)) for g in secs]
    b = unary_union(rot).bounds
    w = max(b[2] - b[0], b[3] - b[1])
    if w > 2 * RMAX * 0.95:
        return None, ""
    lim = max(0.0, RMAX * 0.9 - w / 2)
    dx = r.uniform(-1, 1) * lim * 0.5 - (b[0] + b[2]) / 2
    dy = r.uniform(-1, 1) * lim * 0.5 - (b[1] + b[3]) / 2
    moved = [affinity.translate(g, dx, dy) for g in rot]
    if not disc(RMAX * 0.98, res=128).contains(unary_union(moved)):
        return None, ""
    mesh = stack_safe(moved)
    return (mesh, f"{desc} 폭~{w:.0f} {len(moved)}층") if mesh is not None else (None, "")


def make(out: Path, n: int, seed: int):
    r = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    order = [KINDS[i % len(KINDS)] for i in range(n)]            # 방식이 골고루 나오게 돌아가며
    r.shuffle(order)
    for i, kind in enumerate(order):
        d = out / f"{i + 1:02d}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC_CFG, d / "config.yaml")
        mesh, desc, tries = None, "", 0
        for tries in range(1, 41):
            mesh, desc = make_one(r, kind)
            if mesh is None or not mesh.is_watertight or mesh.bounds[1, 2] < 4:
                mesh = None
                continue
            mesh.export(d / "target.stl")
            if validator_accepts(d):                             # 채점기가 실제로 읽고 모든 층이 비어 있지 않아야 채택
                break
            mesh = None
        if mesh is None:
            raise RuntimeError(f"{d.name} {kind}: 40번 만들어도 조건을 넘는 형상이 없음")
        (d / "desc.txt").write_text(desc, encoding="utf-8")
        print(f"{d.name}: [{KO[kind]}] {desc} (시도 {tries})", flush=True)


if __name__ == "__main__":
    make(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
