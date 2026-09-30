# -*- coding: utf-8 -*-
"""다양한 계열의 학습·검증 형상 생성기 (v4).

사용자 지적: "어려운 것만 만들어서 돌리면 그게 일반화냐" → 쉬운 것부터 어려운 것까지 골고루.

계열 21가지 (난이도는 스케줄링 관점의 대략적인 구분)
  쉬움 : line(막대), ring(원형 벽), rect(사각 벽), block(작은 속찬 덩어리), wave(물결선)
  중간 : hexgrid(벌집, 공개01), trigrid/sqgrid(격자), spiral(나선, 공개02), truss(삼각 트러스, 공개03),
         islands(떨어진 섬 여러 개), star(별 모양 벽)
  어려움: plate(판+기둥, 공개04), vase(비틀린 꽃병, 공개05), bracket(L자+보강판 속찬 덩어리, 공개06)
  추가(2026-09-30, 마지막 테스트용 다양성): gear(톱니바퀴), impeller(허브+굽은 날개), letters(속찬 글자),
         flange(구멍 뚫린 원판), slotplate(슬롯·구멍 판), flare(위로 갈수록 넓어지거나 좁아지는 형상)

높이 방향 변화 (계열과 무관하게 무작위로 붙임)
  prism  : 층마다 같음
  steps  : 몇 구간으로 나눠 구간마다 크기를 비선형으로 바꿈(좁아짐·넓어짐 섞임)
  lean   : 구간마다 조금씩 옆으로 이동
  drop   : 위로 갈수록 일부 요소(섬·선)가 사라짐 → 조각 수 변화
  (vase 는 매끄러운 비선형 변화, plate/bracket 은 자체 규칙)

크기(작음·중간·큼)와 중심 위치(가운데에서 벗어남)도 무작위.
구간마다 따로 밀어 올린 덩어리를 쌓는 방식이라 항상 방수(watertight) 메시가 된다.

usage: python make_jobs_diverse.py <출력폴더> <개수> <시드>
"""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # 제출용 코드(src/)를 불러 쓴다
from make_train_jobs3 import plate as plate_body, vase as vase_body  # noqa: E402

SRC_CFG = Path(__file__).resolve().parents[1] / "WAAM_Validator" / "tests" / "01" / "config.yaml"
H = 2.0
RMAX = 680.0
BEAD = 8.0
GAP = 0.01          # 구간 사이 틈(mm). 층 가운데에서 자르므로 결과에 영향 없음


def wall(geom, w):
    return geom.buffer(w / 2, 8)


# ---------------------------------------------------------------- 2D 단면 계열 (크기 S = 대략 반폭)
def f_line(r, S, w):
    n = int(r.integers(1, 4))
    ys = np.linspace(-S / 2, S / 2, n) if n > 1 else [0.0]
    return unary_union([wall(LineString([(-S, y), (S, y)]), w) for y in ys])


def f_ring(r, S, w):
    return Point(0, 0).buffer(S, 96).difference(Point(0, 0).buffer(S - w, 96))


def f_rect(r, S, w):
    a = S * r.uniform(0.5, 1.0)
    return box(-S, -a, S, a).difference(box(-S + w, -a + w, S - w, a - w))


def f_block(r, S, w):
    S = min(S, 90.0)                                   # 속찬 덩어리는 작게
    return box(-S, -S * r.uniform(0.4, 1.0), S, S * r.uniform(0.4, 1.0))


def f_wave(r, S, w):
    x = np.linspace(-S, S, 200)
    k = int(r.integers(1, 4))
    lines = [LineString(np.column_stack([x, y0 + S * 0.12 * np.sin(k * np.pi * x / S)]))
             for y0 in np.linspace(-S / 2, S / 2, int(r.integers(1, 5)))]
    return unary_union([wall(l, w) for l in lines])


def f_hexgrid(r, S, w):
    a = S / r.uniform(2.0, 3.5)
    segs = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            cx, cy = 1.5 * a * i, math.sqrt(3) * a * (j + 0.5 * (i % 2))
            if math.hypot(cx, cy) > S:
                continue
            pts = [(cx + a * math.cos(math.pi / 3 * t), cy + a * math.sin(math.pi / 3 * t)) for t in range(7)]
            segs.append(LineString(pts))
    return wall(unary_union(segs), w) if segs else f_ring(r, S, w)


def f_sqgrid(r, S, w):
    n = int(r.integers(2, 5))
    xs = np.linspace(-S, S, n + 1)
    return unary_union([wall(LineString([(x, -S), (x, S)]), w) for x in xs] +
                       [wall(LineString([(-S, y), (S, y)]), w) for y in xs])


def f_trigrid(r, S, w):
    lines = []
    for ang in (0, 60, 120):
        for off in np.linspace(-S, S, int(r.integers(3, 6))):
            ln = affinity.rotate(LineString([(-2 * S, off), (2 * S, off)]), ang, origin=(0, 0))
            lines.append(ln)
    # 벽을 두른 뒤 원으로 자르면 끝이 비스듬히 잘려 비드보다 얇은 조각이 생긴다 → 중심선을 먼저 자름
    clipped = unary_union(lines).intersection(Point(0, 0).buffer(S, 64))
    return wall(clipped, w)


def f_spiral(r, S, w):
    turns = r.uniform(1.5, 4.0)
    th = np.linspace(0, 2 * math.pi * turns, 500)
    rr = np.linspace(S * 0.1, S, th.size)
    return wall(LineString(np.column_stack([rr * np.cos(th), rr * np.sin(th)])), w)


def f_truss(r, S, w):
    P = [(S * math.cos(a), S * math.sin(a)) for a in np.radians([90, 210, 330])]
    C = (0.0, 0.0)
    M = [((P[i][0] + P[(i + 1) % 3][0]) / 2, (P[i][1] + P[(i + 1) % 3][1]) / 2) for i in range(3)]
    segs = [LineString([P[i], P[(i + 1) % 3]]) for i in range(3)]
    segs += [LineString([p, C]) for p in P] + [LineString([M[i], M[(i + 1) % 3]]) for i in range(3)]
    return wall(unary_union(segs), w)


def f_islands(r, S, w):
    parts, cs = [], []
    for _ in range(60):
        if len(parts) >= int(r.integers(3, 9)):
            break
        c = (r.uniform(-S, S), r.uniform(-S, S))
        rad = r.uniform(15, 50)
        if any(math.hypot(c[0] - q[0], c[1] - q[1]) < rad + q[2] + 20 for q in cs):
            continue
        cs.append((c[0], c[1], rad))
        g = Point(c).buffer(rad, 48)
        if r.random() < 0.6:
            g = g.difference(Point(c).buffer(max(rad - w, 3), 48))
        parts.append(g)
    return unary_union(parts)


def f_star(r, S, w):
    k = int(r.integers(4, 9))
    th = np.linspace(0, 2 * math.pi, 2 * k + 1)
    rr = np.where(np.arange(2 * k + 1) % 2 == 0, S, S * r.uniform(0.35, 0.7))
    return wall(LineString(np.column_stack([rr * np.cos(th), rr * np.sin(th)])), w)


def f_gear(r, S, w):
    """속찬 톱니바퀴: 이(齒)가 바깥으로 튀어나온 원판 + 가운데 구멍. 이 폭은 비드 폭 이상."""
    S = min(S, 200.0)
    n = int(r.integers(8, 21))
    n = max(4, min(n, int(math.pi * S / max(w, 8.0))))    # 이 하나의 폭(반 피치)이 w 이상이 되게
    depth = float(np.clip(S * r.uniform(0.10, 0.18), w, S * 0.3))
    ri = S - depth
    pts = []
    for k in range(n):
        a0 = 2 * math.pi * k / n
        for f0, f1, rad in ((0.0, 0.5, ri), (0.5, 1.0, S)):
            for t in np.linspace(a0 + f0 * 2 * math.pi / n, a0 + f1 * 2 * math.pi / n, 4):
                pts.append((rad * math.cos(t), rad * math.sin(t)))
    g = Polygon(pts).buffer(0)
    bore = float(np.clip(S * r.uniform(0.2, 0.4), 5.0, max(5.0, ri - w)))
    return g.difference(Point(0, 0).buffer(bore, 64))


def f_impeller(r, S, w):
    """임펠러: 가운데 허브(원판 또는 고리) + 바깥으로 굽어 나가는 날개(두께 w) 여러 장."""
    hub = float(np.clip(S * r.uniform(0.2, 0.35), w, max(w, S - w)))
    n = int(r.integers(3, 8))
    sweep = math.radians(r.uniform(40, 120)) * float(r.choice([-1, 1]))
    r0 = hub * 0.8
    blades = []
    for k in range(n):
        a0 = 2 * math.pi * k / n
        rr = np.linspace(r0, S, 60)
        th = a0 + sweep * (rr - r0) / max(S - r0, 1.0)
        blades.append(wall(LineString(np.column_stack([rr * np.cos(th), rr * np.sin(th)])), w))
    disc = Point(0, 0).buffer(hub, 64)
    if r.random() < 0.5 and hub * 0.5 >= w:
        disc = disc.difference(Point(0, 0).buffer(hub * 0.5, 64))
    return unary_union([disc] + blades)


def _letter(ch, t):
    """단위 사각형(폭 0.7, 높이 1) 안의 속찬 대문자. t = 획 두께(높이 기준 비율)."""
    B = box
    m = 0.35
    L = {
        "I": [B(m - t / 2, 0, m + t / 2, 1)],
        "L": [B(0, 0, t, 1), B(0, 0, 0.7, t)],
        "T": [B(0, 1 - t, 0.7, 1), B(m - t / 2, 0, m + t / 2, 1)],
        "H": [B(0, 0, t, 1), B(0.7 - t, 0, 0.7, 1), B(0, 0.5 - t / 2, 0.7, 0.5 + t / 2)],
        "E": [B(0, 0, t, 1), B(0, 0, 0.7, t), B(0, 1 - t, 0.7, 1), B(0, 0.5 - t / 2, 0.6, 0.5 + t / 2)],
        "F": [B(0, 0, t, 1), B(0, 1 - t, 0.7, 1), B(0, 0.5 - t / 2, 0.6, 0.5 + t / 2)],
        "U": [B(0, 0, t, 1), B(0.7 - t, 0, 0.7, 1), B(0, 0, 0.7, t)],
        "C": [B(0, 0, t, 1), B(0, 0, 0.7, t), B(0, 1 - t, 0.7, 1)],
        "O": [B(0, 0, 0.7, 1).difference(B(t, t, 0.7 - t, 1 - t))],
        "J": [B(0.7 - t, 0, 0.7, 1), B(0, 0, 0.7, t), B(0, 0, t, 0.4)],
        "P": [B(0, 0, t, 1), B(0, 1 - t, 0.7, 1), B(0.7 - t, 0.5, 0.7, 1), B(0, 0.5, 0.7, 0.5 + t)],
    }
    return unary_union(L[ch])


def f_letters(r, S, w):
    """속찬 대문자 1~3개를 나란히. 획 두께는 비드 폭 이상."""
    h = float(min(1.6 * S, 320.0))
    t = max(w / h, 0.2)
    n = 1 if h > 220 else int(r.integers(1, 4))
    chars = r.choice(list("ILTHEFUCOJP"), size=n)
    gap = 0.15
    parts = []
    for i, ch in enumerate(chars):
        g = affinity.scale(_letter(str(ch), t), h, h, origin=(0, 0))
        parts.append(affinity.translate(g, i * (0.7 + gap) * h, 0))
    g = unary_union(parts)
    minx, miny, maxx, maxy = g.bounds
    return affinity.translate(g, -(minx + maxx) / 2, -(miny + maxy) / 2)


def f_flange(r, S, w):
    """플랜지: 원판 + 가운데 큰 구멍 + 둘레의 볼트 구멍. 구멍 사이 살은 비드 폭 이상."""
    S = float(np.clip(S, 60, 220))
    bore = S * r.uniform(0.2, 0.4)
    n = int(r.integers(3, 9))
    pc = (S + bore) / 2
    hr = float(np.clip(S * r.uniform(0.06, 0.12), 3.0, max(3.0, (S - bore) / 2 - w)))
    g = Point(0, 0).buffer(S, 96).difference(Point(0, 0).buffer(bore, 64))
    holes = [Point(pc * math.cos(a), pc * math.sin(a)).buffer(hr, 32)
             for a in np.linspace(0, 2 * math.pi, n, endpoint=False) + r.uniform(0, math.pi)]
    return g.difference(unary_union(holes))


def f_slotplate(r, S, w):
    """슬롯·구멍이 뚫린 속찬 판(사각 또는 원). 구멍끼리 너무 가까우면 합쳐 얇은 살을 없앤다."""
    S = min(S, 220.0)
    a = S * r.uniform(0.5, 1.0)
    plate = box(-S, -a, S, a) if r.random() < 0.6 else Point(0, 0).buffer(S, 96)
    cuts = []
    for _ in range(int(r.integers(2, 7))):
        if r.random() < 0.5:                                    # 긴 슬롯(크기는 판에 비례)
            L, ww = r.uniform(0.3, 0.9) * S, max(8.0, r.uniform(0.06, 0.14) * S)
            g = LineString([(-L / 2, 0), (L / 2, 0)]).buffer(ww / 2, 16)
            g = affinity.rotate(g, float(r.choice([0.0, 90.0, r.uniform(0, 180)])), origin=(0, 0))
        else:                                                   # 둥근 구멍
            g = Point(0, 0).buffer(max(8.0, r.uniform(0.08, 0.2) * S), 32)
        cuts.append(affinity.translate(g, r.uniform(-S, S) * 0.7, r.uniform(-a, a) * 0.7))
    cut = unary_union(cuts).buffer(w / 2, 8).buffer(-w / 2, 8)   # 구멍 사이 틈이 w 보다 좁으면 합침
    inner = plate.buffer(-1.5 * w)                              # 가장자리 살은 남긴다
    return plate.difference(cut.intersection(inner))


SECTIONS = {
    "line": f_line, "ring": f_ring, "rect": f_rect, "block": f_block, "wave": f_wave,
    "hexgrid": f_hexgrid, "sqgrid": f_sqgrid, "trigrid": f_trigrid, "spiral": f_spiral,
    "truss": f_truss, "islands": f_islands, "star": f_star,
    "gear": f_gear, "impeller": f_impeller, "letters": f_letters, "flange": f_flange, "slotplate": f_slotplate,
}
LEVEL = {"line": "쉬움", "ring": "쉬움", "rect": "쉬움", "block": "쉬움", "wave": "쉬움",
         "hexgrid": "중간", "sqgrid": "중간", "trigrid": "중간", "spiral": "중간", "truss": "중간",
         "islands": "중간", "star": "중간", "plate": "어려움", "vase": "어려움", "bracket": "어려움",
         "gear": "중간", "impeller": "어려움", "letters": "중간", "flange": "중간", "slotplate": "중간",
         "flare": "어려움"}


# ---------------------------------------------------------------- 구간 쌓기
def stack(sections, seg_layers, offsets):
    """구간별 단면을 순서대로 밀어 올려 쌓는다 (구간마다 독립 덩어리 → 방수)."""
    parts, z = [], 0.0
    for g, nl, (dx, dy) in zip(sections, seg_layers, offsets):
        g = affinity.translate(g, dx, dy).intersection(Point(0, 0).buffer(RMAX, 128))
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon" or p.area < 30:
                continue
            m = trimesh.creation.extrude_polygon(p, nl * H - GAP)   # 위 구간과 맞닿지 않게 틈
            m.apply_translation([0, 0, z])
            parts.append(m)
        z += nl * H
    return trimesh.util.concatenate(parts) if parts else None


def drop_parts(g, frac, r):
    """요소 일부를 없앤다(조각 수 변화)."""
    geoms = list(getattr(g, "geoms", [g]))
    if len(geoms) <= 1:
        return g.difference(box(-2000, r.uniform(-1, 1) * 200, 2000, 2000)) if r.random() < 0.5 else g
    keep = [x for x in geoms if r.random() > frac]
    return unary_union(keep or geoms[:1])


def gen_section_family(r, name):
    size = r.choice(["작음", "중간", "큼"], p=[0.35, 0.35, 0.30])
    S = {"작음": r.uniform(50, 120), "중간": r.uniform(120, 230), "큼": r.uniform(230, 420)}[size]
    w = float(r.choice([8.0, 8.0, 16.0, 24.0]))
    sub = int(r.integers(0, 2 ** 31))                  # 같은 모양을 크기만 바꿔 다시 그리기 위한 난수 씨앗
    base = SECTIONS[name](np.random.default_rng(sub), S, w)
    mode = r.choice(["prism", "steps", "lean", "drop"], p=[0.3, 0.3, 0.2, 0.2])
    if name == "impeller" and mode in ("steps", "drop"):        # 날개가 구간마다 바뀌면 뒤엉켜 보인다
        mode = "prism"
    total = int(r.integers(3, 61))
    nseg = 1 if mode == "prism" else int(r.integers(2, 6))
    cut = np.sort(r.choice(np.arange(1, total), size=min(nseg - 1, total - 1), replace=False)) if nseg > 1 else []
    seg_layers = np.diff(np.concatenate([[0], cut, [total]])).astype(int)
    secs, offs = [], []
    cx, cy = r.uniform(-1, 1) * (RMAX - S) * 0.5, r.uniform(-1, 1) * (RMAX - S) * 0.5
    sc, dx, dy, g = 1.0, 0.0, 0.0, base
    for i in range(len(seg_layers)):
        if mode == "steps" and i > 0:
            sc = float(np.clip(sc * r.uniform(0.7, 1.25), 0.4, 1.4))   # 커졌다 작아졌다 (비선형)
            # 다각형을 통째로 줄이면 벽 두께도 줄어 비드(8mm)보다 얇아진다 → 크기만 바꿔 다시 그림
            g = SECTIONS[name](np.random.default_rng(sub), S * sc, w)
        if mode == "lean" and i > 0:
            dx += r.uniform(-1, 1) * min(3.0 * seg_layers[i], 40)
            dy += r.uniform(-1, 1) * min(3.0 * seg_layers[i], 40)
        if mode == "drop" and i > 0:
            g = drop_parts(g, 0.35, r)
        secs.append(g)
        offs.append((cx + dx, cy + dy))
    return stack(secs, seg_layers, offs), f"{name}/{mode} {size} 폭~{2 * S:.0f} 벽{w:.0f} {total}층"


def gen_bracket(r):
    """공개 06 계열: 속찬 L자 판 + 위로 갈수록 줄어드는 삼각 보강판."""
    L = r.uniform(80, 220)
    t = r.uniform(10, 30)
    base_layers = int(r.integers(4, 12))
    Lsec = unary_union([box(0, 0, L, t), box(0, 0, t, L)])
    secs, segs = [Lsec], [base_layers]
    rib_total = int(r.integers(10, 50))
    n = int(r.integers(2, 5))
    for i in range(n):
        f = 1 - (i + 1) / (n + 1) * r.uniform(0.6, 1.0)
        tri = Polygon([(t, t), (t + (L - t) * f, t), (t, t + (L - t) * f)])
        secs.append(unary_union([box(0, 0, t, L * r.uniform(0.6, 1.0)), tri]))
        segs.append(max(1, rib_total // n))
    cx, cy = r.uniform(-250, 150), r.uniform(-250, 150)
    return stack(secs, segs, [(cx, cy)] * len(secs)), f"bracket L{L:.0f} 두께{t:.0f} {sum(segs)}층"


def gen_flare(r):
    """위로 갈수록 넓어지거나(오버행) 좁아지는 형상: 같은 단면을 크기만 단조롭게 바꿔 구간마다 다시 그린다."""
    base = str(r.choice(["ring", "rect", "star", "block", "hexgrid", "gear", "flange"]))
    S = r.uniform(60, 260)
    w = float(r.choice([8.0, 8.0, 16.0]))
    sub = int(r.integers(0, 2 ** 31))
    grow = r.random() < 0.7
    s0, s1 = r.uniform(0.45, 0.7), r.uniform(1.1, 1.4)
    if not grow:
        s0, s1 = s1, s0
    nseg = int(r.integers(5, 13))
    total = int(r.integers(2 * nseg, 61))
    seg_layers = np.diff(np.linspace(0, total, nseg + 1).astype(int))
    p = r.uniform(0.6, 1.8)                                   # 비선형으로 커지거나 작아짐
    secs = [SECTIONS[base](np.random.default_rng(sub), S * (s0 + (s1 - s0) * (i / (nseg - 1)) ** p), w)
            for i in range(nseg)]
    cx, cy = r.uniform(-1, 1) * (RMAX - S * max(s0, s1)) * 0.5, r.uniform(-1, 1) * (RMAX - S * max(s0, s1)) * 0.5
    desc = f"flare/{base} {'넓어짐' if grow else '좁아짐'} 폭~{2 * S:.0f} 벽{w:.0f} {total}층"
    return stack(secs, seg_layers, [(cx, cy)] * nseg), desc


def make_one(r, name):
    if name == "flare":
        return gen_flare(r)
    if name == "plate":
        for _ in range(100):
            m, d = plate_body(r)
            if m is not None:
                return m, d
    if name == "vase":
        for _ in range(400):
            m, d = vase_body(r)
            if m is not None:
                return m, d
    if name == "bracket":
        return gen_bracket(r)
    return gen_section_family(r, name)


def validator_accepts(d: Path) -> bool:
    """채점기의 형상 로더로 읽어 보고, 모든 층 단면이 비어 있지 않은지 확인."""
    from waam_validator.config.loader import load_config
    from waam_validator.shape.target import load_target_mesh, slice_target_layers
    try:
        cfg = load_config(d / "config.yaml")
        m = load_target_mesh(d / "target.stl", cfg)
        n = int(np.ceil((m.bounds[1, 2] - cfg.process.build_plane_z_mm) / cfg.process.layer_height_mm - 1e-9))
        L = slice_target_layers(m, list(range(n)), cfg)
        return all(not L[k].is_empty for k in range(n))
    except Exception:
        return False


def make(out: Path, n: int, seed: int):
    r = np.random.default_rng(seed)
    names = list(SECTIONS) + ["plate", "vase", "bracket", "flare"]
    out.mkdir(parents=True, exist_ok=True)
    order = [names[i % len(names)] for i in range(n)]      # 계열이 골고루 나오게 돌아가며
    r.shuffle(order)
    for i, name in enumerate(order):
        mesh, desc = None, ""
        d = out / f"{i + 1:02d}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC_CFG, d / "config.yaml")
        for _ in range(30):
            mesh, desc = make_one(r, name)
            if mesh is None or not mesh.is_watertight or mesh.bounds[1, 2] < 4:
                continue
            mesh.export(d / "target.stl")
            if validator_accepts(d):                   # 채점기가 실제로 읽고 모든 층이 비어 있지 않아야 채택
                break
        (d / "desc.txt").write_text(f"{LEVEL[name]} {desc}", encoding="utf-8")
        print(f"{d.name}: [{LEVEL[name]}] {desc}  watertight={mesh.is_watertight}")


if __name__ == "__main__":
    make(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
