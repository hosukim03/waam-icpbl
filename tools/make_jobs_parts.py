# -*- coding: utf-8 -*-
"""연습 3: 실제 부품을 본뜬 확인용 형상 생성기(2026-11-26).

연습 1(make_jobs_diverse.py)은 정해진 21종류에 벽 두께 8·16·24 mm, 연습 2(make_jobs_novel.py)는 정해진 종류 없이
무작위 윤곽이다. 여기서는 금속 3D 프린팅(WAAM)으로 실제 만드는 부품 11종류를 본뜨고 치수는 무작위로 정한다.
실제 부품처럼
  - 높이에 따라 요소가 생기고 끝난다(바닥판·플랜지 위의 벽, 중간에 끝나는 보강재·보스, 문·창 구멍)
  - 두께는 어중간한 값이다(대부분 9–24 mm 연속값)
  - 바깥 모서리는 둥글거나 뾰족하다(보강재 끝, 판 모서리)
누구도 통과할 수 없는 형상은 넣지 않는다: 층마다 비드(폭 8 mm)로 넘치지 않게 채울 수 있는 넓이(단면을 4 mm
깎았다 되살린 넓이)가 99% 이상이어야 한다. 연습 1 14번은 6.6 mm 띠라 82%, 20번 톱니바퀴는 97.9% 였다.

종류(첫 낱말이 desc.txt 의 첫 낱말, KO 는 대시보드에 보이는 이름)
  panel    보강재 붙은 패널  항공기 날개·동체 판을 세워 쌓음. 곧거나 휜 판 + T·L·평판 보강재, 일부는 중간에 끝남
  column   H형강 기둥       볼트 구멍 뚫린 바닥판 위 H 단면. 위로 갈수록 좁아지거나 중간에 가로 보강판
  pipe     플랜지 관        볼트 구멍 뚫린 플랜지 위 관. pipe(곧음)·reducer(지름이 바뀜)·nozzle(좁아졌다 넓어짐)
  housing  상자형 하우징     둥근 모서리 사각 벽 + 안쪽 보강 벽 + 모서리 보스(볼트 구멍) + 바닥 플랜지
  wheel    바퀴(풀리)       테 + 허브 + 살 3–8개(곧거나 휨). 테에 홈이 있거나 허브가 테보다 높을 수 있음
  link     연결 막대        큰 고리 + 작은 고리(conrod) 또는 벌린 입(spanner) + 가운데가 파인 몸통
  blade    터빈 날개        뿌리 블록 + 받침판 + 비틀리며 가늘어지는 날개 단면
  manifold 유압 블록        속찬 블록에 지름이 다른 세로 구멍들(위쪽만 넓은 구멍, 위에서만 판 구멍), 한쪽이 낮은 계단
  heatsink 방열판          바닥판 위 얇은 지느러미 여러 장(fins) 또는 둥근 핀 여러 개(pins)
  cylinder 보강 원통        로켓 연료 탱크 조각. 큰 원통 벽 + 세로 보강재 + 몇 층마다 두꺼운 링
  facade   창 뚫린 벽       건축 부재. 곧거나 꺾인 벽, 문·창 자리에서 벽이 끊김

usage: python make_jobs_parts.py <출력폴더> <개수> <시드>   (WAAM venv 파이썬: 채점기로 읽어 보고 넣는다)
"""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import substring, unary_union

sys.path.insert(0, str(Path(__file__).parent))
from make_jobs_diverse import RMAX, SRC_CFG, stack, validator_accepts  # noqa: E402

KINDS = ["panel", "column", "pipe", "housing", "wheel", "link", "blade", "manifold", "heatsink", "cylinder", "facade"]
KO = {"panel": "보강재 붙은 패널", "column": "H형강 기둥", "pipe": "플랜지 관", "reducer": "지름이 바뀌는 관",
      "nozzle": "노즐", "housing": "상자형 하우징", "wheel": "바퀴(풀리)", "conrod": "커넥팅 로드", "spanner": "스패너",
      "blade": "터빈 날개", "manifold": "유압 블록", "heatsink": "방열판", "cylinder": "보강 원통", "facade": "창 뚫린 벽"}
VCAP = 4.5e6          # 전체 부피 상한(mm³), 연습 2 와 같다(공개 05 가 약 9e6). 넘으면 위층을 버린다
STEP = 2.0            # 서서히 바뀌는 곳에서 한 층에 옆으로 움직여도 되는 거리(mm)
RB = 4.0              # 비드 반지름(mm)
FILL_MIN = 0.99       # 층마다 비드로 넘치지 않게 채울 수 있는 넓이 비율의 하한


def thick(r, lo=9.0, hi=24.0) -> float:
    """어중간한 두께(연속값)."""
    return float(r.uniform(lo, hi))


def rbox(w, h, rc=0.0, cx=0.0, cy=0.0):
    """가운데 (cx, cy), w×h 사각형. rc > 0 이면 모서리를 반지름 rc 로 둥글린다."""
    b = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    rc = min(rc, w / 2 - 0.5, h / 2 - 0.5)
    return b.buffer(-rc, join_style="mitre").buffer(rc, 16) if rc > 0.5 else b


def disc(rad, c=(0.0, 0.0), res=48):
    return Point(*c).buffer(rad, res)


def annulus(R, t, c=(0.0, 0.0)):
    """바깥 반지름 R, 두께 t 인 고리."""
    return disc(R, c, 96).difference(disc(R - t, c, 96))


def strip(p, q, w, round_end=False):
    """p→q 를 가운데로 하는 폭 w 띠. 끝은 뾰족한 사각(기본) 또는 반원."""
    return LineString([p, q]).buffer(w / 2, 16, cap_style="round" if round_end else "flat")


def polar(rad, a):
    return (rad * math.cos(a), rad * math.sin(a))


def build(n, add, cut=()):
    """층 k 단면 = (k0 ≤ k < k1 인 add 요소의 합) − (같은 조건인 cut 요소의 합). 같은 조합은 한 번만 계산한다."""
    memo, secs = {}, []
    for k in range(n):
        a = tuple(i for i, (_, k0, k1) in enumerate(add) if k0 <= k < k1)
        c = tuple(i for i, (_, k0, k1) in enumerate(cut) if k0 <= k < k1)
        if (a, c) not in memo:
            g = unary_union([add[i][0] for i in a]) if a else Polygon()
            if c and not g.is_empty:
                g = g.difference(unary_union([cut[i][0] for i in c]))
            memo[(a, c)] = g.buffer(0)
        secs.append(memo[(a, c)])
    return secs


def fill_ratio(g) -> float:
    """비드로 넘치지 않게 채울 수 있는 넓이 비율 = 단면을 비드 반지름만큼 깎았다 되살린 넓이 / 단면 넓이."""
    if g.is_empty or g.area <= 0:
        return 0.0
    return g.buffer(-RB * 0.999, 16).buffer(RB * 0.999, 16).area / g.area


# ---------------------------------------------------------------- 부품 (층별 단면 목록, 설명)
def k_panel(r):
    L, ts = r.uniform(300, 1000), thick(r, 9, 14)              # 판 길이·두께
    n = int(r.integers(10, 61))
    sty = str(r.choice(["T", "L", "I"]))                         # 보강재: T(끝에 덧판), L(한쪽 덧판), I(평판)
    m = max(2, int(L // r.uniform(90, 200)))
    xs = np.linspace(-L / 2, L / 2, m + 1)[:-1] + L / m / 2
    hd, tr = r.uniform(30, 80), thick(r, 9, 14)                 # 보강재 깊이·두께
    wf, tf = r.uniform(24, 50), thick(r, 9, 12)                 # 덧판 폭·두께
    rnd = r.random() < 0.4                                       # 끝을 둥글게
    curved, Rc = r.random() < 0.4, r.uniform(500, 1800)          # 휜 판(반지름 Rc)

    def bend(g):
        if not curved:
            return g
        def f(xy):
            x, y = xy[:, 0], xy[:, 1]
            return np.column_stack([(Rc - y) * np.sin(x / Rc), Rc - (Rc - y) * np.cos(x / Rc)])
        return shapely.transform(shapely.segmentize(g, 4.0), f).buffer(0)

    add = [(bend(rbox(L, ts, ts / 2 if rnd else 0.0, 0.0, ts / 2)), 0, n)]
    for x in xs:
        y1 = ts + hd
        g = box(x - tr / 2, ts - 0.5, x + tr / 2, y1)            # 판에 0.5 mm 묻어 이어지게
        if sty == "T":
            g = g.union(rbox(wf, tf, tf / 2 if rnd else 0.0, x, y1 - tf / 2))
        elif sty == "L":
            g = g.union(rbox(wf, tf, tf / 2 if rnd else 0.0, x + wf / 2 - tr / 2, y1 - tf / 2))
        elif rnd:
            g = g.union(disc(tr / 2, (x, y1), 16))
        stop = n if r.random() < 0.7 else max(2, int(n * r.uniform(0.3, 0.85)))   # 일부는 중간에 끝남
        add.append((bend(g), 0, stop))
    return build(n, add), f"panel/{sty} {'휜 판' if curved else '곧은 판'} 판{ts:.1f} 보강재{m}개 두께{tr:.1f}"


def k_column(r):
    B, D = r.uniform(80, 300), r.uniform(100, 450)              # 플랜지 폭, 단면 높이
    tf, tw = thick(r, 10, 24), thick(r, 9, 18)
    n, nb = int(r.integers(12, 61)), int(r.integers(2, 7))      # 전체·바닥판 층수
    e = r.uniform(30, 70)                                        # 바닥판이 H 밖으로 나온 폭
    plate = rbox(B + 2 * e, D + 2 * e, r.uniform(0, 25))
    dh = min(r.uniform(14, 26), e - 16)                          # 볼트 구멍(가장자리에 8 mm 이상 남김)
    if dh >= 10:
        plate = plate.difference(unary_union([disc(dh / 2, (sx * (B / 2 + e / 2), sy * (D / 2 + e / 2)), 32)
                                              for sx in (-1, 1) for sy in (-1, 1)]))
    D1 = D
    if r.random() < 0.4:                                         # 위로 갈수록 좁아짐(한 층에 STEP 넘게 줄지 않게)
        D1 = max(D * r.uniform(0.55, 0.9), D - 2 * STEP * (n - nb - 1))
    dl = int(r.integers(2, 4))                                   # 가로 보강판 층수
    dia = sorted({int(x) for x in r.integers(nb + 3, n - dl, size=int(r.integers(0, 3)))}) if n - dl > nb + 3 else []
    memo = {}

    def H(Dk, full):
        key = (round(Dk, 2), full)
        if key not in memo:
            memo[key] = box(-B / 2, -Dk / 2, B / 2, Dk / 2) if full else unary_union(
                [box(-B / 2, Dk / 2 - tf, B / 2, Dk / 2), box(-B / 2, -Dk / 2, B / 2, -Dk / 2 + tf),
                 box(-tw / 2, -Dk / 2, tw / 2, Dk / 2)])
        return memo[key]
    secs = []
    for k in range(n):
        if k < nb:
            secs.append(plate)
            continue
        Dk = D + (D1 - D) * (k - nb) / max(n - nb - 1, 1)
        secs.append(H(Dk, any(a <= k < a + dl for a in dia)))   # 가로 보강판: 플랜지 사이를 채운 층
    return secs, f"column{'/taper' if D1 != D else ''} 플랜지{tf:.1f} 웹{tw:.1f} 가로판{len(dia)}곳"


def k_pipe(r):
    sub = str(r.choice(["pipe", "reducer", "nozzle"]))
    t = thick(r, 9, 16)
    R0 = r.uniform(max(60.0, t + 30), 400)                      # 바깥 반지름
    n, nf, wf = int(r.integers(14, 61)), int(r.integers(3, 8)), r.uniform(25, 60)
    dh0, kh, a0 = r.uniform(12, 22), int(r.integers(4, 13)), r.uniform(0, 1)

    def flange(R):
        g = disc(R + wf, res=128).difference(disc(R - t, res=128))
        dh = min(dh0, wf - 16)                                   # 구멍 양옆에 8 mm 이상
        k = min(kh, int(2 * math.pi * (R + wf / 2) / (dh + 10)))
        if dh >= 10 and k >= 3:
            g = g.difference(unary_union([disc(dh / 2, polar(R + wf / 2, a0 + 2 * math.pi * i / k), 32) for i in range(k)]))
        return g
    m = n - nf
    Rs = np.full(m, R0)
    if sub == "reducer":
        R1 = max(R0 * (r.uniform(0.55, 0.85) if r.random() < 0.6 else r.uniform(1.15, 1.4)), t + 30)
        span = max(int(math.ceil(abs(R1 - R0) / STEP)), int(m * r.uniform(0.2, 0.4)), 1)
        a = int(r.integers(0, max(1, m - span)))
        Rs = R0 + (R1 - R0) * np.clip((np.arange(m) - a) / span, 0, 1)
    elif sub == "nozzle":
        Rt, Re, ut = max(R0 * r.uniform(0.5, 0.8), t + 30), R0 * r.uniform(0.9, 1.5), r.uniform(0.3, 0.55)
        u = np.linspace(0, 1, m)
        Rs = np.where(u < ut, Rt + (R0 - Rt) * (1 - u / ut) ** 2,
                      Rt + (Re - Rt) * np.clip((u - ut) / (1 - ut), 0, None) ** 1.5)
        for i in range(1, m):                                    # 한 층에 STEP 넘게 바뀌지 않게
            Rs[i] = Rs[i - 1] + np.clip(Rs[i] - Rs[i - 1], -STEP, STEP)
    ring, secs = {}, [flange(R0)] * nf
    for R in Rs:
        key = round(float(R), 2)
        if key not in ring:
            ring[key] = annulus(key, t)
        secs.append(ring[key])
    top = sub != "nozzle" and r.random() < 0.3
    if top:                                                      # 위쪽 플랜지
        nt = int(r.integers(2, 5))
        secs[-nt:] = [flange(float(Rs[-1]))] * nt
    return secs, f"{sub} 관 지름~{2 * R0:.0f} 벽{t:.1f} 플랜지{nf}층" + (" 위 플랜지" if top else "")


def k_housing(r):
    W = r.uniform(200, 800)
    Hh = r.uniform(150, min(600.0, W))
    t = thick(r, 10, 18)
    rc = r.uniform(t + 3, 60)
    n = int(r.integers(12, 61))
    outer, inner = rbox(W, Hh, rc), rbox(W - 2 * t, Hh - 2 * t, rc - t)
    add, cut = [(outer.difference(inner), 0, n)], []
    nr = int(r.integers(0, 3))                                   # 안쪽 보강 벽(짧은 변과 나란히)
    for x in np.linspace(-W / 2, W / 2, nr + 2)[1:-1]:
        tr = thick(r, 9, 14)
        add.append((box(x - tr / 2, -Hh / 2 + t / 2, x + tr / 2, Hh / 2 - t / 2), 0,
                    n if r.random() < 0.6 else max(2, int(n * r.uniform(0.4, 0.9)))))
    rb = r.uniform(14, 28)                                       # 모서리 보스 + 볼트 구멍
    dh = min(r.uniform(8, 14), 2 * (rb - 8))
    hb = n if r.random() < 0.6 else max(2, int(n * r.uniform(0.5, 0.9)))
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = (sx * (W / 2 - t - rb * 0.6), sy * (Hh / 2 - t - rb * 0.6))
            add.append((disc(rb, c, 32), 0, hb))
            if dh >= 8:
                cut.append((disc(dh / 2, c, 24), 0, hb))
    nf = int(r.integers(0, 6))                                   # 0 이면 바닥 플랜지 없음
    if nf:
        wf = r.uniform(20, 45)
        add.append((rbox(W + 2 * wf, Hh + 2 * wf, rc + wf).difference(inner), 0, nf))
        dfh = min(r.uniform(10, 16), wf - 16)
        if dfh >= 9:
            pos = [(sx * (W / 2 + wf / 2), 0.0) for sx in (-1, 1)]
            pos += [(x, sy * (Hh / 2 + wf / 2)) for sy in (-1, 1)
                    for x in np.linspace(-W / 2 + rc, W / 2 - rc, max(2, int(W // 250) + 1))]
            cut.append((unary_union([disc(dfh / 2, p, 24) for p in pos]), 0, nf))
    return build(n, add, cut), f"housing 벽{t:.1f} 보강 벽{nr}개 보스{hb}층" + (f" 바닥 플랜지{nf}층" if nf else "")


def k_wheel(r):
    trim, R = thick(r, 10, 20), r.uniform(120, 550)             # 테 두께·바깥 반지름
    rh, th = r.uniform(20, 70), thick(r, 12, 30)                # 허브 구멍 반지름·허브 두께
    Rh = rh + th
    k, ws = int(r.integers(3, 9)), thick(r, 12, 30)             # 살 개수·폭
    curved = r.random() < 0.4
    n = int(r.integers(6, 41))
    a0, bend = r.uniform(0, 2 * math.pi), r.uniform(0.3, 0.8) * (1 if r.random() < 0.5 else -1)
    spokes = []
    for i in range(k):
        a = a0 + 2 * math.pi * i / k
        rr = np.linspace(Rh - th / 2, R - trim / 2, 24 if curved else 2)
        aa = a + (bend * (rr - rr[0]) / (rr[-1] - rr[0]) if curved else 0.0)
        spokes.append(LineString(np.column_stack([rr * np.cos(aa), rr * np.sin(aa)])).buffer(ws / 2, 16, cap_style="flat"))
    hub = annulus(Rh, th)
    add = [(unary_union([annulus(R, trim), hub] + spokes), 0, n)]
    extra = int(r.integers(3, 16)) if r.random() < 0.5 else 0  # 허브만 더 높이
    if extra:
        add.append((hub, n, n + extra))
    groove = r.random() < 0.4                                    # 풀리: 테 위아래 몇 층이 바깥으로 더 두꺼움
    if groove:
        gw, ga = r.uniform(8, 20), int(r.integers(2, 5))
        lip = annulus(R + gw, gw + 1)
        add += [(lip, 0, ga), (lip, n - ga, n)]
    return build(n + extra, add), (f"wheel{'/groove' if groove else ''} 지름~{2 * R:.0f} 살{k}개{'(휨)' if curved else ''}"
                                   f" 테{trim:.1f}" + (f" 허브만 {extra}층 더" if extra else ""))


def k_link(r):
    Lc = r.uniform(150, 600)                                     # 두 고리 중심 거리
    Rb = r.uniform(35, 110)
    rb = Rb - thick(r, 12, 30)                                   # 큰 고리 바깥·구멍 반지름
    Rs = r.uniform(25, max(26.0, Rb * 0.8))
    rs = Rs - thick(r, 10, 22)
    n, a = int(r.integers(6, 31)), int(r.integers(2, 6))        # 전체 층수, 몸통 바닥(웹) 층수
    cb, cs = (-Lc / 2, 0.0), (Lc / 2, 0.0)
    wb, wsm = 2 * Rb * r.uniform(0.5, 0.8), 2 * Rs * r.uniform(0.5, 0.8)
    shank = Polygon([(cb[0], -wb / 2), (cs[0], -wsm / 2), (cs[0], wsm / 2), (cb[0], wb / 2)])
    e = thick(r, 9, 16)                                          # 파인 곳 둘레 폭
    pocket = shank.buffer(-e, join_style="mitre").difference(disc(Rb + e / 2, cb, 64)).difference(disc(Rs + e / 2, cs, 64))
    wj = min(2 * rs, 2 * (Rs - 9))                               # 스패너 입 벌림(양쪽 턱 9 mm 이상)
    spanner = r.random() < 0.35 and wj >= 12
    small = disc(Rs, cs, 96)
    if spanner:
        small = small.difference(box(cs[0], -wj / 2, cs[0] + 2 * Rs, wj / 2))
    holes = [disc(rb, cb, 64)] if rb >= 8 else []
    if not spanner and rs >= 8:
        holes.append(disc(rs, cs, 64))
    cut = [(unary_union(holes), 0, n)] if holes else []
    if pocket.area > 300:
        cut.append((pocket, a, n))
    return (build(n, [(unary_union([disc(Rb, cb, 96), small, shank]), 0, n)], cut),
            f"{'spanner' if spanner else 'conrod'} 고리{Rb:.0f}/{Rs:.0f} 몸통 테두리{e:.1f}" + (" 파임" if pocket.area > 300 else ""))


def airfoil(c, tmax, m, p, npts=60):
    """NACA 4자리 날개 단면(시위 c). 뒤끝이 비드보다 얇아지지 않게 두께를 9.4 mm 이상으로 두고(뭉툭한 뒤끝),
    앞끝·뒤끝을 반지름 4.5 mm 로 둥글린다. 시위의 35% 자리가 원점(비틀림 축)."""
    x = (1 - np.cos(np.linspace(0, math.pi, npts))) / 2
    yt = 5 * tmax * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4) * c
    yt = np.maximum(yt, 4.7)
    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2)) * c
    th = np.arctan(np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x)))
    up = np.column_stack([x * c - yt * np.sin(th), yc + yt * np.cos(th)])
    lo = np.column_stack([x * c + yt * np.sin(th), yc - yt * np.cos(th)])
    P = Polygon(np.vstack([up, lo[::-1]])).buffer(0).buffer(-4.5).buffer(4.5, 16)
    return affinity.translate(P, -0.35 * c, 0.0)


def k_blade(r):
    c = r.uniform(80, 300)                                       # 시위 길이
    tmax, m, p = r.uniform(0.12, 0.22), r.uniform(0.0, 0.07), r.uniform(0.3, 0.5)
    nr, npl, na = int(r.integers(3, 9)), int(r.integers(2, 4)), int(r.integers(10, 51))
    lim = math.degrees(STEP * na / (0.65 * c))                   # 뒤끝이 한 층에 STEP 넘게 돌지 않게
    tw = min(r.uniform(15, 60), lim) * (1 if r.random() < 0.5 else -1)
    s1 = max(r.uniform(0.6, 1.0), 1 - STEP * na / (0.65 * c))   # 위끝 시위 비율
    th = tmax * c
    root = rbox(c * r.uniform(0.75, 0.95), max(th * r.uniform(1.6, 2.4), 30.0), r.uniform(5, 15), 0.15 * c, 0.0)
    plat = rbox(c * 1.15, max(th * 2.6, 50.0), r.uniform(5, 20), 0.15 * c, 0.0)
    secs = [root] * nr + [plat] * npl
    for i in range(na):
        u = i / max(na - 1, 1)
        secs.append(affinity.rotate(airfoil(c * (1 + (s1 - 1) * u), tmax, m, p), tw * u, origin=(0, 0)))
    return secs, f"blade 시위{c:.0f} 두께{100 * tmax:.0f}% 비틀림{abs(tw):.0f}° 끝 시위 {100 * s1:.0f}%"


def k_manifold(r):
    W = r.uniform(80, 260)
    D = r.uniform(60, min(220.0, W))
    n = int(r.integers(10, 51))
    block = rbox(W, D, r.uniform(0, 20))
    cut, placed = [], []
    for _ in range(int(r.integers(3, 9))):
        d = r.uniform(12, 40)
        cb = r.random() < 0.35                                   # 위쪽 몇 층만 넓은 구멍
        dcb = d + r.uniform(8, 16) if cb else d
        rad = dcb / 2
        if W - 2 * rad - 20 <= 0 or D - 2 * rad - 20 <= 0:
            continue
        for _ in range(60):
            x, y = r.uniform(-W / 2 + rad + 10, W / 2 - rad - 10), r.uniform(-D / 2 + rad + 10, D / 2 - rad - 10)
            if all(math.hypot(x - px, y - py) >= rad + pr + 10 for px, py, pr in placed):
                break
        else:
            continue
        placed.append((x, y, rad))
        k0 = 0 if r.random() < 0.7 else int(n * r.uniform(0.3, 0.7))   # 위에서만 판 구멍
        cut.append((disc(d / 2, (x, y), 32), k0, n))
        if cb:
            cut.append((disc(dcb / 2, (x, y), 32), max(k0, n - int(r.integers(2, 7))), n))
    step = r.random() < 0.5                                      # 한쪽이 낮은 계단
    if step:
        xs, ns = r.uniform(-0.3, 0.3) * W, max(2, int(n * r.uniform(0.4, 0.8)))
        add = [(block.intersection(box(-W, -D, xs, D)), 0, n), (block.intersection(box(xs, -D, W, D)), 0, ns)]
    else:
        add = [(block, 0, n)]
    return build(n, add, cut), f"manifold 구멍{len(placed)}개" + (" 계단" if step else "")


def k_heatsink(r):
    W = r.uniform(120, 420)
    D = r.uniform(100, min(360.0, W))
    nb = int(r.integers(3, 6))                                   # 바닥판 층수
    n = nb + int(r.integers(8, 41))
    add = [(rbox(W, D, r.uniform(0, 15)), 0, nb)]
    mg = r.uniform(4, 15)
    if r.random() < 0.6:                                         # 지느러미
        tf, g = thick(r, 8.5, 14), r.uniform(10, 30)
        k = int(min(25, max(2, (D - 2 * mg + g) // (tf + g))))
        rnd, split, arch = r.random() < 0.4, r.random() < 0.3, r.random() < 0.35
        ch = r.uniform(20, 40)
        x0, x1 = -W / 2 + mg + (tf / 2 if rnd else 0), W / 2 - mg - (tf / 2 if rnd else 0)
        for y in (np.arange(k) - (k - 1) / 2) * (tf + g):
            top = n if not arch else nb + max(3, int((n - nb) * (1 - 0.5 * (2 * y / D) ** 2)))   # 가운데가 높음
            for a, b in ([(x0, -ch / 2), (ch / 2, x1)] if split else [(x0, x1)]):
                add.append((strip((a, y), (b, y), tf, rnd), nb, top))
        desc = (f"heatsink/fins 지느러미{k}장 두께{tf:.1f} 간격{g:.0f}" + (" 가운데 홈" if split else "")
                + (" 가운데가 높음" if arch else ""))
    else:                                                        # 둥근 핀(최대 약 120개)
        dp = r.uniform(10, 24)
        pitch = max(dp + r.uniform(10, 30), math.sqrt(W * D / 120))
        stag = r.random() < 0.5
        nx, ny = int((W - 2 * mg - dp) // pitch) + 1, int((D - 2 * mg - dp) // pitch) + 1
        pins = [disc(dp / 2, (x, (j - (ny - 1) / 2) * pitch), 24)
                for j in range(ny) for i in range(nx)
                for x in [(i - (nx - 1) / 2) * pitch + (pitch / 2 if stag and j % 2 else 0.0)]
                if abs(x) <= W / 2 - mg - dp / 2]
        add.append((unary_union(pins), nb, n))
        desc = f"heatsink/pins 핀{len(pins)}개 지름{dp:.1f} 간격{pitch:.0f}" + (" 엇갈림" if stag else "")
    return build(n, add), desc


def k_cylinder(r):
    t, R = thick(r, 9, 13), r.uniform(200, 560)                 # 원통 벽 두께·바깥 반지름
    n = int(r.integers(12, 61))
    hd, ts = r.uniform(18, 45), thick(r, 9, 13)                 # 세로 보강재 깊이·두께
    inside, cap, rnd = r.random() < 0.7, r.random() < 0.4, r.random() < 0.5
    cw, ct = r.uniform(20, 36), thick(r, 9, 11)                 # T 덧판 폭·두께
    k = min(int(r.integers(8, 37)), int(2 * math.pi * (R - t - hd) / ((cw if cap else ts) + 12)))
    parts = [annulus(R, t)]
    for i in range(k):
        a = 2 * math.pi * i / k
        r1 = R - t - hd if inside else R + hd
        parts.append(strip(polar(R - t / 2, a), polar(r1, a), ts, rnd and not cap))
        if cap:
            q = polar(r1 + (ct / 2 if inside else -ct / 2), a)
            tx, ty = -math.sin(a) * cw / 2, math.cos(a) * cw / 2
            parts.append(strip((q[0] - tx, q[1] - ty), (q[0] + tx, q[1] + ty), ct, rnd))
    fm, fb, fw = int(r.integers(8, 21)), int(r.integers(2, 4)), r.uniform(18, 40)   # 몇 층마다, 몇 층, 폭
    frame = annulus(R - t + 1, fw + 1) if inside else annulus(R + fw, fw + 1)
    add = [(unary_union(parts), 0, n)] + [(frame, k0, k0 + fb) for k0 in range(fm // 2, n - fb, fm)]
    return build(n, add), (f"cylinder 지름~{2 * R:.0f} 벽{t:.1f} 보강재{k}개({'안' if inside else '밖'}"
                           f"{', T' if cap else ''}) 링 {fm}층마다")


def k_facade(r):
    t = thick(r, 10, 24)
    plan = str(r.choice(["line", "L", "U", "zigzag"]))
    n = int(r.integers(15, 61))
    if plan == "line":
        L = r.uniform(400, 1100)
        pts = [(-L / 2, 0.0), (L / 2, 0.0)]
    elif plan in ("L", "U"):
        a, b = r.uniform(300, 800), r.uniform(200, 600 if plan == "L" else 500)
        pts = [(-a / 2, b / 2), (-a / 2, -b / 2), (a / 2, -b / 2)] + ([(a / 2, b / 2)] if plan == "U" else [])
    else:
        m, seg, ang = int(r.integers(3, 6)), r.uniform(150, 300), math.radians(r.uniform(30, 60))
        pts = [(i * seg * math.cos(ang), seg * math.sin(ang) if i % 2 else 0.0) for i in range(m + 1)]
    path = LineString(pts)
    wall = path.buffer(t / 2, 16, cap_style="flat", join_style="mitre")
    cum = np.concatenate([[0.0], np.cumsum([math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)])])
    used, cut = [], []
    for _ in range(int(r.integers(1, 6))):                      # 문·창은 곧은 구간 안에만(끝·모서리에서 떨어짐)
        i, w = int(r.integers(0, len(pts) - 1)), r.uniform(60, 200)
        lo, hi = cum[i] + t + 40, cum[i + 1] - t - 40 - w
        if hi <= lo:
            continue
        s0 = r.uniform(lo, hi)
        if any(s0 < b + 30 and a - 30 < s0 + w for a, b in used):
            continue
        used.append((s0, s0 + w))
        if r.random() < 0.35:                                    # 문: 바닥부터
            k0, k1 = 0, max(2, int(n * r.uniform(0.5, 0.8)))
        else:                                                    # 창: 중간 높이
            k0 = max(1, int(n * r.uniform(0.15, 0.45)))
            k1 = min(n - 2, k0 + max(3, int(n * r.uniform(0.2, 0.4))))
        cut.append((substring(path, s0, s0 + w).buffer(t, cap_style="flat"), k0, k1))
    return build(n, [(wall, 0, n)], cut), f"facade/{plan} 벽{t:.1f} 문·창{len(cut)}개"


FN = {"panel": k_panel, "column": k_column, "pipe": k_pipe, "housing": k_housing, "wheel": k_wheel, "link": k_link,
      "blade": k_blade, "manifold": k_manifold, "heatsink": k_heatsink, "cylinder": k_cylinder, "facade": k_facade}


def make_one(r, kind):
    """부품 하나: 층별 단면 → (부피 상한) → 채울 수 있는지 → 돌리고 옮기기 → 같은 단면이 이어지는 층을 한 덩어리로 쌓기."""
    secs, desc = FN[kind](r)
    area = np.array([g.area for g in secs])
    if len(secs) < 4 or area.min() < 100:
        return None, ""
    if area.sum() * 2.0 > VCAP:                                  # 너무 크면 위층을 버린다
        secs = secs[:max(4, int(VCAP / (area.mean() * 2.0)))]
    uniq = list({id(g): g for g in secs}.values())
    if min(fill_ratio(g) for g in uniq) < FILL_MIN:
        return None, ""
    ang = r.uniform(0, 360)
    rot = {id(g): affinity.rotate(g, ang, origin=(0, 0)) for g in uniq}
    b = unary_union(list(rot.values())).bounds
    w = max(b[2] - b[0], b[3] - b[1])
    if w > 2 * RMAX * 0.95:
        return None, ""
    lim = max(0.0, RMAX * 0.9 - w / 2)
    dx = r.uniform(-1, 1) * lim * 0.5 - (b[0] + b[2]) / 2
    dy = r.uniform(-1, 1) * lim * 0.5 - (b[1] + b[3]) / 2
    moved = {i: affinity.translate(g, dx, dy) for i, g in rot.items()}
    if not disc(RMAX * 0.98, res=128).contains(unary_union(list(moved.values()))):
        return None, ""
    segs, nl = [], []
    for g in secs:
        mg = moved[id(g)]
        if segs and (mg is segs[-1] or mg.equals_exact(segs[-1], 0.0)):
            nl[-1] += 1
        else:
            segs.append(mg)
            nl.append(1)
    return stack(segs, nl, [(0.0, 0.0)] * len(segs)), f"{desc} 폭~{w:.0f} {len(secs)}층"


def make(out: Path, n: int, seed: int):
    r = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    order = [KINDS[i % len(KINDS)] for i in range(n)]            # 종류가 골고루 나오게 돌아가며
    r.shuffle(order)
    for i, kind in enumerate(order):
        d = out / f"{i + 1:02d}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC_CFG, d / "config.yaml")
        mesh, desc, tries = None, "", 0
        for tries in range(1, 81):
            mesh, desc = make_one(r, kind)
            if mesh is None or not mesh.is_watertight or mesh.bounds[1, 2] < 4:
                mesh = None
                continue
            mesh.export(d / "target.stl")
            if validator_accepts(d):                             # 채점기가 실제로 읽고 모든 층이 비어 있지 않아야 채택
                break
            mesh = None
        if mesh is None:
            raise RuntimeError(f"{d.name} {kind}: 80번 만들어도 조건을 넘는 형상이 없음")
        (d / "desc.txt").write_text(desc, encoding="utf-8")
        print(f"{d.name}: [{KO[desc.split()[0].split('/')[0]]}] {desc} (시도 {tries})", flush=True)


if __name__ == "__main__":
    make(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
