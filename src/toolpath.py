"""층 단면 -> 용접 중심선(폴리라인) 생성 + 채점기와 같은 방식으로 IoU 자체 평가."""
from __future__ import annotations
import numpy as np
import shapely
from shapely import LineString, MultiPoint, Polygon, MultiPolygon, union_all
from shapely.geometry.base import BaseGeometry

MEDIAL_SPACING = 1.0   # 중심선용 경계 표본 간격
# 아래 값은 configure(cfg) 가 config 로 덮어쓴다(공개 모델 값이 기본값).
W = 8.0          # bead width
R = W / 2.0
QUAD = 8         # config.shape_validation.polygon_buffer_resolution
V_DEP = 8.72     # 적층 속도(예상 시간 계산용) = 기준 8mm/s 의 +9%
ARC = 0.0        # 조각당 아크 켜기+끄기 시간(예상 시간 계산용). 채점기가 보지 않으므로 기본 0
# 채점기는 구간 평균 속도가 기준의 ±speed_relative_tolerance(10%) 안이면 통과시킨다.
# 그 허용폭의 90%까지 쓴다(나머지 10% = 1%p 는 float32 좌표·소수 6자리 시각 반올림 여유).
SPEED_USE = 0.9
# 동심원 고리 간격 = 비드 폭 × 비율(비드 폭 8mm 에서 7.0·7.5·8.0mm)
STEP_RATIOS = (0.875, 0.9375, 1.0)
MAIN_RATIO = 0.9375


def fast_speed(cfg, v: float) -> float:
    """기준 속도 v 를 채점기 허용 오차 안에서 가장 빠르게(허용폭의 SPEED_USE 만큼) 올린 값."""
    return float(v) * (1.0 + SPEED_USE * float(cfg.validation.speed_relative_tolerance))


def configure(cfg, arc: bool = False) -> None:
    """config 의 비드 폭·버퍼 해상도·적층 속도·아크 시간으로 경로 생성 상수를 맞춘다.
    아크 켜기·끄기 대기는 채점기가 검사하지 않으므로 arc=True 일 때만 넣는다."""
    global W, R, QUAD, V_DEP, ARC, GLOBAL_IOU, LAYER_IOU_MIN
    pr = cfg.process
    W = float(pr.bead_width_mm)
    R = W / 2.0
    QUAD = int(cfg.shape_validation.polygon_buffer_resolution)
    V_DEP = fast_speed(cfg, pr.deposition_speed_mm_s)
    ARC = float((pr.arc_on_time_s or 0.0) + (pr.arc_off_time_s or 0.0)) if arc else 0.0
    sv = cfg.shape_validation
    GLOBAL_IOU = float(sv.minimum_overall_iou) + GLOBAL_IOU_MARGIN
    LAYER_IOU_MIN = float(sv.minimum_layer_iou) + LAYER_IOU_MARGIN


def main_step() -> float:
    return W * MAIN_RATIO


def _polys(g: BaseGeometry) -> list[Polygon]:
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon) and not p.is_empty]


def _rings(g: BaseGeometry, simplify: float) -> list[np.ndarray]:
    out = []
    for p in _polys(g):
        for ring in [p.exterior, *p.interiors]:
            r = LineString(ring.coords).simplify(simplify, preserve_topology=False)
            c = np.asarray(r.coords)[:, :2]
            if len(c) >= 2 and LineString(c).length > 1e-6:
                out.append(c)
    return out


def concentric(A: BaseGeometry, step: float, first: float | None = None, simplify: float = 0.05, join: str = "round") -> list[np.ndarray]:
    paths, d = [], (R if first is None else first)
    while True:
        inner = A.buffer(-d, quad_segs=QUAD, join_style=join, mitre_limit=10.0)
        if inner.is_empty or inner.area < 1e-3:
            break
        paths += _rings(inner, simplify)
        d += step
    return paths


def sliver(A: BaseGeometry, delta: float, simplify: float = 0.05) -> list[np.ndarray]:
    """얇은 벽용: 폭 2*delta 인 가느다란 띠의 테두리 = 사실상 중심선 왕복."""
    inner = A.buffer(-(R - delta), quad_segs=QUAD, join_style="round")
    return _rings(inner, simplify)


def medial(A: BaseGeometry, spacing: float = MEDIAL_SPACING, min_clear: float = 3.0, simplify: float = 0.1) -> list[np.ndarray]:
    """얇은 벽용 진짜 중심선: 경계점 Voronoi 모서리 중 벽 안쪽이고 경계에서 min_clear 이상 떨어진 것.
    교차점을 지나며 이어 그을 수 있게 trail(한붓그리기 조각) 단위로 묶어 아크 on/off 횟수를 줄인다."""
    if A.is_empty:
        return []
    pts = shapely.get_coordinates(shapely.segmentize(A.boundary, spacing))
    if len(pts) < 4:
        return []
    edges = shapely.get_parts(shapely.voronoi_polygons(MultiPoint(pts), only_edges=True, extend_to=A))
    if len(edges) == 0:
        return []
    shapely.prepare(A)
    inside = shapely.contains(A, edges)
    edges = edges[inside]
    if len(edges) == 0:
        return []
    bnd = A.boundary
    shapely.prepare(bnd)
    ends = shapely.get_coordinates(edges).reshape(-1, 2, 2)
    d0 = shapely.distance(shapely.points(ends[:, 0]), bnd)
    d1 = shapely.distance(shapely.points(ends[:, 1]), bnd)
    edges = edges[(d0 >= min_clear) & (d1 >= min_clear)]
    if len(edges) == 0:
        return []
    merged = shapely.line_merge(union_all(edges))
    lines = [np.asarray(g.coords)[:, :2] for g in shapely.get_parts(merged) if g.length > 1e-6]
    lines = _prune_spurs(lines, max_len=R)
    return _trails(lines, simplify)


def _prune_spurs(lines: list[np.ndarray], max_len: float, tol: float = 1e-3) -> list[np.ndarray]:
    """볼록 모서리 쪽으로 뻗는 짧은 잔가지(한쪽 끝이 막다른 짧은 선)를 제거하고 다시 잇는다.
    잔가지 길이는 비드 반폭 이하라 제거해도 둥근 비드 끝이 그 부분을 덮는다."""
    key = lambda p: (round(p[0] / tol), round(p[1] / tol))
    for _ in range(3):
        deg: dict = {}
        for ln in lines:
            for end in (0, -1):
                deg[key(ln[end])] = deg.get(key(ln[end]), 0) + 1
        keep = []
        removed = False
        for ln in lines:
            a, b = deg[key(ln[0])], deg[key(ln[-1])]
            L = float(np.sum(np.hypot(*np.diff(ln, axis=0).T)))
            if L < max_len and ((a == 1 and b >= 3) or (b == 1 and a >= 3)):
                removed = True
                continue
            keep.append(ln)
        lines = keep
        if not removed:
            break
        merged = shapely.line_merge(shapely.MultiLineString([LineString(l) for l in lines]))
        lines = [np.asarray(g.coords)[:, :2] for g in shapely.get_parts(merged) if g.length > 1e-6]
    return lines


def _trails(lines: list[np.ndarray], simplify: float, tol: float = 1e-3) -> list[np.ndarray]:
    """끝점이 만나는 폴리라인들을 탐욕적으로 이어 붙여 조각 수를 줄인다(홀수 차수 노드에서 시작)."""
    key = lambda p: (round(p[0] / tol), round(p[1] / tol))
    adj: dict = {}
    for i, ln in enumerate(lines):
        for end in (0, -1):
            adj.setdefault(key(ln[end]), []).append(i)
    used = [False] * len(lines)

    def walk(start_node):
        chain, node, prev_dir = [], start_node, None
        while True:
            cands = [i for i in adj.get(node, []) if not used[i]]
            if not cands:
                break
            best, best_cos, best_seq = None, -2.0, None
            for i in cands:
                ln = lines[i]
                seq = ln if key(ln[0]) == node else ln[::-1]
                v = seq[min(1, len(seq) - 1)] - seq[0]
                nv = np.linalg.norm(v)
                c = 1.0 if prev_dir is None or nv == 0 else float(np.dot(prev_dir, v / nv))
                if c > best_cos:
                    best, best_cos, best_seq = i, c, seq
            used[best] = True
            chain.append(best_seq if not chain else best_seq[1:])
            tail = best_seq[-1] - best_seq[-2] if len(best_seq) > 1 else None
            prev_dir = None if tail is None or np.linalg.norm(tail) == 0 else tail / np.linalg.norm(tail)
            node = key(best_seq[-1])
        return np.vstack(chain) if chain else None

    out = []
    odd = [n for n, v in adj.items() if len(v) % 2 == 1]
    for n in odd + list(adj.keys()):
        while any(not used[i] for i in adj[n]):
            c = walk(n)
            if c is None:
                break
            c = np.asarray(LineString(c).simplify(simplify, preserve_topology=False).coords)[:, :2]
            if len(c) >= 2:
                out.append(c)
    return out


def est_time(sc: dict, v_dep: float | None = None, arc: float | None = None) -> float:
    v_dep = V_DEP if v_dep is None else v_dep
    arc = ARC if arc is None else arc
    return sc["length"] / v_dep + arc * sc["pieces"]


def deposit(paths: list[np.ndarray]) -> BaseGeometry:
    if not paths:
        return Polygon()
    B = [LineString(p).buffer(R, quad_segs=QUAD, cap_style="round", join_style="round") for p in paths]
    try:
        return union_all(B)
    except shapely.errors.GEOSException:
        # 나선처럼 자기 자신과 닿는 경로는 합집합이 실패할 수 있다 → 격자에 맞춰 다시 시도
        return union_all([shapely.make_valid(b) for b in B], grid_size=1e-6)


def _inter(a: BaseGeometry, b: BaseGeometry) -> BaseGeometry:
    """교집합. 빈틈 영역처럼 버퍼를 거친 도형이 살짝 깨져 GEOS 가 실패하면 도형을 바로잡고 격자에 맞춰 다시 계산한다."""
    try:
        return a.intersection(b)
    except shapely.errors.GEOSException:
        return shapely.intersection(shapely.make_valid(a), shapely.make_valid(b), grid_size=1e-6)


def score(A: BaseGeometry, paths: list[np.ndarray]) -> dict:
    Dp = deposit(paths)
    inter = A.intersection(Dp).area
    union = A.union(Dp).area
    return {"iou": inter / union if union else 0.0,
            "inter": inter, "union": union,
            "coverage": inter / A.area if A.area else 0.0,
            "overfill": (Dp.area - inter) / A.area if A.area else 0.0,
            "length": float(sum(LineString(p).length for p in paths)),
            "pieces": len(paths),
            "vertices": int(sum(len(p) for p in paths))}


GAP_SPACING = 3.0      # 빈틈 중심선용 경계 표본 간격(크면 Voronoi 가 빨라진다)
IOU_TARGET = 0.92      # 층별 기준 0.80 / 전체 0.90 보다 여유 있게
OVERFILL_MAX = 0.25    # 기준 0.30 보다 여유 있게


def _candidates(A: BaseGeometry, legacy: bool = False, spiral: bool = True) -> list[tuple[str, list[np.ndarray]]]:
    """한 층 단면의 경로 후보들 (이름, 폴리라인 목록)."""
    cands = []
    s_main = main_step()
    if not legacy:
        inner = A.buffer(-R, quad_segs=QUAD)
        if inner.area < 0.5 * A.area:                 # 얇은 벽이 대부분인 층
            cands.append(("medial", medial(A)))
        else:                                         # 두꺼운 영역 + 남은 얇은 틈은 중심선으로 보강
            base = concentric(A, s_main, join="mitre")
            gap = A.difference(deposit(base)).buffer(-0.5).buffer(0.5)
            if gap.area > 4.0:
                cands.append((f"mitre{s_main}+medial", base + medial(_inter(gap.buffer(R * 0.5), A), spacing=GAP_SPACING)))
    for s in tuple(W * r for r in STEP_RATIOS):
        rings = {jn: concentric(A, s, join=jn) for jn in ("round", "mitre")}
        cands.append((f"concentric{s}", rings["round"]))
        cands.append((f"mitre{s}", rings["mitre"]))
        # 고리를 이어 붙인 나선: 조각 수가 줄어 아크 재점화(조각당 2초)가 준다.
        # 다만 조각이 길어 로봇끼리 나눠 갖기 어려워지는 형상도 있어(04 +8.9%) 켜고 끌 수 있게 둔다.
        # 간격 7.0 은 표본에서 한 번도 뽑히지 않아 나선 후보에서 뺐다(경로 계산 시간 절약).
        for jn in (("round", "mitre") if spiral and s >= s_main else ()):
            P = link_rings(rings[jn], s)
            cands.append((f"linked{s}{jn}", P))
            gap = A.difference(deposit(P)).buffer(-0.5).buffer(0.5)
            if gap.area > 4.0:
                cands.append((f"linked{s}{jn}+medial", P + medial(_inter(gap.buffer(R * 0.5), A), spacing=GAP_SPACING)))
    for dl in (0.15, 0.3, 0.5):
        cands.append((f"sliver{dl}", sliver(A, dl)))
    # 두꺼운 영역은 동심원, 거기서 남은 얇은 틈은 sliver 로 보강
    base = concentric(A, s_main)
    gap = A.difference(deposit(base)).buffer(0)
    if gap.area > 1.0:
        extra = [p for p in sliver(A, 0.3) if _inter(LineString(p).buffer(R), gap).area > 0.5 * LineString(p).length]
        cands.append((f"concentric{s_main}+sliver", base + extra))
    # 모서리 보강: 남은 빈틈의 각 조각 중심선을 짧게 추가
    for name0, P0 in list(cands):
        if not name0.startswith("mitre"):
            continue
        gap = A.difference(deposit(P0)).buffer(-0.5).buffer(0.5)
        extra = []
        for g in _polys(gap):
            if g.area < 4.0:
                continue
            c = np.asarray(g.centroid.coords)[0]
            mrr = g.minimum_rotated_rectangle
            xy = np.asarray(mrr.exterior.coords)[:4]
            e1, e2 = xy[1] - xy[0], xy[2] - xy[1]
            ax = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
            L = max(np.linalg.norm(ax) - W, 0.6)
            u = ax / (np.linalg.norm(ax) + 1e-12)
            extra.append(np.array([c - u * L / 2, c + u * L / 2]))
        if extra:
            cands.append((name0 + "+fill", P0 + extra))
    return cands


def best_paths(A: BaseGeometry, legacy: bool = False, spiral: bool = True, both: bool = False):
    """후보 경로들 중 형상 기준(여유 포함)을 만족하는 것 가운데 예상 적층시간이 가장 짧은 것.
    만족하는 후보가 없으면 IoU 가 가장 높은 것."""
    cands = _candidates(A, legacy, spiral)
    # both=True: 후보를 한 번만 평가해 '나선 포함 최선'과 '고리만 최선'을 함께 돌려준다.
    # 포트폴리오는 두 경로 묶음을 다 쓰는데, 따로 두 번 계산하면 고리 후보를 두 번 평가하게 된다.
    best, best_ring = None, None
    for name, P in cands:
        if not P:
            continue
        sc = score(A, P)
        if legacy:
            key = (sc["overfill"] <= 0.30, round(sc["iou"], 3), -sc["length"])
        else:
            ok = sc["iou"] >= IOU_TARGET and sc["overfill"] <= OVERFILL_MAX
            key = (ok, -est_time(sc) if ok else sc["iou"])
        if best is None or key > best[0]:
            best = (key, P, sc, name)
        if both and not name.startswith("linked") and (best_ring is None or key > best_ring[0]):
            best_ring = (key, P, sc, name)
    if both:
        return (best[1], best[2], best[3]), (best_ring[1], best_ring[2], best_ring[3])
    return best[1], best[2], best[3]


# 전체 IoU 예산으로 층별 경로 고르기
# 채점기의 전체 IoU 는 '층별 교집합 면적 합 ÷ 층별 합집합 면적 합'이다(층 높이가 같으므로).
# 그래서 모든 층이 IOU_TARGET 을 넘길 필요 없이, 합계가 기준을 넘기면 된다. 얇은 벽처럼 비드 줄 수가
# 정수로 끊기는 층(06 의 18mm 벽: 2줄이면 IoU 0.89, 3줄이면 0.99)은 짧은 경로를 쓰고, 그 부족분을
# 여유 있는 층이 메운다. 층별 최저선(LAYER_IOU_MIN)은 채점기의 층별 기준보다 여유 있게 지킨다.
GLOBAL_IOU = 0.92        # 전체 기준 0.90 + 여유 (configure 가 config 로 다시 맞춘다)
LAYER_IOU_MIN = 0.85     # 층별 기준 0.80 + 여유
GLOBAL_IOU_MARGIN = 0.02
LAYER_IOU_MARGIN = 0.05


def frontier(A: BaseGeometry, spiral: bool = True, both: bool = False):
    """층 하나의 후보 중 '더 오래 걸리면 IoU 여유도 더 큰' 것만 남긴 목록 [(예상시간, 교집합, 합집합, 경로, IoU)].
    층별 최저선·넘침 기준을 넘는 후보가 없으면 IoU 가 가장 높은 후보 하나만 둔다.
    both=True 면 (나선 포함 목록, 고리만 목록) 두 개를 돌려준다."""
    rows = []
    for name, P in _candidates(A, spiral=spiral):
        if not P:
            continue
        sc = score(A, P)
        rows.append((name, est_time(sc), sc, P))

    def pick(rs):
        ok = [r for r in rs if r[2]["iou"] >= LAYER_IOU_MIN and r[2]["overfill"] <= OVERFILL_MAX]
        if not ok:
            r = max(rs, key=lambda r: r[2]["iou"])
            return [(r[1], r[2]["inter"], r[2]["union"], r[3], r[2]["iou"])]
        ok.sort(key=lambda r: (r[1], -r[2]["iou"]))
        out, best_slack = [], -float("inf")
        for _, t, sc, P in ok:
            slack = sc["inter"] - GLOBAL_IOU * sc["union"]
            if slack > best_slack + 1e-9:
                out.append((t, sc["inter"], sc["union"], P, sc["iou"]))
                best_slack = slack
        return out

    if both:
        return pick(rows), pick([r for r in rows if not r[0].startswith("linked")])
    return pick(rows)


def select_global(fronts: dict) -> dict:
    """층마다 frontier 하나씩 받아 전체 IoU ≥ GLOBAL_IOU 를 지키며 예상 적층시간 합이 가장 작게 고른다.

    전체 IoU ≥ G  ⇔  Σ(교집합 − G·합집합) ≥ 0 이라 층별 '여유'의 합 문제가 된다. 모두 가장 빠른 후보에서
    시작해, 여유가 모자라면 '늘어나는 시간당 여유가 가장 많이 느는' 층부터 한 단계씩 올린다.
    돌려주는 값: {층: (경로, IoU, 여유)}  여유 = 교집합 − G·합집합 (스케줄러가 경로를 바꿀 때 예산을 지키는 데 쓴다)"""
    import heapq
    G = GLOBAL_IOU
    cur = {k: 0 for k in fronts}
    slack = sum(f[0][1] - G * f[0][2] for f in fronts.values())

    def best_step(k):
        f, i = fronts[k], cur[k]
        s0, t0 = f[i][1] - G * f[i][2], f[i][0]
        best = None
        for j in range(i + 1, len(f)):
            ds, dt = f[j][1] - G * f[j][2] - s0, f[j][0] - t0
            if ds <= 0:
                continue
            ratio = ds / max(dt, 1e-9)
            if best is None or ratio > best[0]:
                best = (ratio, j, ds)
        return best

    heap = []
    for k in sorted(fronts):
        b = best_step(k)
        if b:
            heapq.heappush(heap, (-b[0], k, b[1], b[2]))
    while slack < 0 and heap:
        _, k, j, ds = heapq.heappop(heap)
        if j <= cur[k]:
            continue
        cur[k] = j
        slack += ds
        b = best_step(k)
        if b:
            heapq.heappush(heap, (-b[0], k, b[1], b[2]))
    out = {}
    for k in fronts:
        t, inter, union, P, iou = fronts[k][cur[k]]
        out[k] = (P, iou, inter - G * union)
    return out


def linked(A: BaseGeometry, step: float | None = None, join: str = "mitre", simplify: float = 0.05) -> list[np.ndarray]:
    """동심원 고리들을 이어 붙여 '나선'처럼 한 붓으로 만든다.

    고리마다 아크를 껐다 켜면 한 번에 2초가 든다(05번 기준 makespan 의 6.3%). 고리 간격(7.5mm)이
    비드 폭(8mm)보다 좁으므로, 이웃 고리의 가장 가까운 점끼리 이어도 그 구간은 이미 덮이는 자리다.
    """
    step = main_step() if step is None else step
    return link_rings(concentric(A, step, join=join, simplify=simplify), step)


def link_rings(rings: list[np.ndarray], step: float | None = None) -> list[np.ndarray]:
    """이미 만들어 둔 동심원 고리들을 가까운 것끼리 이어 붙인다(고리 재계산을 피하려고 분리)."""
    step = main_step() if step is None else step
    if len(rings) < 2:
        return rings
    rings = [r[:-1] if len(r) > 2 and np.allclose(r[0], r[-1]) else r for r in rings]   # 닫힌 고리는 중복점 제거
    out, used = [], [False] * len(rings)
    while not all(used):
        i = next(j for j, u in enumerate(used) if not u)          # 남은 것 중 가장 바깥(바깥부터 생성됨)
        used[i] = True
        cur = [np.vstack([rings[i], rings[i][:1]])]               # 고리는 닫아서 시작
        end = cur[-1][-1]
        while True:
            best, bd, bidx = None, step * 1.8, None
            for j, r in enumerate(rings):
                if used[j]:
                    continue
                d = np.hypot(*(r - end).T)
                m = int(np.argmin(d))
                if d[m] < bd:
                    bd, best, bidx = d[m], np.roll(r, -m, axis=0), j
            if best is None:
                break
            used[bidx] = True
            cur.append(np.vstack([best, best[:1]]))               # 이어 붙인 고리도 닫는다
            end = cur[-1][-1]
        out.append(np.vstack(cur))
    return out
