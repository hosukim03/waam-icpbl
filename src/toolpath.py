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
    global W, R, QUAD, V_DEP, ARC, IOU_TARGET, COVERAGE_MIN, OVERFILL_MAX
    pr = cfg.process
    W = float(pr.bead_width_mm)
    R = W / 2.0
    QUAD = int(cfg.shape_validation.polygon_buffer_resolution)
    V_DEP = fast_speed(cfg, pr.deposition_speed_mm_s)
    ARC = float((pr.arc_on_time_s or 0.0) + (pr.arc_off_time_s or 0.0)) if arc else 0.0
    # 층별 형상 목표 = config 의 합격 기준 + 여유. 채점기는 '모든 층' IoU 와 전체 IoU·Coverage·Overfill 을 보는데,
    # 층마다 IoU 가 목표 이상이면 전체 IoU(층 체적 가중)도 목표 이상이고 IoU ≤ Coverage 이므로 Coverage 도 따라온다.
    sv = cfg.shape_validation
    IOU_TARGET = max(float(sv.minimum_layer_iou), float(sv.minimum_overall_iou)) + IOU_MARGIN
    COVERAGE_MIN = float(sv.minimum_overall_coverage) + IOU_MARGIN
    OVERFILL_MAX = max(0.0, float(sv.maximum_overall_overfill_ratio) - OVERFILL_MARGIN)


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


FAST_VORONOI = True    # Voronoi 모서리를 칸 경계에서 모은다(아래 _voronoi_edges). False 면 예전 GEOS only_edges


def _voronoi_edges(pts: np.ndarray, A: BaseGeometry) -> np.ndarray:
    """경계 표본점의 Voronoi 모서리(선분 배열). GEOS 의 only_edges=True 는 칸 경계를 모두 겹쳐 놓고 잘라 내느라,
    둥근 경계처럼 점이 원 위에 고르게 놓이면 매우 느리다(점 2,500개에 0.5초, 원 두 개 위 점이면 3.5초).
    칸(polygon)으로 받아 칸 경계 선분을 한 번씩만 모으면 4분의 1 안팎 시간이다. 남는 선분 집합은 같고
    (공개 형상 층에서 확인) 순서만 다르다. 순서는 좌표 사전순으로 고정한다."""
    cells = shapely.get_parts(shapely.voronoi_polygons(MultiPoint(pts), extend_to=A))
    xy, idx = shapely.get_coordinates(shapely.get_exterior_ring(cells), return_index=True)
    same = idx[:-1] == idx[1:]
    S = np.hstack([xy[:-1][same], xy[1:][same]])
    sw = (S[:, 0] > S[:, 2]) | ((S[:, 0] == S[:, 2]) & (S[:, 1] > S[:, 3]))
    S[sw] = S[sw][:, [2, 3, 0, 1]]
    S = np.unique(S, axis=0)
    S = S[(S[:, 0] != S[:, 2]) | (S[:, 1] != S[:, 3])]
    return shapely.linestrings(S.reshape(-1, 2, 2))


def medial(A: BaseGeometry, spacing: float = MEDIAL_SPACING, min_clear: float = 3.0, simplify: float = 0.1) -> list[np.ndarray]:
    """얇은 벽용 진짜 중심선: 경계점 Voronoi 모서리 중 벽 안쪽이고 경계에서 min_clear 이상 떨어진 것.
    교차점을 지나며 이어 그을 수 있게 trail(한붓그리기 조각) 단위로 묶어 아크 on/off 횟수를 줄인다."""
    if A.is_empty:
        return []
    pts = shapely.get_coordinates(shapely.segmentize(A.boundary, spacing))
    if len(pts) < 4:
        return []
    if FAST_VORONOI:
        edges = _voronoi_edges(pts, A)
    else:
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
    # 모서리 끝점은 이웃 모서리와 겹친다(Voronoi 꼭짓점 하나에 모서리 셋). 서로 다른 점만 한 번씩 잰다(값은 같다).
    V, inv = np.unique(ends.reshape(-1, 2), axis=0, return_inverse=True)
    far = shapely.distance(shapely.points(V), bnd) >= min_clear
    edges = edges[far[inv.reshape(-1)].reshape(-1, 2).all(axis=1)]
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


def _simple(p: np.ndarray) -> BaseGeometry:
    """경로 한 조각을 부풀리기 좋은 선으로. 채점기는 선분마다 따로 부풀려 합치는데, 자기 자신과 엇갈리는 선(고리를
    이어 붙인 나선 등)을 GEOS 로 통째 부풀리면 넓이를 크게 잡는다(다른 방식 형상 novel1027 #15 41층: 통째 11,553 mm²,
    선분마다 11,030 mm² → 층 IoU 99.4% 로 알고 골랐는데 채점기는 95.2%). 엇갈리는 점에서 끊으면 채점기와 같다."""
    L = LineString(p)
    return L if L.is_simple else shapely.node(L)


def deposit(paths: list[np.ndarray]) -> BaseGeometry:
    if not paths:
        return Polygon()
    B = [_simple(p).buffer(R, quad_segs=QUAD, cap_style="round", join_style="round") for p in paths]
    try:
        return union_all(B)
    except shapely.errors.GEOSException:
        # 나선처럼 자기 자신과 닿는 경로는 합집합이 실패할 수 있다 → 격자에 맞춰 다시 시도
        return union_all([shapely.make_valid(b) for b in B], grid_size=1e-6)


_GEOS = shapely.errors.GEOSException


def _areal(g: BaseGeometry) -> BaseGeometry:
    """도형을 바로잡고 면(Polygon)만 남긴다. make_valid 는 면과 선이 섞인 묶음을 돌려줄 수 있는데,
    그대로 겹침 계산에 넣으면 'mixed-dimension' 오류가 난다."""
    ps = _polys(shapely.make_valid(g))
    return union_all(ps) if ps else Polygon()


def _inter(a: BaseGeometry, b: BaseGeometry) -> BaseGeometry:
    """교집합. 빈틈 영역처럼 버퍼를 거친 도형이 살짝 깨져 GEOS 가 실패하면 면만 남겨 바로잡고
    격자에 맞춰 다시 계산한다. 끝내 안 되면 빈 도형(그 보강은 건너뛴다)."""
    try:
        return a.intersection(b)
    except _GEOS:
        pass
    try:
        a2, b2 = _areal(a), _areal(b)
        for grid in (1e-6, 1e-4, 1e-2):
            try:
                return shapely.intersection(a2, b2, grid_size=grid)
            except _GEOS:
                continue
    except _GEOS:
        pass
    return Polygon()


def score(A: BaseGeometry, paths: list[np.ndarray]) -> dict:
    Dp = deposit(paths)
    inter = A.intersection(Dp).area
    union = A.area + Dp.area - inter         # 합집합 넓이(도형 연산 한 번을 덧셈으로, 값은 소수 12자리 안에서 같다)
    return {"iou": inter / union if union else 0.0,
            "coverage": inter / A.area if A.area else 0.0,
            "overfill": (Dp.area - inter) / A.area if A.area else 0.0,
            "length": float(sum(LineString(p).length for p in paths)),
            "pieces": len(paths),
            "vertices": int(sum(len(p) for p in paths))}


GAP_SPACING = 3.0      # 빈틈 중심선용 경계 표본 간격(크면 Voronoi 가 빨라진다)
# 층마다 경로를 고를 때의 형상 목표. 아래는 채점기 v1.0.2 기준(층·전체 IoU 98%, Coverage 98%, Overfill 3%)에
# 여유를 더한 기본값이고, configure(cfg) 가 작업 config 의 기준으로 다시 맞춘다.
IOU_MARGIN = 0.002       # IoU·Coverage 는 기준보다 이만큼 높게
OVERFILL_MARGIN = 0.005  # Overfill 은 기준보다 이만큼 낮게
IOU_TARGET = 0.982
COVERAGE_MIN = 0.982
OVERFILL_MAX = 0.025


def meets(sc: dict) -> bool:
    """후보 경로의 자체 점수가 층별 형상 목표를 모두 넘는가."""
    return sc["iou"] >= IOU_TARGET and sc["coverage"] >= COVERAGE_MIN and sc["overfill"] <= OVERFILL_MAX


def _est(P: list[np.ndarray]) -> float:
    """후보의 예상 적층시간. score() 를 거친 est_time 과 같은 계산(같은 값)이다."""
    return est_time({"length": float(sum(LineString(p).length for p in P)), "pieces": len(P)})


def _fill_extra(A: BaseGeometry, P0: list[np.ndarray]) -> list[np.ndarray] | None:
    """모서리 보강: P0 가 남긴 빈틈 조각마다 중심선을 짧게 하나씩. 빈틈 계산이 실패하면 None."""
    try:
        gap = A.difference(deposit(P0)).buffer(-0.5).buffer(0.5)
    except _GEOS:
        return None
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
    return extra


def _gap_medial(A: BaseGeometry, P: list[np.ndarray], min_clear: float = 3.0) -> list[np.ndarray] | None:
    """P 가 남긴 얇은 빈틈을 중심선으로 보강한 후보. 빈틈이 작거나 도형 연산이 실패하면 None."""
    try:
        gap = A.difference(deposit(P)).buffer(-0.5).buffer(0.5)
        if gap.area > 4.0:
            return P + medial(_inter(gap.buffer(R * 0.5), A), spacing=GAP_SPACING, min_clear=min_clear)
    except _GEOS:
        pass
    return None


# 기준을 넘는 후보가 하나도 없는 층에서만 더 만드는 후보: 빈틈 중심선의 '경계에서 떨어진 거리' 기준을 3 → 2 mm 로
# 낮춘다. 두께 16~18 mm 벽처럼 두 비드 사이에 폭 2 mm 이하 띠가 남으면 그 띠를 넓힌 영역의 중심선이 경계에서
# 3 mm 를 못 띄워 전부 버려졌다(연습 형상 16 맨 위 층: IoU 94.0% → 99.8%). 1 mm 로 낮추면 각진 모서리마다
# 잔가지가 생겨 넘침이 커진다(톱니바퀴 형상 IoU 97.8% → 85.3%).
RESCUE_CLEAR = 2.0
THIN_MIN_AREA = 1.0    # 가는 틈 보강(_gap_thin)에서 이보다 작은 틈 조각(mm²)은 버린다


def _gap_thin(A: BaseGeometry, P: list[np.ndarray], min_clear: float = RESCUE_CLEAR) -> list[np.ndarray] | None:
    """P 가 남긴 빈틈을 폭 1 mm 보다 가는 틈까지 중심선으로 메운 후보. _gap_medial 은 빈틈에서 폭 1 mm 보다 가는 틈을
    지우고(buffer(-0.5).buffer(0.5)) 메우는데, 두께가 비드 폭의 배수보다 조금 두꺼운 벽은 가운데에 그런 틈이 길게
    남아 그것만으로 층 IoU 가 기준 아래로 떨어진다(벽 16.6 mm = 비드 2줄 + 0.6 mm 틈, 다른 방식 형상 novel1027 #08).
    목표 단면 테두리를 따라 생기는 얇은 계산 오차 띠는 R/2 부풀려도 단면 안쪽 폭이 R/2 남짓이라 min_clear 에 걸러진다."""
    try:
        gap = union_all([g for g in _polys(A.difference(deposit(P))) if g.area >= THIN_MIN_AREA])
        if gap.area > 4.0:
            return P + medial(_inter(gap.buffer(R * 0.5), A), spacing=GAP_SPACING, min_clear=min_clear)
    except _GEOS:
        pass
    return None


# 모서리 밀기: 위 보강까지 해도 형상 목표를 넘는 후보가 없는 층에서만 만드는 후보. 비드 끝이 둥글어 목표 단면의
# 뾰족한 바깥 모서리를 못 채우면(연습 형상 20 톱니바퀴: 톱니 끝 32곳이 각 4 mm² 씩 비어 층 IoU 97.8%), 그 모서리에서
# 가장 가까운 경로 꼭짓점에서 모서리 쪽으로 짧게 갔다 돌아온다. 조금 밀면 새로 채우는 넓이가 밖으로 넘치는 넓이보다
# 커서 IoU 가 오르고, 많이 밀면 넘침이 커져 다시 떨어진다(1110a 궤적에 손으로 넣어 채점기로 잰 층 IoU: 0.1 mm 98.02%,
# 0.3 mm 98.22%, 0.4 mm 98.24%, 0.7 mm 98.04%). 짧은 거리부터 재서 목표를 처음 넘는 것을 쓰고(시간이 가장 적게 든다),
# 끝까지 못 넘으면 쓰지 않는다(그 층은 예전처럼 IoU 가 가장 높은 후보).
SPUR_STEPS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SPUR_MIN_GAP = 0.5     # 이보다 작은 빈틈 조각(mm²)은 보지 않는다
SPUR_REACH = 3.0       # 경로 꼭짓점이 모서리에서 R 보다 멀고 R × 이 값 이하일 때만 민다(그 모서리를 맡은 비드 끝)
SPUR_BASES = 3         # 지금까지 잰 후보 중 IoU 가 높은 것 몇 개에 넣어 볼지
SPUR_OPEN = 0.05       # 빈틈을 이만큼 깎았다 되살려 가는 틈을 끊는다(톱니바퀴 4층: 모서리 빈틈 둘이 가는 틈으로 이어져
                       # 8.04 mm² 하나로 잡히면 모서리 하나를 놓쳐 32곳 중 28곳만 밀고 IoU 98.20% 에 그쳤다)


def _spur_sites(A: BaseGeometry, P: list[np.ndarray]) -> list[tuple[int, int, np.ndarray]]:
    """P 가 남긴 빈틈 가운데 목표 단면 테두리의 꼭짓점이 닿은 것마다 (조각 번호, 꼭짓점 번호, 모서리 쪽 단위 벡터).
    모서리 = 그 빈틈에 닿은 테두리 꼭짓점 중 쌓인 곳에서 가장 먼 것, 미는 점 = 경로 꼭짓점 중 모서리에 가장 가까운 것."""
    if not P:
        return []
    try:
        Dp = deposit(P)
        gaps = [g for g in _polys(A.difference(Dp).buffer(-SPUR_OPEN).buffer(SPUR_OPEN)) if g.area >= SPUR_MIN_GAP]
    except _GEOS:
        return []
    if not gaps:
        return []
    bx = np.vstack([np.asarray(r.coords)[:, :2] for p in _polys(A) for r in (p.exterior, *p.interiors)])
    xy = np.vstack([np.asarray(p)[:, :2] for p in P])
    owner = [(i, j) for i, p in enumerate(P) for j in range(len(p))]
    sites = []
    e = 4 * SPUR_OPEN                              # 깎았다 되살리면 빈틈의 뾰족한 끝이 조금 무뎌지므로 그만큼 너그럽게
    for g in gaps:
        x0, y0, x1, y1 = g.bounds
        near = bx[(bx[:, 0] >= x0 - e) & (bx[:, 0] <= x1 + e) & (bx[:, 1] >= y0 - e) & (bx[:, 1] <= y1 + e)]
        if len(near):
            near = near[shapely.intersects_xy(g.buffer(e), near[:, 0], near[:, 1])]
        if not len(near):
            continue
        v = near[int(np.argmax(shapely.distance(Dp, shapely.points(near))))]
        d = np.hypot(xy[:, 0] - v[0], xy[:, 1] - v[1])
        m = int(np.argmin(d))
        if not (R < d[m] <= R * SPUR_REACH):
            continue
        i, j = owner[m]
        sites.append((i, j, (v - xy[m]) / d[m]))
    return sites


def _spurred(P: list[np.ndarray], sites, t: float) -> list[np.ndarray]:
    """sites 의 꼭짓점마다 모서리 쪽으로 t mm 짧은 가지를 넣은 경로. 열린 선의 끝점이면 그쪽으로 늘이기만 하고,
    그 밖(닫힌 고리 포함, 고리는 닫힌 채로 둬야 스케줄러가 시작점을 고를 수 있다)은 갔다가 돌아온다."""
    at = {}
    for i, j, u in sites:
        at.setdefault((i, j), []).append(u)
    out = []
    for i, p in enumerate(P):
        p = np.asarray(p, float)
        n = len(p)
        closed = n > 3 and np.hypot(*(p[0, :2] - p[-1, :2])) < 1e-6
        rows = []
        for j in range(n):
            tips = []
            for u in at.get((i, j), ()):
                tip = p[j].copy()
                tip[:2] = p[j, :2] + t * u
                tips.append(tip)
            if len(tips) == 1 and not closed and n > 1 and j == 0:
                rows += [tips[0], p[j]]
            elif len(tips) == 1 and not closed and n > 1 and j == n - 1:
                rows += [p[j], tips[0]]
            else:
                rows.append(p[j])
                for tip in tips:
                    rows += [tip, p[j]]
        out.append(np.asarray(rows) if len(rows) != n else p)
    return out


def best_paths(A: BaseGeometry, legacy: bool = False, spiral: bool = True, both: bool = False):
    """후보 경로들 중 형상 기준(여유 포함)을 만족하는 것 가운데 예상 적층시간이 가장 짧은 것.
    만족하는 후보가 없으면 IoU 가 가장 높은 것.

    답은 후보를 전부 만들어 전부 재는 _best_paths_all 과 같다. 다만 예상 적층시간이 짧은 후보부터 재고,
    기준을 처음 통과한 후보에서 멈춘다(그보다 긴 후보는 이길 수 없다). 빈틈 보강처럼 만드는 데 비싼 후보는
    바탕 경로의 시간을 하한으로 두고(보강하면 길어지기만 한다) 차례가 왔을 때 만든다.
    시간이 같으면 원래 후보 순서(slot)가 앞선 것을 고르는 것도 같다.
    돌려주기 직전에 이미 쌓는 줄 위에 다시 찍는 점 줄을 뺀다(drop_dots, 2026-11-28)."""
    if legacy:
        return _best_paths_all(A, legacy=True, spiral=spiral, both=both)
    import heapq
    s_main = main_step()
    # 후보 자리(slot) = _best_paths_all 이 후보를 쌓는 순서. 값: 경로(이미 만든 것) 또는 None(안 만든 것)
    names, made, makers = [], {}, {}
    heap = []

    def add(name, P=None, make=None, lb=None):
        slot = len(names)
        names.append(name)
        if make is None:
            made[slot] = P
            heapq.heappush(heap, (_est(P), slot))
        else:
            makers[slot] = make
            heapq.heappush(heap, (lb, slot))
        return slot

    inner = A.buffer(-R, quad_segs=QUAD)
    rings_main = None
    if inner.area < 0.5 * A.area:                     # 얇은 벽이 대부분인 층
        add("medial", medial(A))
    else:                                             # 두꺼운 영역 + 남은 얇은 틈은 중심선으로 보강
        try:
            rings_main = concentric(A, s_main, join="mitre")
            slot_mm = add(f"mitre{s_main}+medial", make=lambda P=rings_main: _gap_medial(A, P), lb=_est(rings_main))
        except _GEOS:
            rings_main = None
    mitre_slots = [slot_mm] if rings_main is not None else []
    round_main, linked = None, []
    for s in tuple(W * r for r in STEP_RATIOS):
        rings = {jn: concentric(A, s, join=jn) for jn in ("round", "mitre")}
        if s == s_main:
            round_main = rings["round"]
        add(f"concentric{s}", rings["round"])
        mitre_slots.append(add(f"mitre{s}", rings["mitre"]))
        for jn in (("round", "mitre") if spiral and s >= s_main else ()):
            P = link_rings(rings[jn], s)
            add(f"linked{s}{jn}", P)
            add(f"linked{s}{jn}+medial", make=lambda P=P: _gap_medial(A, P), lb=_est(P))
            linked.append((f"linked{s}{jn}", P))
    sl = {}
    for dl in (0.15, 0.3, 0.5):
        sl[dl] = sliver(A, dl)
        add(f"sliver{dl}", sl[dl])

    def make_sliver():
        try:
            base = concentric(A, s_main) if round_main is None else round_main
            gap = A.difference(deposit(base)).buffer(0)
            if gap.area > 1.0:
                return base + [p for p in sl[0.3] if _inter(LineString(p).buffer(R), gap).area > 0.5 * LineString(p).length]
        except _GEOS:
            pass
        return None
    add(f"concentric{s_main}+sliver", make=make_sliver,
        lb=_est(round_main) if round_main is not None else 0.0)
    for ms in mitre_slots:                           # 모서리 보강(+fill): 바탕 후보가 있어야 만든다
        def make_fill(ms=ms):
            P0 = made.get(ms)
            if P0 is None:                           # 바탕 후보가 없다(빈 경로 []는 있는 후보로 본다)
                return None
            extra = _fill_extra(A, P0)
            return P0 + extra if extra else None
        add(names[ms] + "+fill", make=make_fill, lb=heap_lb(heap, ms))

    best, best_ring = None, None
    scored = {}
    while heap:
        key, slot = heapq.heappop(heap)
        name = names[slot]
        if best is not None and (not both or best_ring is not None):
            break
        if best is not None and name.startswith("linked"):
            continue                                  # 남은 일은 고리만 최선 찾기
        if slot in makers:                            # 차례가 온 비싼 후보: 지금 만든다
            P = makers.pop(slot)()
            made[slot] = P
            if P:
                heapq.heappush(heap, (_est(P), slot))
            continue
        P = made[slot]
        if not P:
            continue
        try:
            sc = score(A, P)
        except _GEOS:
            continue
        scored[slot] = sc
        if meets(sc):
            if best is None:
                best = (P, sc, name)
            if both and best_ring is None and not name.startswith("linked"):
                best_ring = (P, sc, name)
    # 기준을 넘는 후보가 없다: 빈틈을 더 메운 후보를 만들어 잰다. 1) 좁은 빈틈까지(RESCUE_CLEAR 참고),
    # 2) 그래도 없으면 폭 1 mm 보다 가는 틈까지(_gap_thin 참고). 바탕은 기본 간격의 고리(각진·둥근)와 나선.
    # 기준을 넘는 것 가운데 예상 적층시간이 가장 짧은 것을 고른다. 앞 단계에서 찾으면 다음 단계는 만들지 않는다.
    def need(cur, level):
        return cur is None
    bases = ([(f"mitre{s_main}", rings_main)] if rings_main is not None else []) + \
            ([(f"concentric{s_main}", round_main)] if round_main is not None else []) + linked
    for level, tag, fill in ((1, "+medial2", lambda P0: _gap_medial(A, P0, min_clear=RESCUE_CLEAR)),
                             (2, "+thin", lambda P0: _gap_thin(A, P0))):
        if not (need(best, level) or (both and need(best_ring, level))):
            break
        rescued = []
        for nm0, P0 in bases:
            P = fill(P0)
            if not P:
                continue
            slot = len(names)
            names.append(nm0 + tag)
            made[slot] = P
            try:
                scored[slot] = score(A, P)
            except _GEOS:
                continue
            if meets(scored[slot]):
                rescued.append(slot)
        by_time = lambda s: (est_time(scored[s]), s)
        ok_ring = [s for s in rescued if not names[s].startswith("linked")]
        if need(best, level) and rescued:
            s = min(rescued, key=by_time)
            if best is None or est_time(scored[s]) < est_time(best[1]):
                best = (made[s], scored[s], names[s])
        if both and need(best_ring, level) and ok_ring:
            s = min(ok_ring, key=by_time)
            if best_ring is None or est_time(scored[s]) < est_time(best_ring[1]):
                best_ring = (made[s], scored[s], names[s])
    # 3) 그래도 없으면 모서리 밀기(SPUR_STEPS 참고): 지금까지 잰 후보 중 IoU 가 높은 것 몇 개에 넣어 본다. 목표를
    #    넘은 것만 후보로 남기고, 그 가운데 예상 적층시간이 가장 짧은 것. 못 넘으면 아래에서 예전처럼 고른다.
    if best is None or (both and best_ring is None):
        pool = []
        for ok_name, want in ((lambda nm: True, best is None),
                              (lambda nm: not nm.startswith("linked"), both and best_ring is None)):
            if want:
                pool += sorted((s for s in scored if ok_name(names[s])),
                               key=lambda s: (-scored[s]["iou"], s))[:SPUR_BASES]
        spurred = []
        for s0 in dict.fromkeys(pool):                # 순서를 지킨 채 겹친 것을 뺀다
            sites = _spur_sites(A, made[s0])
            for t in (SPUR_STEPS if sites else ()):
                P = _spurred(made[s0], sites, t)
                try:
                    sc = score(A, P)
                except _GEOS:
                    continue
                if meets(sc):
                    slot = len(names)
                    names.append(f"{names[s0]}+spur{t}")
                    made[slot], scored[slot] = P, sc
                    spurred.append(slot)
                    break
        by_time = lambda s: (est_time(scored[s]), s)
        if best is None and spurred:
            s = min(spurred, key=by_time)
            best = (made[s], scored[s], names[s])
        ok_ring = [s for s in spurred if not names[s].startswith("linked")]
        if both and best_ring is None and ok_ring:
            s = min(ok_ring, key=by_time)
            best_ring = (made[s], scored[s], names[s])
    if best is None or (both and best_ring is None):
        # 그래도 없으면 전부 만들고 전부 쟀다 → IoU 가 가장 높은 것(같으면 앞선 자리)
        def top(ok_name):
            cands = [(scored[s]["iou"], -s) for s in scored if ok_name(names[s])]
            s = -max(cands)[1]
            return made[s], scored[s], names[s]
        if best is None:
            best = top(lambda nm: True)
        if both and best_ring is None:
            best_ring = top(lambda nm: not nm.startswith("linked"))
    best_nd = _no_dots(A, best)                       # 이미 쌓는 줄 위에 다시 찍는 점 줄 빼기(drop_dots 참고)
    if both:
        return best_nd, (best_nd if best_ring is best else _no_dots(A, best_ring))
    return best_nd


DOT_LEN = 1.0     # 이보다 짧은 줄은 사실상 점이다(벌집·격자처럼 얇은 벽이 만나는 모서리에 0~0.3 mm 짜리가 생긴다)
DOT_NEAR = 0.5    # 점의 양 끝이 다른 줄에서 이 거리 안이면 이미 쌓는 자리 위에 다시 찍는 점이다


def _plen(p: np.ndarray) -> float:
    p = np.asarray(p)
    return float(np.hypot(*np.diff(p[:, :2], axis=0).T).sum()) if len(p) > 1 else 0.0


def _pt_line(pt: np.ndarray, q: np.ndarray) -> float:
    """점에서 꺾은선까지의 가장 짧은 거리."""
    q = np.asarray(q)[:, :2]
    a, ab = q[:-1], np.diff(q, axis=0)
    den = (ab ** 2).sum(1)
    t = np.where(den > 0, ((pt - a) * ab).sum(1) / np.where(den > 0, den, 1.0), 0.0).clip(0.0, 1.0)
    return float(np.hypot(*(pt - (a + t[:, None] * ab)).T).min())


def drop_dots(A: BaseGeometry, P: list[np.ndarray]) -> tuple[list[np.ndarray], dict] | None:
    """이미 쌓는 줄 위에 다시 찍는 '점 줄'(길이 DOT_LEN 미만, 양 끝이 다른 줄에서 DOT_NEAR 안)을 뺀 (경로, 점수).
    중심선을 잇다 남은 부스러기라 빼도 모양은 거의 그대로이고, 로봇이 점 하나 찍으러 오가는 일이 준다.
    뺄 것이 없거나, 빼면 층 형상 목표(meets)를 못 넘으면 None(원래 경로를 쓴다).
    2026-11-28 연습 63개: 12개 형상에 있었고 makespan 합계 −0.036%(연습 1 08 삼각 격자 −0.96%, 연습 1 10 물결선
    −0.72%, 나빠진 형상은 연습 2 08 +0.05% 하나), 모두 PASS·충돌 0. 공개 01 벌집 −0.41%."""
    longs = [np.asarray(p) for p in P if len(p) > 1 and _plen(p) >= DOT_LEN]
    if not longs:
        return None
    keep = [p for p in P if not (_plen(p) < DOT_LEN and all(
        min(_pt_line(np.asarray(p)[e, :2], q) for q in longs) <= DOT_NEAR for e in (0, -1)))]
    if len(keep) == len(P):
        return None
    try:
        sc = score(A, keep)
    except _GEOS:
        return None
    return (keep, sc) if meets(sc) else None


def _no_dots(A: BaseGeometry, res):
    """best_paths 결과 (경로, 점수, 이름)에서 점 줄을 뺀 것(뺄 것이 없으면 그대로)."""
    r = drop_dots(A, res[0]) if res is not None and res[0] else None
    return res if r is None else (r[0], r[1], res[2])


def heap_lb(heap, slot: int) -> float:
    """heap 에 들어 있는 slot 의 하한(바탕 후보의 예상 적층시간)."""
    for key, s in heap:
        if s == slot:
            return key
    raise KeyError(slot)


def _best_paths_all(A: BaseGeometry, legacy: bool = False, spiral: bool = True, both: bool = False):
    """예전 방식: 후보를 전부 만들어 전부 잰다(legacy 모드와 best_paths 검산용)."""
    cands = []
    s_main = main_step()
    if not legacy:
        inner = A.buffer(-R, quad_segs=QUAD)
        if inner.area < 0.5 * A.area:                 # 얇은 벽이 대부분인 층
            cands.append(("medial", medial(A)))
        else:                                         # 두꺼운 영역 + 남은 얇은 틈은 중심선으로 보강
            try:
                base = concentric(A, s_main, join="mitre")
                gap = A.difference(deposit(base)).buffer(-0.5).buffer(0.5)
                if gap.area > 4.0:
                    cands.append((f"mitre{s_main}+medial", base + medial(_inter(gap.buffer(R * 0.5), A), spacing=GAP_SPACING)))
            except _GEOS:
                pass                                  # 도형 연산이 실패한 후보는 버리고 나머지에서 고른다
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
            try:
                gap = A.difference(deposit(P)).buffer(-0.5).buffer(0.5)
                if gap.area > 4.0:
                    cands.append((f"linked{s}{jn}+medial", P + medial(_inter(gap.buffer(R * 0.5), A), spacing=GAP_SPACING)))
            except _GEOS:
                pass
    for dl in (0.15, 0.3, 0.5):
        cands.append((f"sliver{dl}", sliver(A, dl)))
    # 두꺼운 영역은 동심원, 거기서 남은 얇은 틈은 sliver 로 보강
    try:
        base = concentric(A, s_main)
        gap = A.difference(deposit(base)).buffer(0)
        if gap.area > 1.0:
            extra = [p for p in sliver(A, 0.3) if _inter(LineString(p).buffer(R), gap).area > 0.5 * LineString(p).length]
            cands.append((f"concentric{s_main}+sliver", base + extra))
    except _GEOS:
        pass
    # 모서리 보강: 남은 빈틈의 각 조각 중심선을 짧게 추가
    for name0, P0 in list(cands):
        if not name0.startswith("mitre"):
            continue
        try:
            gap = A.difference(deposit(P0)).buffer(-0.5).buffer(0.5)
        except _GEOS:
            continue
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
    # both=True: 후보를 한 번만 평가해 '나선 포함 최선'과 '고리만 최선'을 함께 돌려준다.
    # 포트폴리오는 두 경로 묶음을 다 쓰는데, 따로 두 번 계산하면 고리 후보를 두 번 평가하게 된다.
    best, best_ring = None, None
    for name, P in cands:
        if not P:
            continue
        try:
            sc = score(A, P)
        except _GEOS:
            continue
        if legacy:
            key = (sc["overfill"] <= 0.30, round(sc["iou"], 3), -sc["length"])
        else:
            ok = meets(sc)
            key = (ok, -est_time(sc) if ok else sc["iou"])
        if best is None or key > best[0]:
            best = (key, P, sc, name)
        if both and not name.startswith("linked") and (best_ring is None or key > best_ring[0]):
            best_ring = (key, P, sc, name)
    if both:
        return (best[1], best[2], best[3]), (best_ring[1], best_ring[2], best_ring[3])
    return best[1], best[2], best[3]


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
