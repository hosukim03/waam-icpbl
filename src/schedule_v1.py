"""V1 스케줄러

층마다 두 계획의 예상 시간을 계산해 빠른 쪽을 채택한다.
  SEQ : 한 번에 한 로봇만 이동 (섹터 주인에게 배정)
  ZONE: 로봇별 안전 쐐기 영역 안의 조각은 세 로봇 동시 적층 -> 틈(띠) 위 조각은 순차 적층
안전 근거: 쐐기 i = 섹터 i 에서 세 경계 반직선으로부터 거리 >= h 인 영역(볼록).
          베이스·주차점·이동경로가 모두 쐐기 i 안에 있으므로 동시 이동 중 팔 간 거리 >= 2h.
주차: 베이스 방향 반지름 PARK_R, 현재 층 위 LIFT 높이 (홈 1000mm 로 오르내리지 않음)
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

from toolpath import best_paths

# 이동 속도는 config 의 travel_speed_mm_s(150) 규격값. 들어올림은 층 높이 2mm 를 넘기면 되므로 3mm.
TRAVEL_V, LIFT, PARK_R = 150.0, 3.0, 900.0
T_PAD_S, T_PAD_BELOW_MM = 2e-6, 1.0     # 짧은 이동의 시각 반올림 여유(travel 참고)
H_STRIP = 135.0          # 띠 절반폭 -> 동시 작업 시 팔 간 최소 270mm
MIN_ZONE_PIECE = 40.0    # 이보다 짧은 영역 조각은 아크 비용 때문에 순차로 돌림

DIRS: list[float] = []
RAYS: list[np.ndarray] = []


def ang_of(p):
    return math.degrees(math.atan2(p[1], p[0]))


def sector_owner(p):
    a = ang_of(p)
    return int(np.argmin([abs((a - d + 180) % 360 - 180) for d in DIRS]))


def in_zone(p, i):
    if sector_owner(p) != i:
        return False
    for u in RAYS:
        t = max(0.0, p[0] * u[0] + p[1] * u[1])
        if math.hypot(p[0] - u[0] * t, p[1] - u[1] * t) < H_STRIP:
            return False
    return True


def label(p):
    """('Z', i) = 로봇 i 안전영역, ('S', i) = 띠 위(섹터 주인 i)"""
    o = sector_owner(p)
    return ("Z", o) if in_zone(p, o) else ("S", o)


def seq_label(p):
    return ("S", sector_owner(p))


def split(poly, labeler):
    """라벨이 바뀌는 곳에서 폴리라인을 끊는다 (경계점은 0.25mm 이내로 삽입)."""
    pts = [np.asarray(poly[0], float)]
    for p in poly[1:]:
        p = np.asarray(p, float)
        q = pts[-1]
        if labeler(q) != labeler(p):
            L = float(np.hypot(*(p - q)))
            n = max(2, int(math.ceil(L / 0.5)))
            prev = q
            for i in range(1, n + 1):
                cur = q + (p - q) * (i / n)
                if labeler(cur) != labeler(prev):
                    pts.append((prev + cur) / 2.0)
                prev = cur
        pts.append(p)
    if len(pts) < 2:
        return []
    pieces, cur, lab = [], [pts[0]], labeler((pts[0] + pts[1]) / 2.0)
    for i in range(1, len(pts)):
        lb = labeler((pts[i - 1] + pts[i]) / 2.0)
        if lb != lab:
            if len(cur) >= 2:
                pieces.append([lab, np.array(cur)])
            cur, lab = [pts[i - 1]], lb
        cur.append(pts[i])
    if len(cur) >= 2:
        pieces.append([lab, np.array(cur)])
    return pieces


def plen(pc):
    return float(np.sum(np.hypot(*np.diff(pc, axis=0).T))) if len(pc) > 1 else 0.0


class Robot:
    def __init__(self, rid, base, home):
        self.rid = rid
        self.base = np.array(base[:2], float)
        self.home = np.array(home, float)
        self.park_xy = self.base / np.linalg.norm(self.base) * PARK_R
        self.rows = [(0.0, *self.home, "W")]
        self.pos, self.t = self.home.copy(), 0.0

    def _row(self, t, p, mode):
        last = self.rows[-1]
        if t <= last[0] + 1e-9:
            self.rows[-1] = (last[0], last[1], last[2], last[3], mode)
            return
        self.rows.append((t, float(p[0]), float(p[1]), float(p[2]), mode))

    def idle_until(self, t):
        if t > self.t + 1e-9:
            last = self.rows[-1]
            self.rows[-1] = (last[0], last[1], last[2], last[3], "W")
            self._row(t, self.pos, "W")
            self.t = t

    def travel(self, q):
        q = np.asarray(q, float)
        d = float(np.linalg.norm(q - self.pos))
        if d < 0.01:
            return
        self._row(self.t, self.pos, "T")
        # CSV 에는 시각을 소수 6자리(1µs)로 적는다. 0.01mm 남짓한 아주 짧은 이동은 반올림으로 시간이 1µs 짧아져도
        # 채점기가 계산한 속도가 한계(기준의 +10%)를 넘는다(공개 03: 0.011mm 이동이 165.07 > 165 mm/s 로 FAIL).
        # 짧은 이동에만 2µs 를 더해 반올림 뒤에도 실제 이동 속도 이하로 기록되게 한다(1mm 이상은 오차 0.02% 이하).
        self.t += d / TRAVEL_V + (T_PAD_S if d < T_PAD_BELOW_MM else 0.0)
        self.pos = q
        self._row(self.t, self.pos, "T")

    def wait(self, dt):
        if dt <= 0:
            return
        self._row(self.t, self.pos, "W")
        self.t += dt
        self._row(self.t, self.pos, "W")

    def deposit(self, pts, z, v):
        for p in pts[1:]:
            q = np.array([p[0], p[1], z])
            d = float(np.linalg.norm(q[:2] - self.pos[:2]))
            if d < 0.01:
                continue
            self._row(self.t, self.pos, "D")
            self.t += d / v
            self.pos = q
            self._row(self.t, self.pos, "D")


def _tour_cost(seq, start_xy):
    cur, c = np.asarray(start_xy, float)[:2], 0.0
    for pc in seq:
        c += float(np.hypot(*(pc[0] - cur)))
        cur = pc[-1]
    return c


def refine_order(seq, start_xy, rounds: int = 6):
    """최근접 이웃으로 만든 순서를 2-opt 로 다듬는다(조각 사이 이동 거리 합을 줄인다).

    구간을 뒤집을 때 그 안의 조각들도 방향을 뒤집어야 이어진다. 닫힌 고리는 시작점을 다시 고른다.
    """
    seq = [np.asarray(p) for p in seq]
    n = len(seq)
    if n < 3:
        return seq
    best = _tour_cost(seq, start_xy)
    for _ in range(rounds):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                trial = seq[:i] + [p[::-1] for p in reversed(seq[i:j + 1])] + seq[j + 1:]
                c = _tour_cost(trial, start_xy)
                if c < best - 1e-9:
                    seq, best, improved = trial, c, True
        if not improved:
            break
    # 닫힌 고리는 앞 조각 끝에서 가장 가까운 점부터 시작하도록 다시 회전
    cur = np.asarray(start_xy, float)[:2]
    out = []
    for pc in seq:
        if len(pc) > 3 and np.hypot(*(pc[0] - pc[-1])) < 1e-6:
            k = int(np.argmin(np.hypot(pc[:, 0] - cur[0], pc[:, 1] - cur[1])))
            pc = np.vstack([pc[k:-1], pc[: k + 1]])
        out.append(pc)
        cur = pc[-1]
    return out


def order_pieces(pieces, start_xy):
    """최근접 이웃. 닫힌 고리는 현재 위치에서 가장 가까운 점부터 시작하도록 회전."""
    todo = [np.asarray(p) for p in pieces]
    cur, out = np.asarray(start_xy, float)[:2], []
    while todo:
        best = None
        for i, pc in enumerate(todo):
            closed = len(pc) > 3 and np.hypot(*(pc[0] - pc[-1])) < 1e-6
            if closed:
                j = int(np.argmin(np.hypot(pc[:, 0] - cur[0], pc[:, 1] - cur[1])))
                d = float(np.hypot(*(pc[j] - cur)))
                cand = np.vstack([pc[j:-1], pc[: j + 1]])
                if best is None or d < best[0]:
                    best = (d, i, cand)
            else:
                for cand in (pc, pc[::-1]):
                    d = float(np.hypot(*(cand[0] - cur)))
                    if best is None or d < best[0]:
                        best = (d, i, cand)
        _, i, cand = best
        todo.pop(i)
        out.append(cand)
        cur = cand[-1]
    return out


def run_cost(start, pieces, z, park, arc, v_dep):
    """start(xyz)에서 조각들을 순서대로 적층하고 park(xyz)로 돌아오기까지의 시간."""
    zl, pos, t = z + LIFT, np.asarray(start, float), 0.0

    def go(q):
        nonlocal pos, t
        q = np.asarray(q, float)
        t += float(np.linalg.norm(q - pos)) / TRAVEL_V
        pos = q

    for pc in pieces:
        go([pos[0], pos[1], max(pos[2], zl)])
        go([pc[0][0], pc[0][1], zl])
        go([pc[0][0], pc[0][1], z])
        t += arc + plen(pc) / v_dep
        pos = np.array([pc[-1][0], pc[-1][1], z])
    go([pos[0], pos[1], zl])
    go([park[0], park[1], zl])
    return t


def do_run(rb, pieces, z, arc_on, arc_off, v_dep):
    zl = z + LIFT
    for pc in pieces:
        rb.travel([rb.pos[0], rb.pos[1], max(rb.pos[2], zl)])
        rb.travel([pc[0][0], pc[0][1], zl])
        rb.travel([pc[0][0], pc[0][1], z])
        rb.wait(arc_on)
        rb.deposit(pc, z, v_dep)
        rb.wait(arc_off)
    rb.travel([rb.pos[0], rb.pos[1], zl])
    rb.travel([rb.park_xy[0], rb.park_xy[1], zl])


def main():
    global DIRS, RAYS
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--no-arc", dest="arc", action="store_false")
    ap.add_argument("--force", choices=["SEQ", "ZONE"], default=None)
    a = ap.parse_args()
    job, out, t0 = Path(a.job), Path(a.out), time.time()

    cfg = load_config(job / "config.yaml")
    mesh = load_target_mesh(job / "target.stl", cfg)
    pr = cfg.process
    arc_on = (pr.arc_on_time_s or 0.0) if a.arc else 0.0
    arc_off = (pr.arc_off_time_s or 0.0) if a.arc else 0.0
    arc, vd = arc_on + arc_off, pr.deposition_speed_mm_s

    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    layers = slice_target_layers(mesh, list(range(n)), cfg)
    robots = [Robot(r.id, r.base_xyz_mm, r.home_xyz_mm) for r in sorted(cfg.robots, key=lambda r: r.id)]

    DIRS = [ang_of(rb.base) for rb in robots]
    srt = sorted(DIRS)
    bis = [srt[i] + ((srt[(i + 1) % 3] - srt[i]) % 360) / 2.0 for i in range(3)]
    RAYS = [np.array([math.cos(math.radians(b)), math.sin(math.radians(b))]) for b in bis]
    need = max(r.arm_envelope_radius_mm for r in cfg.robots) * 2 + cfg.collision.arm_clearance_mm
    assert 2 * H_STRIP > need, "strip too narrow for guaranteed arm clearance"

    clock, stats, min_iou, first = 0.0, {"SEQ": 0, "ZONE": 0}, 1.0, True
    for k in range(n):
        A = layers[k]
        if A.is_empty:
            continue
        z = pr.build_plane_z_mm + (k + 0.5) * pr.layer_height_mm
        zl = z + LIFT
        paths, sc, _ = best_paths(A)
        min_iou = min(min_iou, sc["iou"])

        if first:   # 홈 -> 첫 주차점 (각자 쐐기 안 이동이라 동시에 해도 안전)
            for rb in robots:
                rb.travel([rb.park_xy[0], rb.park_xy[1], zl])
            clock = max(rb.t for rb in robots)
            for rb in robots:
                rb.idle_until(clock)
            first = False

        park = [np.array([rb.park_xy[0], rb.park_xy[1], zl]) for rb in robots]
        start = [rb.pos.copy() for rb in robots]

        # SEQ 계획
        seq_groups = {}
        for P in paths:
            for (_, o), pc in split(P, seq_label):
                seq_groups.setdefault(o, []).append(pc)
        seq_order = {o: order_pieces(v, start[o]) for o, v in seq_groups.items()}
        seq_cost = sum(run_cost(start[o], seq_order[o], z, park[o], arc, vd) for o in seq_order)

        # ZONE 계획
        zone, strip = {}, {}
        for P in paths:
            ps = split(P, label)
            for item in ps:
                if item[0][0] == "Z" and plen(item[1]) < MIN_ZONE_PIECE:
                    item[0] = ("S", item[0][1])
            for (kind, o), pc in ps:
                (zone if kind == "Z" else strip).setdefault(o, []).append(pc)
        zone_order = {o: order_pieces(v, start[o]) for o, v in zone.items()}
        par_cost = max([run_cost(start[o], zone_order[o], z, park[o], arc, vd) for o in zone_order] or [0.0])
        strip_order = {o: order_pieces(v, park[o]) for o, v in strip.items()}
        zone_cost = par_cost + sum(run_cost(park[o], strip_order[o], z, park[o], arc, vd) for o in strip_order)

        mode = a.force or ("ZONE" if zone and zone_cost < seq_cost else "SEQ")
        stats[mode] += 1

        if mode == "SEQ":
            for o in sorted(seq_order):
                rb = robots[o]
                rb.idle_until(clock)
                do_run(rb, seq_order[o], z, arc_on, arc_off, vd)
                clock = rb.t
        else:
            for o, pcs in zone_order.items():
                rb = robots[o]
                rb.idle_until(clock)
                do_run(rb, pcs, z, arc_on, arc_off, vd)
            clock = max([clock] + [robots[o].t for o in zone_order])
            for o in sorted(strip_order):
                rb = robots[o]
                rb.idle_until(clock)
                do_run(rb, strip_order[o], z, arc_on, arc_off, vd)
                clock = rb.t

        # 이번 층에 일 안 한 로봇도 이 층 높이 주차점으로 정렬 (순차)
        for rb in robots:
            if abs(rb.pos[2] - zl) > 1e-6 or np.hypot(*(rb.pos[:2] - rb.park_xy)) > 1e-6:
                rb.idle_until(clock)
                rb.travel([rb.park_xy[0], rb.park_xy[1], zl])
                clock = rb.t

    for rb in robots:
        last = rb.rows[-1]
        rb.rows[-1] = (last[0], last[1], last[2], last[3], "W")
        if len(rb.rows) < 2:
            rb.rows.append((last[0] + 1.0, last[1], last[2], last[3], "W"))

    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(job / "config.yaml", out / "config.yaml")
    shutil.copyfile(job / "target.stl", out / "target.stl")
    nrows = 0
    with open(out / "trajectory.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("robot_id,time_s,x_mm,y_mm,z_mm,mode\n")
        for rb in robots:
            prev = -1.0
            for t, x, y, zz, m in rb.rows:
                tr = round(t, 6)
                if tr <= prev:
                    tr = round(prev + 1e-6, 6)
                prev = tr
                f.write(f"{rb.rid},{tr:.6f},{x:.6f},{y:.6f},{zz:.6f},{m}\n")
                nrows += 1
    mk = max(rb.rows[-1][0] for rb in robots)
    print(f"{job.name}: layers={n} rows={nrows} makespan={mk:.1f}s ({mk / 3600:.2f}h) "
          f"layer_modes={stats} self_min_IoU={min_iou:.3f} gen={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
