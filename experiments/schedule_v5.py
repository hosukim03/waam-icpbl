"""V5 스케줄러 = V3(schedule_v2.py) + 가운데 작업 시차 배치

V3 문제: 세 로봇 모두 바깥 띠부터 하고 가운데를 마지막에 남긴다.
        가운데는 서로 너무 가까워 한 대씩만 할 수 있어서, 층 끝에 직렬 구간이 생긴다.
V5 개선: 반경 CENTER mm 안쪽 조각을 따로 떼어, 로봇마다 다른 시점(0 / 1/3 / 2/3 지점)에 끼워 넣는다.
        한 로봇이 가운데를 하는 동안 나머지 둘은 바깥에서 일할 수 있다.

--- 아래는 V2/V3 원문 설명 ---
V2 스케줄러: 조각 단위 충돌 예약 기반 동시 적층

- 경로를 섹터 주인 로봇에 배정한 뒤 CHUNK mm 이하 조각으로 분할
- 조각 작업의 '점유 영역' = convex hull(베이스, 주차점, 출발 위치, 조각 점들)
  작업 중 팔(베이스-TCP 선분)은 항상 이 볼록 영역 안에 있음
- 서로 다른 로봇의 점유 영역이 CLEAR mm 이상 떨어져 있지 않으면 시간이 겹치지 않게 예약
- 매 단계 가장 일찍 시작 가능한 (로봇, 조각)을 확정하는 탐욕적 배차
- 같은 경로의 다음 조각을 쉬지 않고 이어가면 아크를 끄지 않음
- 층 사이에는 모든 로봇이 끝날 때까지 기다림(층 순서 보장)
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from shapely import MultiPoint

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

from schedule_v1 import Robot, TRAVEL_V, LIFT, PARK_R, order_pieces, plen, split
import schedule_v1 as v1
from toolpath import best_paths

CHUNK = 120.0
CLEAR_MARGIN = 5.0


def chunkify(pc, max_len):
    """폴리라인을 길이 max_len 이하 조각들로 분할 (점 보간)."""
    seg = np.hypot(*np.diff(pc, axis=0).T)
    total = float(seg.sum())
    if total <= max_len:
        return [pc]
    n = int(math.ceil(total / max_len))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    cuts = [total * i / n for i in range(1, n)]
    out, start_pt, j = [], pc[0], 0
    cur = [pc[0]]
    for c in cuts:
        while cum[j + 1] < c:
            j += 1
            cur.append(pc[j])
        t = (c - cum[j]) / max(seg[j], 1e-12)
        p = pc[j] + (pc[j + 1] - pc[j]) * t
        cur.append(p)
        out.append(np.array(cur))
        cur = [p]
    cur += list(pc[j + 1:])
    out.append(np.array(cur))
    return [o for o in out if len(o) >= 2 and plen(o) > 0.01]


def hull(points):
    return MultiPoint([tuple(map(float, p[:2])) for p in points]).convex_hull


def travel_time(a, b):
    return float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float))) / TRAVEL_V


def approach_time(pos, start_xy, z):
    zl = z + LIFT
    p1 = [pos[0], pos[1], max(pos[2], zl)]
    p2 = [start_xy[0], start_xy[1], zl]
    p3 = [start_xy[0], start_xy[1], z]
    return travel_time(pos, p1) + travel_time(p1, p2) + travel_time(p2, p3)


def retreat_time(pos, park_xy, z):
    zl = z + LIFT
    p1 = [pos[0], pos[1], max(pos[2], zl)]
    return travel_time(pos, p1) + travel_time(p1, [park_xy[0], park_xy[1], zl])


def earliest(s, dur, H, rid, booked, clear):
    conf = [(a, b) for (r, a, b, G) in booked if r != rid and G.distance(H) < clear]
    conf.sort()
    moved = True
    while moved:
        moved = False
        for a, b in conf:
            if a < s + dur - 1e-9 and b > s + 1e-9:
                s = b
                moved = True
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--no-arc", dest="arc", action="store_false")
    ap.add_argument("--chunk", type=float, default=60.0)
    ap.add_argument("--order", choices=["nn", "outer"], default="outer")
    ap.add_argument("--band", type=float, default=60.0)
    ap.add_argument("--center", type=float, default=200.0, help="이 반경 안쪽 = 동시 작업이 어려운 가운데 영역")
    ap.add_argument("--no-stagger", dest="stagger", action="store_false", help="가운데 시차 배치 끄기(=V3)")
    a = ap.parse_args()
    job, out, t0 = Path(a.job), Path(a.out), time.time()

    cfg = load_config(job / "config.yaml")
    mesh = load_target_mesh(job / "target.stl", cfg)
    pr = cfg.process
    arc_on = (pr.arc_on_time_s or 0.0) if a.arc else 0.0
    arc_off = (pr.arc_off_time_s or 0.0) if a.arc else 0.0
    vd = pr.deposition_speed_mm_s
    clear = max(r.arm_envelope_radius_mm for r in cfg.robots) * 2 + cfg.collision.arm_clearance_mm + CLEAR_MARGIN

    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    layers = slice_target_layers(mesh, list(range(n)), cfg)
    robots = [Robot(r.id, r.base_xyz_mm, r.home_xyz_mm) for r in sorted(cfg.robots, key=lambda r: r.id)]
    v1.DIRS = [v1.ang_of(rb.base) for rb in robots]
    base3 = [np.array([rb.base[0], rb.base[1], 0.0]) for rb in robots]

    clock, min_iou, first = 0.0, 1.0, True
    for k in range(n):
        A = layers[k]
        if A.is_empty:
            continue
        z = pr.build_plane_z_mm + (k + 0.5) * pr.layer_height_mm
        zl = z + LIFT
        paths, sc, _ = best_paths(A)
        min_iou = min(min_iou, sc["iou"])

        if first:
            for rb in robots:
                rb.travel([rb.park_xy[0], rb.park_xy[1], zl])
            clock = max(rb.t for rb in robots)
            first = False
        for rb in robots:
            rb.idle_until(clock)
            if abs(rb.pos[2] - zl) > 1e-6:
                rb.travel([rb.pos[0], rb.pos[1], zl])   # 주차점에서 층 높이만 맞춤 (자기 쐐기 안)
        clock = max(rb.t for rb in robots)
        for rb in robots:
            rb.idle_until(clock)

        # 로봇별 조각 큐
        groups = {}
        for P in paths:
            for (_, o), pc in split(P, v1.seq_label):
                groups.setdefault(o, []).append(pc)
        queues = {i: [] for i in range(3)}
        for o, pcs in groups.items():
            if a.order == "outer":
                # 바깥 반경 띠부터: 세 로봇이 바깥에서 동시에 일하고 가운데는 나중에
                center = [pc for pc in pcs if float(np.max(np.hypot(pc[:, 0], pc[:, 1]))) < a.center]
                outer = [pc for pc in pcs if float(np.max(np.hypot(pc[:, 0], pc[:, 1]))) >= a.center]
                bands = {}
                for pc in outer:
                    r = float(np.max(np.hypot(pc[:, 0], pc[:, 1])))
                    bands.setdefault(int(r // a.band), []).append(pc)
                ordered, cur = [], robots[o].park_xy
                for b in sorted(bands, reverse=True):
                    seg = order_pieces(bands[b], cur)
                    ordered += seg
                    cur = seg[-1][-1]
                if center:
                    # 가운데 묶음을 로봇마다 다른 지점에 끼워 넣어 한 대씩 번갈아 쓰게 한다
                    at = int(len(ordered) * o / 3) if a.stagger else len(ordered)
                    ref = ordered[at - 1][-1] if at > 0 else robots[o].park_xy
                    ordered = ordered[:at] + order_pieces(center, ref) + ordered[at:]
            else:
                ordered = order_pieces(pcs, robots[o].park_xy)
            for pid, pc in enumerate(ordered):
                for ci, ch in enumerate(chunkify(pc, a.chunk)):
                    queues[o].append((pid, ci, ch))

        booked = []            # (robot index, start, end, hull)
        arc_open = [False] * 3
        last_key = [None] * 3
        while any(queues[i] for i in range(3)):
            best = None
            for i in range(3):
                if not queues[i]:
                    continue
                rb = robots[i]
                pid, ci, ch = queues[i][0]
                pos = rb.pos
                cont = arc_open[i] and last_key[i] == (pid, ci - 1) and np.hypot(*(pos[:2] - ch[0])) < 1e-6
                dep = plen(ch) / vd
                ret_end = retreat_time([ch[-1][0], ch[-1][1], z], rb.park_xy, z)
                if cont:
                    s_req, dur = rb.t, dep
                    H = hull([base3[i], rb.park_xy, pos, *ch])
                    s = earliest(s_req, dur + arc_off + ret_end, H, i, booked, clear)
                    if abs(s - s_req) < 1e-9:
                        cand = (s, i, "cont", s_req, H, dur)
                        if best is None or cand[0] < best[0]:
                            best = cand
                        continue
                off = arc_off if arc_open[i] else 0.0
                s_req = rb.t + off
                dur = approach_time(pos, ch[0], z) + arc_on + dep
                H = hull([base3[i], rb.park_xy, pos, *ch])
                s = earliest(s_req, dur + arc_off + ret_end, H, i, booked, clear)
                if abs(s - s_req) < 1e-9:
                    cand = (s, i, "direct", s_req, H, dur)
                else:
                    ret = retreat_time(pos, rb.park_xy, z)
                    park3 = [rb.park_xy[0], rb.park_xy[1], zl]
                    dur2 = approach_time(park3, ch[0], z) + arc_on + dep
                    H2 = hull([base3[i], rb.park_xy, *ch])
                    s2 = earliest(s_req + ret, dur2 + arc_off + ret_end, H2, i, booked, clear)
                    cand = (s2, i, "retreat", s_req, H2, dur2)
                if best is None or cand[0] < best[0]:
                    best = cand

            s, i, kind, s_req, H, dur = best
            rb = robots[i]
            pid, ci, ch = queues[i].pop(0)
            if kind != "cont":
                if arc_open[i]:
                    rb.wait(arc_off)
                    arc_open[i] = False
                if kind == "retreat":
                    rb.travel([rb.pos[0], rb.pos[1], zl])
                    rb.travel([rb.park_xy[0], rb.park_xy[1], zl])
                rb.idle_until(s)
                rb.travel([rb.pos[0], rb.pos[1], zl])
                rb.travel([ch[0][0], ch[0][1], zl])
                rb.travel([ch[0][0], ch[0][1], z])
                rb.wait(arc_on)
            rb.deposit(ch, z, vd)
            arc_open[i] = True
            last_key[i] = (pid, ci)
            ret_end = retreat_time([ch[-1][0], ch[-1][1], z], rb.park_xy, z)
            booked.append((i, s, rb.t + arc_off + ret_end, H))

        for i, rb in enumerate(robots):
            if arc_open[i]:
                rb.wait(arc_off)
            rb.travel([rb.pos[0], rb.pos[1], zl])
            rb.travel([rb.park_xy[0], rb.park_xy[1], zl])
        clock = max(rb.t for rb in robots)

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
    dep_total = sum(1 for _ in ())
    print(f"{job.name}: layers={n} rows={nrows} makespan={mk:.1f}s ({mk / 3600:.2f}h) "
          f"self_min_IoU={min_iou:.3f} gen={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
