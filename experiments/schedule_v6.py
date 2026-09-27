"""V6 스케줄러 = V3(schedule_v2.py) + 층 파이프라이닝

V3 문제: 한 층이 완전히 끝나야 다음 층을 시작한다(층 장벽).
        층 끝에서 한 로봇이 가운데를 마무리하는 동안 나머지 둘이 논다.
        04번 기준 이 대기만 로봇시간 8.0h (makespan 환산 약 2.7h).
V6 개선: 층 k 가 다 끝나지 않아도, 다음 층 조각 중 '자기 아래가 이미 쌓인' 조각은 먼저 시작한다.
        - 조각 c(층 k+1)의 선행 조건 = 층 k 조각 중 c 와 겹치는(비드 폭 버퍼 기준) 조각들의 완료
        - 선행 조각이 모두 끝난 뒤에만 시작 → 층 순서(아래가 먼저)는 지켜진다
        - 충돌 예약(볼록 영역 거리)은 층과 무관하게 그대로 적용
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

from schedule_v1 import Robot, LIFT, order_pieces, plen, split
import schedule_v1 as v1
from schedule_v2 import approach_time, chunkify, earliest, hull, retreat_time
from toolpath import best_paths


class Task:
    __slots__ = ("tid", "rid", "layer", "pid", "ci", "pts", "geom", "z", "deps")

    def __init__(self, tid, rid, layer, pid, ci, pts, z, bead):
        self.tid, self.rid, self.layer = tid, rid, layer
        self.pid, self.ci, self.pts, self.z = pid, ci, pts, z
        self.geom = LineString([(float(p[0]), float(p[1])) for p in pts]).buffer(bead)
        self.deps = ()


class Layer:
    def __init__(self, k, z, queues):
        self.k, self.z, self.q = k, z, queues


def build_layer(k, A, z, robots, args, tid0, bead):
    """층 하나의 로봇별 조각 큐를 만든다 (V3 와 같은 순서 규칙: 바깥 띠부터)."""
    paths, sc, _ = best_paths(A)
    groups = {}
    for P in paths:
        for (_, o), pc in split(P, v1.seq_label):
            groups.setdefault(o, []).append(pc)
    queues = {i: [] for i in range(3)}
    tid = tid0
    for o, pcs in groups.items():
        if args.order == "outer":
            bands = {}
            for pc in pcs:
                r = float(np.max(np.hypot(pc[:, 0], pc[:, 1])))
                bands.setdefault(int(r // args.band), []).append(pc)
            ordered, cur = [], robots[o].park_xy
            for b in sorted(bands, reverse=True):
                seg = order_pieces(bands[b], cur)
                ordered += seg
                cur = seg[-1][-1]
        else:
            ordered = order_pieces(pcs, robots[o].park_xy)
        for pid, pc in enumerate(ordered):
            for ci, ch in enumerate(chunkify(pc, args.chunk)):
                queues[o].append(Task(tid, o, k, pid, ci, ch, z, bead))
                tid += 1
    return Layer(k, z, queues), tid, sc["iou"]


def link_deps(prev: Layer, nxt: Layer):
    """다음 층 조각마다 '바로 아래에서 겹치는' 이전 층 조각들을 선행 조건으로 연결."""
    below = [t for i in range(3) for t in prev.q[i]]
    if not below:
        return
    tree = STRtree([t.geom for t in below])
    for i in range(3):
        for t in nxt.q[i]:
            idx = tree.query(t.geom, predicate="intersects")
            t.deps = tuple(below[j].tid for j in idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--no-arc", dest="arc", action="store_false")
    ap.add_argument("--chunk", type=float, default=60.0)
    ap.add_argument("--order", choices=["nn", "outer"], default="outer")
    ap.add_argument("--band", type=float, default=60.0)
    ap.add_argument("--no-pipeline", dest="pipeline", action="store_false",
                    help="층 파이프라이닝 끄기(=V3 동작)")
    ap.add_argument("--lookahead", type=int, default=6, help="막힌 조각을 건너뛰고 살펴볼 개수")
    a = ap.parse_args()
    job, out, t0 = Path(a.job), Path(a.out), time.time()

    cfg = load_config(job / "config.yaml")
    mesh = load_target_mesh(job / "target.stl", cfg)
    pr = cfg.process
    arc_on = (pr.arc_on_time_s or 0.0) if a.arc else 0.0
    arc_off = (pr.arc_off_time_s or 0.0) if a.arc else 0.0
    vd = pr.deposition_speed_mm_s
    bead = pr.bead_width_mm / 2
    clear = max(r.arm_envelope_radius_mm for r in cfg.robots) * 2 + cfg.collision.arm_clearance_mm + 5.0

    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    layers_geo = slice_target_layers(mesh, list(range(n)), cfg)
    robots = [Robot(r.id, r.base_xyz_mm, r.home_xyz_mm) for r in sorted(cfg.robots, key=lambda r: r.id)]
    v1.DIRS = [v1.ang_of(rb.base) for rb in robots]
    base3 = [np.array([rb.base[0], rb.base[1], 0.0]) for rb in robots]

    def z_of(k):
        return pr.build_plane_z_mm + (k + 0.5) * pr.layer_height_mm

    done = {}                      # tid -> 완료 시각
    booked = []                    # (robot, start, end, hull)
    arc_open = [False] * 3
    last_key = [None] * 3
    min_iou, tid = 1.0, 0

    # 층 준비 (빈 층 건너뜀)
    ks = [k for k in range(n) if not layers_geo[k].is_empty]
    if not ks:
        return
    for rb in robots:
        rb.travel([rb.park_xy[0], rb.park_xy[1], z_of(ks[0]) + LIFT])
    clock = max(rb.t for rb in robots)
    for rb in robots:
        rb.idle_until(clock)

    cur, tid, iou = build_layer(ks[0], layers_geo[ks[0]], z_of(ks[0]), robots, a, tid, bead)
    min_iou = min(min_iou, iou)
    nxt = None
    pos_in_list = 0

    while cur is not None:
        if a.pipeline and nxt is None and pos_in_list + 1 < len(ks):
            k2 = ks[pos_in_list + 1]
            nxt, tid, iou = build_layer(k2, layers_geo[k2], z_of(k2), robots, a, tid, bead)
            min_iou = min(min_iou, iou)
            link_deps(cur, nxt)

        def candidates(i):
            """로봇 i 가 지금 손댈 수 있는 조각 후보 (현재 층 우선, 없으면 다음 층)."""
            out = []
            for L in (cur, nxt):
                if L is None:
                    continue
                q = L.q[i]
                for j in range(min(a.lookahead, len(q))):
                    t = q[j]
                    if all(d in done for d in t.deps):
                        out.append((L, j, t))
                        break
                if out:
                    break
            return out

        best = None
        for i in range(3):
            for L, j, t in candidates(i):
                rb = robots[i]
                z, zl = t.z, t.z + LIFT
                pos = rb.pos
                ch = t.pts
                cont = (arc_open[i] and last_key[i] == (t.layer, t.pid, t.ci - 1)
                        and abs(pos[2] - z) < 1e-6 and np.hypot(*(pos[:2] - ch[0])) < 1e-6)
                dep = plen(ch) / vd
                ready = max(rb.t, *(done[d] for d in t.deps)) if t.deps else rb.t
                ret_end = retreat_time([ch[-1][0], ch[-1][1], z], rb.park_xy, z)
                if cont and ready <= rb.t + 1e-9:
                    H = hull([base3[i], rb.park_xy, pos, *ch])
                    s = earliest(rb.t, dep + arc_off + ret_end, H, i, booked, clear)
                    if abs(s - rb.t) < 1e-9:
                        cand = (s, i, "cont", H, dep, L, j, t)
                        if best is None or cand[0] < best[0]:
                            best = cand
                        continue
                off = arc_off if arc_open[i] else 0.0
                s_req = max(rb.t + off, ready)
                dur = approach_time(pos, ch[0], z) + arc_on + dep
                H = hull([base3[i], rb.park_xy, pos, *ch])
                s = earliest(s_req, dur + arc_off + ret_end, H, i, booked, clear)
                if abs(s - s_req) < 1e-9:
                    cand = (s, i, "direct", H, dur, L, j, t)
                else:
                    ret = retreat_time(pos, rb.park_xy, z)
                    park3 = [rb.park_xy[0], rb.park_xy[1], zl]
                    dur2 = approach_time(park3, ch[0], z) + arc_on + dep
                    H2 = hull([base3[i], rb.park_xy, *ch])
                    s2 = earliest(s_req + ret, dur2 + arc_off + ret_end, H2, i, booked, clear)
                    cand = (s2, i, "retreat", H2, dur2, L, j, t)
                if best is None or cand[0] < best[0]:
                    best = cand

        if best is None:        # 모두 선행 대기 중 — 가장 이른 완료까지 시계를 민다
            pend = [done[d] for L in (cur, nxt) if L for i in range(3) for t in L.q[i][:1] for d in t.deps if d in done]
            step = max(pend) if pend else max(rb.t for rb in robots) + 1.0
            for rb in robots:
                rb.idle_until(max(rb.t, step))
            continue

        s, i, kind, H, dur, L, j, t = best
        rb = robots[i]
        ch = t.pts
        z, zl = t.z, t.z + LIFT
        del L.q[i][j]
        if kind != "cont":
            if arc_open[i]:
                rb.wait(arc_off)
                arc_open[i] = False
            if kind == "retreat":
                rb.travel([rb.pos[0], rb.pos[1], max(rb.pos[2], zl)])
                rb.travel([rb.park_xy[0], rb.park_xy[1], zl])
            rb.idle_until(s)
            rb.travel([rb.pos[0], rb.pos[1], max(rb.pos[2], zl)])
            rb.travel([ch[0][0], ch[0][1], zl])
            rb.travel([ch[0][0], ch[0][1], z])
            rb.wait(arc_on)
        rb.deposit(ch, z, vd)
        arc_open[i] = True
        last_key[i] = (t.layer, t.pid, t.ci)
        done[t.tid] = rb.t
        ret_end = retreat_time([ch[-1][0], ch[-1][1], z], rb.park_xy, z)
        booked.append((i, s, rb.t + arc_off + ret_end, H))
        cut = min(rb.t for rb in robots) - 1.0
        if len(booked) > 400:
            booked = [b for b in booked if b[2] > cut]

        if all(not cur.q[i] for i in range(3)):      # 현재 층 배차 완료 → 창을 민다
            cur, nxt = nxt, None
            pos_in_list += 1
            if cur is None and pos_in_list < len(ks):
                k2 = ks[pos_in_list]
                cur, tid, iou = build_layer(k2, layers_geo[k2], z_of(k2), robots, a, tid, bead)
                min_iou = min(min_iou, iou)

    for i, rb in enumerate(robots):
        if arc_open[i]:
            rb.wait(arc_off)
        zl = rb.pos[2] + LIFT
        rb.travel([rb.pos[0], rb.pos[1], zl])
        rb.travel([rb.park_xy[0], rb.park_xy[1], zl])

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
            for t_, x, y, zz, m in rb.rows:
                tr = round(t_, 6)
                if tr <= prev:
                    tr = round(prev + 1e-6, 6)
                prev = tr
                f.write(f"{rb.rid},{tr:.6f},{x:.6f},{y:.6f},{zz:.6f},{m}\n")
                nrows += 1
    mk = max(rb.rows[-1][0] for rb in robots)
    print(f"{job.name}: layers={n} rows={nrows} makespan={mk:.1f}s ({mk / 3600:.2f}h) "
          f"self_min_IoU={min_iou:.3f} gen={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
