"""V4 스케줄러 = V2(조각 단위 충돌 예약 동시 적층) + 국소 탐색(Local Search)

국소 탐색
- 이동(move): 섹터 경계 근처 경로 조각을, 리치가 닿는 이웃 로봇에게 통째로 넘겨본다
- 각 시도는 그 층만 '시험 계산'(기록 없이 시간만 계산)해서 층 완료시간이 줄면 채택
- 층마다 시간 한도 = LS_FACTOR x (기본 배차 1회 계산시간), 시도 횟수 상한 LS_MAX_TRIES
- 단면이 같은 층은 앞 층에서 찾은 배정을 재사용 (반복 계산 없음)

합격 보장은 V2와 동일: 배차 규칙(점유 영역 충돌 예약)을 바꾸지 않고 '누가 맡을지'만 바꾼다.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

import schedule_v1 as v1
from schedule_v1 import Robot, order_pieces, plen, split
from schedule_v2 import approach_time, chunkify, earliest, hull, retreat_time
from toolpath import best_paths

LS_FACTOR = 3.0
LS_MAX_TRIES = 40
BOUNDARY_DEG = 30.0


class SimRobot(Robot):
    """기록 없이 위치·시각만 추적하는 시험 계산용 로봇."""

    def __init__(self, src: Robot):
        self.rid, self.base, self.home, self.park_xy = src.rid, src.base, src.home, src.park_xy
        self.pos, self.t = src.pos.copy(), src.t
        self.rows = [(0.0, 0.0, 0.0, 0.0, "W")]

    def _row(self, t, p, mode):
        pass

    def idle_until(self, t):
        if t > self.t:
            self.t = t


def build_queues(pieces, owner, robots, chunk, band):
    """pieces: 경로 조각 목록, owner[j]: 조각 j 담당 로봇 -> 로봇별 (pid, ci, chunk) 큐"""
    queues = {i: [] for i in range(len(robots))}
    for o in range(len(robots)):
        pcs = [pieces[j] for j in range(len(pieces)) if owner[j] == o]
        if not pcs:
            continue
        bands = {}
        for pc in pcs:
            r = float(np.max(np.hypot(pc[:, 0], pc[:, 1])))
            bands.setdefault(int(r // band), []).append(pc)
        ordered, cur = [], robots[o].park_xy
        for b in sorted(bands, reverse=True):
            seg = order_pieces(bands[b], cur)
            ordered += seg
            cur = seg[-1][-1]
        for pid, pc in enumerate(ordered):
            for ci, ch in enumerate(chunkify(pc, chunk)):
                queues[o].append((pid, ci, ch))
    return queues


def dispatch_layer(robots, queues, z, lift, arc_on, arc_off, vd, clear, base3):
    """V2 의 층 배차. robots 가 SimRobot 이면 시간만 계산, Robot 이면 행을 기록."""
    zl = z + lift
    booked = []
    n = len(robots)
    arc_open = [False] * n
    last_key = [None] * n
    queues = {i: list(q) for i, q in queues.items()}
    while any(queues[i] for i in range(n)):
        best = None
        for i in range(n):
            if not queues[i]:
                continue
            rb = robots[i]
            pid, ci, ch = queues[i][0]
            pos = rb.pos
            cont = arc_open[i] and last_key[i] == (pid, ci - 1) and np.hypot(*(pos[:2] - ch[0])) < 1e-6
            dep = plen(ch) / vd
            ret_end = retreat_time([ch[-1][0], ch[-1][1], z], rb.park_xy, z)
            if cont:
                H = hull([base3[i], rb.park_xy, pos, *ch])
                s = earliest(rb.t, dep + arc_off + ret_end, H, i, booked, clear)
                if abs(s - rb.t) < 1e-9:
                    cand = (s, i, "cont", H)
                    if best is None or cand[0] < best[0]:
                        best = cand
                    continue
            s_req = rb.t + (arc_off if arc_open[i] else 0.0)
            dur = approach_time(pos, ch[0], z) + arc_on + dep
            H = hull([base3[i], rb.park_xy, pos, *ch])
            s = earliest(s_req, dur + arc_off + ret_end, H, i, booked, clear)
            if abs(s - s_req) < 1e-9:
                cand = (s, i, "direct", H)
            else:
                park3 = [rb.park_xy[0], rb.park_xy[1], zl]
                dur2 = approach_time(park3, ch[0], z) + arc_on + dep
                H2 = hull([base3[i], rb.park_xy, *ch])
                s2 = earliest(s_req + retreat_time(pos, rb.park_xy, z), dur2 + arc_off + ret_end, H2, i, booked, clear)
                cand = (s2, i, "retreat", H2)
            if best is None or cand[0] < best[0]:
                best = cand

        s, i, kind, H = best
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
    return max(rb.t for rb in robots)


def boundary_angle_dist(pc):
    """조각 점들 중 섹터 경계(인접 베이스 방향의 중간각)에 가장 가까운 각도 거리(도)."""
    srt = sorted(v1.DIRS)
    bounds = [srt[i] + ((srt[(i + 1) % 3] - srt[i]) % 360) / 2.0 for i in range(3)]
    ang = np.degrees(np.arctan2(pc[:, 1], pc[:, 0]))
    d = np.min([np.abs((ang - b + 180) % 360 - 180) for b in bounds], axis=0)
    return float(d.min())


def paths_key(paths):
    h = hashlib.sha1()
    for p in paths:
        h.update(np.round(p, 1).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--no-arc", dest="arc", action="store_false")
    ap.add_argument("--chunk", type=float, default=60.0)
    ap.add_argument("--band", type=float, default=60.0)
    ap.add_argument("--no-ls", dest="ls", action="store_false")
    ap.add_argument("--ls-factor", type=float, default=LS_FACTOR)
    a = ap.parse_args()
    job, out, t0 = Path(a.job), Path(a.out), time.time()

    cfg = load_config(job / "config.yaml")
    mesh = load_target_mesh(job / "target.stl", cfg)
    pr = cfg.process
    arc_on = (pr.arc_on_time_s or 0.0) if a.arc else 0.0
    arc_off = (pr.arc_off_time_s or 0.0) if a.arc else 0.0
    vd = pr.deposition_speed_mm_s
    clear = max(r.arm_envelope_radius_mm for r in cfg.robots) * 2 + cfg.collision.arm_clearance_mm + 5.0
    lift = v1.LIFT

    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    layers = slice_target_layers(mesh, list(range(n)), cfg)
    robots = [Robot(r.id, r.base_xyz_mm, r.home_xyz_mm) for r in sorted(cfg.robots, key=lambda r: r.id)]
    reach = [next(r.xy_reach_radius_mm for r in cfg.robots if r.id == rb.rid) for rb in robots]
    v1.DIRS = [v1.ang_of(rb.base) for rb in robots]
    base3 = [np.array([rb.base[0], rb.base[1], 0.0]) for rb in robots]

    clock, min_iou, first = 0.0, 1.0, True
    cache = {}
    st = {"ls_layers": 0, "cache_hits": 0, "tries": 0, "accepted": 0, "gain_s": 0.0, "ls_time": 0.0}
    for k in range(n):
        A = layers[k]
        if A.is_empty:
            continue
        z = pr.build_plane_z_mm + (k + 0.5) * pr.layer_height_mm
        zl = z + lift
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
                rb.travel([rb.pos[0], rb.pos[1], zl])
        clock = max(rb.t for rb in robots)
        for rb in robots:
            rb.idle_until(clock)

        pieces, owner = [], []
        for P in paths:
            for (_, o), pc in split(P, v1.seq_label):
                pieces.append(pc)
                owner.append(o)

        key = paths_key(paths)
        if a.ls and key in cache and len(cache[key]) == len(pieces):
            owner = list(cache[key])
            st["cache_hits"] += 1
        elif a.ls:
            ls_t0 = time.time()

            def trial(own):
                sims = [SimRobot(rb) for rb in robots]
                q = build_queues(pieces, own, sims, a.chunk, a.band)
                return dispatch_layer(sims, q, z, lift, arc_on, arc_off, vd, clear, base3) - clock

            base_t0 = time.time()
            best_T = trial(owner)
            budget = a.ls_factor * (time.time() - base_t0)
            cand = [j for j, pc in enumerate(pieces) if boundary_angle_dist(pc) <= BOUNDARY_DEG]
            cand.sort(key=lambda j: -plen(pieces[j]))
            T0 = best_T
            tries = 0
            improved = True
            while improved and time.time() - ls_t0 < budget and tries < LS_MAX_TRIES:
                improved = False
                for j in cand:
                    if time.time() - ls_t0 >= budget or tries >= LS_MAX_TRIES:
                        break
                    for newo in range(len(robots)):
                        if newo == owner[j]:
                            continue
                        pc = pieces[j]
                        if np.max(np.hypot(pc[:, 0] - robots[newo].base[0], pc[:, 1] - robots[newo].base[1])) > reach[newo] - 1e-6:
                            continue
                        tries += 1
                        own2 = list(owner)
                        own2[j] = newo
                        T = trial(own2)
                        if T < best_T - 1e-6:
                            owner, best_T, improved = own2, T, True
                            st["accepted"] += 1
                            break
                    if improved:
                        break
            st["ls_layers"] += 1
            st["tries"] += tries
            st["gain_s"] += T0 - best_T
            st["ls_time"] += time.time() - ls_t0
            cache[key] = list(owner)

        queues = build_queues(pieces, owner, robots, a.chunk, a.band)
        clock = dispatch_layer(robots, queues, z, lift, arc_on, arc_off, vd, clear, base3)

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
    print(f"{job.name}: layers={n} rows={nrows} makespan={mk:.1f}s ({mk / 3600:.2f}h) self_min_IoU={min_iou:.3f} "
          f"gen={time.time() - t0:.1f}s | LS layers={st['ls_layers']} cache={st['cache_hits']} tries={st['tries']} "
          f"accepted={st['accepted']} est_gain={st['gain_s']:.0f}s ls_time={st['ls_time']:.1f}s")


if __name__ == "__main__":
    main()
