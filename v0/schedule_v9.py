# -*- coding: utf-8 -*-
"""V9 — 층 장벽을 없앤 V8.

V8 은 세 로봇이 모두 층 k 를 끝내야 층 k+1 로 넘어간다. 그래서 먼저 끝난 로봇은 가장 늦은
로봇을 기다린다(공개 04 기준 로봇당 makespan 의 12.8%, 01 은 16.6%). V9 는 로봇의 큐가 비면
다음 층을 열어 그 로봇 큐 뒤에 붙이므로 세 로봇이 서로 다른 층에 있어도 된다.

물리적으로 맞게 하려고 '지지 조건'을 둔다: 층 k+1 의 조각은 그 발자국과 겹치는 층 k 조각들이
전부 끝났을 때만 시작할 수 있다(빈 공중에 쌓지 않음). 채점기는 층 순서를 검사하지 않지만
실제 공정에서는 필수다. 큐는 층 순서로 쌓이므로 이 조건 때문에 교착이 생기지는 않는다
(층 k+1 이 막혀 있으면 그 밑의 층 k 조각을 누군가 큐 앞쪽에 들고 있다).

경로 생성·충돌 검사·주차·비켜서기는 V8 것을 그대로 쓴다. 옵션도 V8 과 같다(--rollout, --policy 제외).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from shapely import LineString, STRtree

import schedule_v8 as v8
from schedule_v8 import (Checker, plan, seg_dist, LIFT, TRAVEL_V, MAX_TRIES, RETRACT_R, REACH_OK,
                         SectorLabeler, order_pieces, split, chunkify, best_paths, load_config,
                         load_target_mesh, slice_target_layers, Robot, v1)

AHEAD = 1            # 가장 늦은 로봇보다 최대 몇 층 앞서 열어 둘지
SUPPORT_R = 4.5      # 발자국 겹침 판정 반경(mm) — 비드 반폭 4 + 여유


def build_queue(k, paths, z, zl, robots, P, starts=None):
    """층 k 의 조각을 로봇별로 나눠 (k, pid, ci, ch, z, zl) 목록으로 만든다. V8 _place_layer 와 같은 규칙."""
    opts, nrob, band_dir = P["opts"], P["nrob"], P["band_dir"]
    order2, link_gap = P["order2"], P["link_gap"]
    labeler = None
    if nrob < 3:
        Pxy = np.vstack([np.asarray(Q, float)[:, :2] for Q in paths])
        if nrob == 1:
            far = [float(np.max(np.hypot(*(Pxy - rb.base).T))) for rb in robots]
            j = int(np.argmin(far))
            if far[j] <= REACH_OK:
                labeler = lambda p, _j=j: ("S", _j)
        else:
            lb = SectorLabeler(paths, [rb.base for rb in robots[:nrob]])
            if getattr(lb, "unreach", 1.0) <= 0.0:
                labeler = lb
    if labeler is None:
        labeler = (SectorLabeler(paths, [rb.base for rb in robots])
                   if getattr(opts, "balance", True) else v1.seq_label)
    groups = {}
    for Pl in paths:
        for (_, o), pc in split(Pl, labeler):
            groups.setdefault(o, []).append(pc)
    out = {i: [] for i in range(3)}
    for o, pcs in groups.items():
        bands = {}
        for pc in pcs:
            r = float(np.max(np.hypot(pc[:, 0], pc[:, 1])))
            bands.setdefault(int(r // opts.band), []).append(pc)
        ordered, cur = [], (starts[o] if starts and starts.get(o) is not None else robots[o].pos[:2])
        outward = {"out": True, "in": False}.get(band_dir, k % 2 == 0)
        for b in sorted(bands, reverse=outward):
            seg = order_pieces(bands[b], cur)
            if order2:
                seg = v1.refine_order(seg, cur)
            ordered += seg
            cur = seg[-1][-1]
        if link_gap > 0 and len(ordered) > 1:
            merged = [ordered[0]]
            for pc in ordered[1:]:
                if float(np.hypot(*(pc[0] - merged[-1][-1]))) <= link_gap:
                    merged[-1] = np.vstack([merged[-1], pc])   # 첫 점을 버리면 긴 직선이 통째로 비껴간다(새형상06)
                else:
                    merged.append(pc)
            ordered = merged
        for pid, pc in enumerate(ordered):
            for ci, ch in enumerate(chunkify(pc, opts.chunk)):
                out[o].append((k, pid, ci, ch, z, zl))
    return out


class Support:
    """층별 조각 발자국과 완료 여부. 층 k+1 조각이 시작 가능한지(밑의 층 k 조각이 다 끝났는지) 답한다."""

    def __init__(self):
        self.tree = {}      # k -> STRtree of chunk footprints
        self.done = {}      # k -> list[bool]
        self.geom = {}      # k -> list of buffered footprints

    def add_layer(self, k, chunks):
        g = [LineString(np.asarray(ch, float)[:, :2]).buffer(SUPPORT_R) for ch in chunks]
        self.geom[k] = g
        self.done[k] = [False] * len(g)
        self.tree[k] = STRtree(g) if g else None

    def finish(self, k, idx):
        self.done[k][idx] = True

    def ok(self, k, ch):
        """층 k 의 조각 ch 가 시작 가능한가 (층 k-1 의 겹치는 조각이 모두 끝났는가)."""
        if k - 1 not in self.tree or self.tree[k - 1] is None:
            return True
        q = LineString(np.asarray(ch, float)[:, :2]).buffer(SUPPORT_R)
        hits = self.tree[k - 1].query(q)
        d = self.done[k - 1]
        return all(d[int(h)] for h in hits)


def run(job: Path, opts):
    cfg = load_config(job / "config.yaml")
    mesh = load_target_mesh(job / "target.stl", cfg)
    pr = cfg.process
    arc_on = (pr.arc_on_time_s or 0.0) if opts.arc else 0.0
    arc_off = (pr.arc_off_time_s or 0.0) if opts.arc else 0.0
    vd = pr.deposition_speed_mm_s
    required = max(r.arm_envelope_radius_mm for r in cfg.robots) * 2 + cfg.collision.arm_clearance_mm

    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    layers = slice_target_layers(mesh, list(range(n)), cfg)
    robots = [Robot(r.id, r.base_xyz_mm, r.home_xyz_mm) for r in sorted(cfg.robots, key=lambda r: r.id)]
    v1.DIRS = [v1.ang_of(rb.base) for rb in robots]
    chk = Checker(robots, required, getattr(opts, "max_tries", MAX_TRIES))
    retract = getattr(opts, "retract", RETRACT_R)
    P = {"opts": opts, "nrob": int(getattr(opts, "robots", 3) or 3),
         "band_dir": getattr(opts, "band_dir", "alt") or "alt",
         "order2": bool(getattr(opts, "order2", False)),
         "link_gap": float(getattr(opts, "link_gap", 0.0) or 0.0)}

    path_cache = None
    if getattr(opts, "paths", None):
        import pickle
        with open(opts.paths, "rb") as _f:
            path_cache = pickle.load(_f)

    def layer_paths(k):
        A = layers[k]
        if A.is_empty:
            return None, 1.0
        if path_cache is not None and k in path_cache:
            return path_cache[k]
        paths, sc, _ = best_paths(A)
        return paths, sc["iou"]

    zs = {k: pr.build_plane_z_mm + (k + 0.5) * pr.layer_height_mm for k in range(n)}
    valid = [k for k in range(n) if not layers[k].is_empty]
    min_iou, yields = 1.0, 0
    parked = [True] * 3
    sup = Support()
    queues = {i: [] for i in range(3)}
    chunks_by_layer = {}            # k -> list of ch (지지 판정용 인덱스)
    next_i = 0                      # 다음에 열 층 (valid 의 인덱스)
    lowest = 0                      # 아직 조각이 남아 있는 가장 낮은 층 (valid 인덱스)
    remaining = {}                  # k -> 남은 조각 수

    for rb in robots:               # 시작: 주차점에서 첫 층 높이로
        rb.travel([rb.park_xy[0], rb.park_xy[1], zs[valid[0]] + LIFT])
    chk.sync()

    def open_layer():
        nonlocal next_i, min_iou
        k = valid[next_i]
        paths, iou_k = layer_paths(k)
        min_iou = min(min_iou, iou_k)
        starts = {i: (queues[i][-1][3][-1] if queues[i] else None) for i in range(3)}
        q = build_queue(k, paths, zs[k], zs[k] + LIFT, robots, P, starts)
        chs = []
        for i in range(3):
            for e in q[i]:
                idx = len(chs)
                chs.append(e[3])
                queues[i].append(e + (idx,))          # (k, pid, ci, ch, z, zl, idx)
        chunks_by_layer[k] = chs
        sup.add_layer(k, chs)
        remaining[k] = len(chs)
        next_i += 1

    def go_park(j, zl):
        rb = robots[j]
        if np.hypot(*(rb.pos[:2] - rb.park_xy)) < 1e-6:
            parked[j] = True
            return True
        v = rb.pos[:2] - rb.base
        L = float(np.hypot(*v))
        mid = rb.base + v * (retract / L) if L > retract else rb.pos[:2].copy()

        def mk(s):
            p = [(s, float(rb.pos[0]), float(rb.pos[1]))]
            t = s + abs(max(rb.pos[2], zl) - rb.pos[2]) / TRAVEL_V
            p.append((t, float(rb.pos[0]), float(rb.pos[1])))
            t += float(np.hypot(*(mid - rb.pos[:2]))) / TRAVEL_V
            p.append((t, float(mid[0]), float(mid[1])))
            t += float(np.hypot(*(rb.park_xy - mid))) / TRAVEL_V
            p.append((t, float(rb.park_xy[0]), float(rb.park_xy[1])))
            return p
        s, _ = chk.earliest(j, mk, rb.t)
        if s is None:
            return False
        rb.idle_until(s)
        rb.travel([rb.pos[0], rb.pos[1], max(rb.pos[2], zl)])
        rb.travel([mid[0], mid[1], max(rb.pos[2], zl)])
        rb.travel([rb.park_xy[0], rb.park_xy[1], zl])
        chk.sync(j)
        parked[j] = True
        return True

    open_layer()
    arc_open = [False] * 3
    last_key = [None] * 3
    while True:
        # 큐가 빈 로봇이 있고 아직 열 층이 남았으며 너무 앞서지 않았으면 다음 층을 연다
        while next_i < len(valid) and any(not queues[i] for i in range(3)) and next_i - lowest <= AHEAD:
            open_layer()
        if not any(queues[i] for i in range(3)):
            break
        best, bkey, blocked = None, (float("inf"), 0), set()
        unsupported = 0
        for i in range(3):
            if not queues[i]:
                continue
            rb = robots[i]
            pos = rb.pos.copy()
            if not sup.ok(queues[i][0][0], queues[i][0][3]):
                unsupported += 1                 # 밑층이 아직 안 끝남 → 순서를 지키려 기다린다
                continue
            for qi in range(min(opts.lookahead, len(queues[i]))):
                k, pid, ci, ch, z, zl, idx = queues[i][qi]
                if not sup.ok(k, ch):
                    break
                cont = (qi == 0 and arc_open[i] and last_key[i] == (k, pid, ci - 1)
                        and np.hypot(*(pos[:2] - ch[0])) < 1e-6)
                if cont:
                    pts, _ = plan(pos, rb.t, ch, z, vd, arc_on, cont=True)
                    if not chk.conflict(i, pts):
                        key = (rb.t, qi)
                        if best is None or key < bkey:
                            best, bkey = (rb.t, qi, i, "cont"), key
                        break
                off = arc_off if arc_open[i] else 0.0
                s0 = rb.t + off
                s, bl = chk.earliest(i, lambda s_, ch=ch, z=z: plan(pos, s_, ch, z, vd, arc_on)[0], s0)
                if s is None:
                    blocked.update(bl)
                    continue
                key = (s, qi)
                if best is None or key < bkey:
                    best, bkey = (s, qi, i, "direct"), key
                if abs(s - s0) < 1e-9:
                    break

        if best is None:
            # 지지 조건 때문에만 막혔다면: 밑층 조각을 든 로봇이 일하는 중 → 가장 빠른 로봇 시각까지 잠깐 기다린다
            if not blocked and unsupported:
                tmin = min(chk.tl[j].end for j in range(3))
                for j in range(3):
                    if queues[j] and robots[j].t < tmin + 1e-9:
                        robots[j].wait(max(1.0, tmin - robots[j].t))
                        chk.sync(j)
                continue
            moved = False
            for j in sorted(blocked | {0, 1, 2}, key=lambda j: robots[j].t):
                if parked[j]:
                    continue
                if arc_open[j]:
                    robots[j].wait(arc_off)
                    arc_open[j] = False
                zl_j = (queues[j][0][5] if queues[j] else robots[j].pos[2] + LIFT)
                if go_park(j, max(zl_j, robots[j].pos[2])):
                    yields += 1
                    moved = True
                    break
            if not moved:
                info = [(j + 1, parked[j], round(robots[j].t, 1), len(queues[j])) for j in range(3)]
                raise RuntimeError(f"교착 상태 — (로봇, 주차, 시각, 남은조각)={info}")
            continue

        s, qi, i, kind = best
        rb = robots[i]
        k, pid, ci, ch, z, zl, idx = queues[i].pop(qi)
        if kind != "cont":
            if arc_open[i]:
                rb.wait(arc_off)
                arc_open[i] = False
            rb.idle_until(s)
            rb.travel([rb.pos[0], rb.pos[1], max(rb.pos[2], zl)])
            rb.travel([ch[0][0], ch[0][1], zl])
            rb.travel([ch[0][0], ch[0][1], z])
            rb.wait(arc_on)
        rb.deposit(ch, z, vd)
        arc_open[i] = True
        parked[i] = False
        last_key[i] = (k, pid, ci)
        chk.sync(i)
        sup.finish(k, idx)
        remaining[k] -= 1
        while lowest < next_i and remaining.get(valid[lowest], 0) == 0:
            lowest += 1

    for i, rb in enumerate(robots):
        if arc_open[i]:
            rb.wait(arc_off)
        chk.sync(i)
    for rb in robots:
        last = rb.rows[-1]
        rb.rows[-1] = (last[0], last[1], last[2], last[3], "W")
        if len(rb.rows) < 2:
            rb.rows.append((last[0] + 1.0, last[1], last[2], last[3], "W"))
    return robots, max(rb.rows[-1][0] for rb in robots), min_iou, n, yields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--no-arc", dest="arc", action="store_false")
    ap.add_argument("--chunk", type=float, default=160.0)
    ap.add_argument("--band", type=float, default=60.0)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--lookahead", type=int, default=6)
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--max-tries", dest="max_tries", type=int, default=MAX_TRIES)
    ap.add_argument("--retract", type=float, default=RETRACT_R)
    ap.add_argument("--robots", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--band-dir", dest="band_dir", choices=["out", "in", "alt"], default="alt")
    ap.add_argument("--order2", action="store_true")
    ap.add_argument("--link-gap", dest="link_gap", type=float, default=9.0)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    job, out = Path(a.job), Path(a.out)
    robots, mk, iou, n, yields = run(job, a)
    nrows = v8.write(job, out, robots)
    print(f"{job.name}: layers={n} rows={nrows} makespan={mk:.1f}s ({mk / 3600:.2f}h) "
          f"self_min_IoU={iou:.3f} yields={yields} gen={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
