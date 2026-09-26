"""RL 스케줄러 = V7 과 같은 구조인데, '어느 후보를 집을지'를 신경망 정책이 고른다.

V7(그리디): 후보 중 시작 시각이 가장 이른 것 → 동률이면 큐 앞쪽
RL        : 후보마다 특징 11개를 뽑아 작은 신경망(11-8-1)이 점수를 매기고 가장 높은 것을 고른다
            학습 때는 softmax 로 뽑고(탐험), 평가 때는 최고점(argmax)만 쓴다.

정책을 주지 않으면(policy=None) V7 과 똑같이 동작한다.
usage: python schedule_rl.py <job> <out> [--weights w.npz] [--chunk 40] [--lookahead 6]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

from schedule_v1 import Robot, LIFT, order_pieces, plen, split
import schedule_v1 as v1
from schedule_v2 import approach_time, chunkify, earliest, hull, retreat_time
from toolpath import best_paths

NFEAT = 11


class Policy:
    """작은 2층 신경망. score = w2·tanh(W1 x + b1) + b2"""

    TEMP = 0.3               # 샘플링 온도 (작을수록 그리디에 가깝게 뽑는다)
    WINDOW_S = 30.0          # 시작이 최선보다 이만큼 늦은 후보는 후보에서 제외

    def __init__(self, hidden=8, seed=0, scale=0.1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, scale, (hidden, NFEAT))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, scale, hidden)
        self.b2 = 0.0

    # --- 저장/불러오기 ---
    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        p = cls(hidden=len(d["b1"]))
        p.W1, p.b1, p.w2, p.b2 = d["W1"], d["b1"], d["w2"], float(d["b2"])
        return p

    # --- 계산 ---
    # 그리디(V7) 규칙을 사전 지식으로 깔아 두고, 신경망은 거기서의 '보정'만 학습한다.
    # 그래서 학습 시작점이 곧 V7 이고, 나빠질 위험 없이 조금씩 개선하는 방향으로 움직인다.
    PRIOR_START = 8.0        # 시작이 이른 후보 선호 (특징 0 = 시작 지연)
    PRIOR_QUEUE = 0.5        # 동률이면 큐 앞쪽 선호 (특징 3 = 큐 위치)

    def scores(self, X):
        H = np.tanh(X @ self.W1.T + self.b1)
        prior = -self.PRIOR_START * X[:, 0] - self.PRIOR_QUEUE * X[:, 3]
        return prior + H @ self.w2 + self.b2, H

    def grad_logp(self, X, H, probs, idx):
        """선택한 후보의 log 확률에 대한 기울기 (REINFORCE 용)."""
        d = -probs
        d[idx] += 1.0
        d /= self.TEMP                                  # dlogp/dscore (온도 반영)
        gw2 = d @ H
        gb2 = d.sum()
        dH = np.outer(d, self.w2) * (1 - H ** 2)
        gW1 = dH.T @ X
        gb1 = dH.sum(axis=0)
        return gW1, gb1, gw2, gb2

    def params(self):
        return [self.W1, self.b1, self.w2, self.b2]

    def add(self, grads, lr):
        self.W1 += lr * grads[0]
        self.b1 += lr * grads[1]
        self.w2 += lr * grads[2]
        self.b2 += lr * grads[3]


def features(cand, ctx):
    """후보 하나를 숫자 11개로 요약 (모두 대략 -3~3 범위가 되도록 나눠 둔다)."""
    s, qi, i, kind, H, dur, ch, dep, appr, cont = cand
    now = ctx["now"]
    loads = ctx["loads"]
    return np.array([
        min((s - now) / 60.0, 5.0),                      # 시작까지 기다리는 시간
        dep / 30.0,                                      # 적층 시간
        appr / 10.0,                                     # 접근 이동 시간
        qi / 6.0,                                        # 큐에서의 위치
        float(np.mean(np.hypot(ch[:, 0], ch[:, 1]))) / 700.0,   # 조각의 중심 거리(가운데인지 바깥인지)
        1.0 if cont else 0.0,                            # 아크를 끄지 않고 이어가는가
        1.0 if kind == "retreat" else 0.0,               # 주차점까지 물러났다 가는가
        (ctx["rt"][i] - now) / 60.0,                     # 이 로봇이 얼마나 앞서 있나
        (loads[i] - loads.mean()) / 600.0,               # 이 로봇의 누적 부하 편차
        min(ctx["qlen"][i] / 50.0, 3.0),                 # 남은 조각 수
        min(ctx["hdist"][i] / 500.0, 2.0),               # 다른 로봇 작업영역과의 거리
    ])


def run(job: Path, opts, policy: Policy | None = None, rng=None, trace=None, cache=None):
    """한 작업(job)을 스케줄링한다. trace 가 주어지면 (X, H, probs, idx) 를 기록한다."""
    cfg = load_config(job / "config.yaml")
    mesh = load_target_mesh(job / "target.stl", cfg)
    pr = cfg.process
    arc_on = (pr.arc_on_time_s or 0.0) if opts.arc else 0.0
    arc_off = (pr.arc_off_time_s or 0.0) if opts.arc else 0.0
    vd = pr.deposition_speed_mm_s
    clear = max(r.arm_envelope_radius_mm for r in cfg.robots) * 2 + cfg.collision.arm_clearance_mm + 5.0

    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    if getattr(opts, "max_layers", 0):
        n = min(n, opts.max_layers)
    layers = slice_target_layers(mesh, list(range(n)), cfg)
    robots = [Robot(r.id, r.base_xyz_mm, r.home_xyz_mm) for r in sorted(cfg.robots, key=lambda r: r.id)]
    v1.DIRS = [v1.ang_of(rb.base) for rb in robots]
    base3 = [np.array([rb.base[0], rb.base[1], 0.0]) for rb in robots]

    clock, min_iou, first = 0.0, 1.0, True
    loads = np.zeros(3)
    steps, max_steps = 0, getattr(opts, "max_steps", 0)
    for k in range(n):
        A = layers[k]
        if A.is_empty:
            continue
        z = pr.build_plane_z_mm + (k + 0.5) * pr.layer_height_mm
        zl = z + LIFT
        if cache is not None and k in cache:            # 같은 형상을 반복 평가할 때 경로 재사용(학습용)
            paths, iou_k = cache[k]
        else:
            paths, sc, _ = best_paths(A)
            iou_k = sc["iou"]
            if cache is not None:
                cache[k] = (paths, iou_k)
        min_iou = min(min_iou, iou_k)

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

        groups = {}
        for P in paths:
            for (_, o), pc in split(P, v1.seq_label):
                groups.setdefault(o, []).append(pc)
        queues = {i: [] for i in range(3)}
        for o, pcs in groups.items():
            bands = {}
            for pc in pcs:
                r = float(np.max(np.hypot(pc[:, 0], pc[:, 1])))
                bands.setdefault(int(r // opts.band), []).append(pc)
            ordered, cur = [], robots[o].park_xy
            for b in sorted(bands, reverse=True):
                seg = order_pieces(bands[b], cur)
                ordered += seg
                cur = seg[-1][-1]
            for pid, pc in enumerate(ordered):
                for ci, ch in enumerate(chunkify(pc, opts.chunk)):
                    queues[o].append((pid, ci, ch))

        booked = []
        arc_open = [False] * 3
        last_key = [None] * 3
        while any(queues[i] for i in range(3)):
            cands = []
            for i in range(3):
                if not queues[i]:
                    continue
                rb = robots[i]
                pos = rb.pos
                for qi in range(min(opts.lookahead, len(queues[i]))):
                    pid, ci, ch = queues[i][qi]
                    cont = (qi == 0 and arc_open[i] and last_key[i] == (pid, ci - 1)
                            and np.hypot(*(pos[:2] - ch[0])) < 1e-6)
                    dep = plen(ch) / vd
                    ret_end = retreat_time([ch[-1][0], ch[-1][1], z], rb.park_xy, z)
                    if cont:
                        H = hull([base3[i], rb.park_xy, pos, *ch])
                        s = earliest(rb.t, dep + arc_off + ret_end, H, i, booked, clear)
                        if abs(s - rb.t) < 1e-9:
                            cands.append((s, qi, i, "cont", H, dep, ch, dep, 0.0, True))
                            break
                    off = arc_off if arc_open[i] else 0.0
                    s_req = rb.t + off
                    appr = approach_time(pos, ch[0], z)
                    dur = appr + arc_on + dep
                    H = hull([base3[i], rb.park_xy, pos, *ch])
                    s = earliest(s_req, dur + arc_off + ret_end, H, i, booked, clear)
                    if abs(s - s_req) < 1e-9:
                        cands.append((s, qi, i, "direct", H, dur, ch, dep, appr, False))
                    else:
                        ret = retreat_time(pos, rb.park_xy, z)
                        park3 = [rb.park_xy[0], rb.park_xy[1], zl]
                        appr2 = approach_time(park3, ch[0], z)
                        dur2 = appr2 + arc_on + dep
                        H2 = hull([base3[i], rb.park_xy, *ch])
                        s2 = earliest(s_req + ret, dur2 + arc_off + ret_end, H2, i, booked, clear)
                        cands.append((s2, qi, i, "retreat", H2, dur2, ch, dep, appr2 + ret, False))
                    if abs(cands[-1][0] - rb.t) < 1e-9:
                        break

            if policy is None:
                pick = min(range(len(cands)), key=lambda j: (cands[j][0], cands[j][1]))
            else:
                now = min(rb.t for rb in robots)
                hd = []
                for i in range(3):
                    p = Point(float(robots[i].pos[0]), float(robots[i].pos[1]))
                    others = [b[3] for b in booked[-4:] if b[0] != i]
                    hd.append(min((g.distance(p) for g in others), default=1000.0))
                ctx = {"now": now, "loads": loads, "rt": [rb.t for rb in robots],
                       "qlen": [len(queues[i]) for i in range(3)], "hdist": hd}
                X = np.array([features(c, ctx) for c in cands])
                sc_, Hh = policy.scores(X)
                s0 = min(c[0] for c in cands)
                near = np.array([c[0] <= s0 + policy.WINDOW_S for c in cands])
                sc_ = np.where(near, sc_, -1e9)          # 너무 늦게 시작하는 후보는 제외
                if rng is None:
                    pick = int(np.argmax(sc_))
                else:
                    e = np.exp((sc_ - sc_.max()) / policy.TEMP)
                    p = e / e.sum()
                    pick = int(rng.choice(len(p), p=p))
                    if trace is not None:
                        trace.append((X, Hh, p, pick))

            steps += 1
            if max_steps and steps > max_steps:          # 병적으로 오래 끄는 정책은 조기 중단
                return robots, float("inf"), min_iou, n
            s, qi, i, kind, H, dur, ch, dep, appr, cont = cands[pick]
            rb = robots[i]
            pid, ci, ch = queues[i].pop(qi)
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
            t_before = rb.t
            rb.deposit(ch, z, vd)
            loads[i] += rb.t - t_before
            arc_open[i] = True
            last_key[i] = (pid, ci)
            ret_end = retreat_time([ch[-1][0], ch[-1][1], z], rb.park_xy, z)
            booked.append((i, s, rb.t + arc_off + ret_end, H))
            if len(booked) > 400:
                cut = min(rb2.t for rb2 in robots) - 1.0
                booked = [b for b in booked if b[2] > cut]

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
    return robots, max(rb.rows[-1][0] for rb in robots), min_iou, n


def write(job, out, robots):
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
    return nrows


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--no-arc", dest="arc", action="store_false")
    ap.add_argument("--chunk", type=float, default=40.0)
    ap.add_argument("--band", type=float, default=60.0)
    ap.add_argument("--lookahead", type=int, default=6)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--max-layers", type=int, default=0, help="학습용: 앞쪽 N개 층만 스케줄링")
    ap.add_argument("--max-steps", type=int, default=0, help="학습용: 배차 결정 횟수 상한(넘으면 중단)")
    return ap.parse_args(argv)


def main():
    a = parse()
    t0 = time.time()
    job, out = Path(a.job), Path(a.out)
    pol = Policy.load(a.weights) if a.weights else None
    robots, mk, iou, n = run(job, a, policy=pol)
    nrows = write(job, out, robots)
    print(f"{job.name}: layers={n} rows={nrows} makespan={mk:.1f}s ({mk / 3600:.2f}h) "
          f"self_min_IoU={iou:.3f} gen={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
