"""V8 스케줄러 = V7 + '채점 기준 그대로' 충돌 판정

V7 까지의 충돌 판정: 작업 하나마다 볼록 영역(베이스·주차점·현재 위치·조각)을 통째로 예약하고,
                    두 영역이 250mm 안이면 시간이 겹치지 않게 했다. 실제 팔이 지나가지 않는 곳까지 막아
                    동시에 일할 기회를 크게 놓친다 (04번: 로봇당 적층 4~5h 인데 makespan 12h).
V8 의 충돌 판정: 채점기와 같은 규칙을 시간축에서 직접 검사한다.
  - 팔 = XY 평면의 선분 (베이스 → 현재 토치 위치)
  - 모든 시각에 두 팔 선분의 최단거리 ≥ 250mm (팔 반경 100+100 + 여유 50) + 안전 여유
  - 각 로봇의 이미 확정된 궤적(rows)을 시간 보간해서, 후보 작업의 계획 궤적과 0.1초 간격으로 비교
  - 확정된 궤적 뒤로는 '마지막 위치에 그대로 서 있다'고 본다 (실제로도 다음 명령 전까지 그 자리에 있음)
  - 모두 막혀 아무도 움직일 수 없으면, 서 있는 로봇 하나를 주차점으로 물린다(양보)
조각 분할·순서 규칙과 후보 선택(가장 일찍 시작 가능한 것)은 V7 과 같다.
"""
from __future__ import annotations

import argparse
import bisect
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

from schedule_v1 import Robot, LIFT, TRAVEL_V, order_pieces, plen, split
import schedule_v1 as v1
from schedule_v2 import chunkify
from toolpath import best_paths

DT = 0.1            # 검사 간격(s) — 채점기 max_time_step_s 와 같음
MARGIN = 20.0       # 샘플 사이 이동(최대 150mm/s × 0.1s × 2대 / 2)을 덮는 안전 여유(mm)
MAX_TRIES = 25      # 시작 시각 후보를 몇 개까지 시험할지
RETRACT_R = 500.0   # 비켜 설 때 먼저 팔을 이 길이(베이스로부터)까지 당긴다


# ---------------------------------------------------------------- 기하
def seg_dist(a0, a1, b0, b1):
    """XY 선분 쌍들의 최단거리 (a0,b0: (2,), a1,b1: (N,2))."""
    def proj(p, s, e):
        d = e - s
        dd = np.einsum("ij,ij->i", d, d)
        t = np.divide(np.einsum("ij,ij->i", p - s, d), dd, out=np.zeros_like(dd), where=dd > 0)
        return s + np.clip(t, 0, 1)[:, None] * d

    n = len(a1)
    A0 = np.broadcast_to(a0, (n, 2))
    B0 = np.broadcast_to(b0, (n, 2))
    c = [np.hypot(*(A0 - proj(A0, B0, b1)).T),
         np.hypot(*(a1 - proj(a1, B0, b1)).T),
         np.hypot(*(B0 - proj(B0, A0, a1)).T),
         np.hypot(*(b1 - proj(b1, A0, a1)).T)]
    dist = np.minimum.reduce(c)
    # 교차하면 0
    da, db = a1 - A0, b1 - B0
    cross = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    den = cross(da, db)
    off = B0 - A0
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = cross(off, db) / den
        tb = cross(off, da) / den
    hit = (np.abs(den) > 1e-12) & (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1)
    dist[hit] = 0.0
    return dist


class Timeline:
    """로봇의 확정 궤적을 시간으로 조회 (rows 를 따라가며 목록을 늘린다)."""

    def __init__(self, rb):
        self.rb, self.T, self.X, self.Y, self.n = rb, [], [], [], 0

    def sync(self):
        rows = self.rb.rows
        if self.n and rows[self.n - 1][0] != self.T[-1]:
            # 마지막 행이 덮어써졌을 수 있음 → 마지막 행부터 다시
            self.n -= 1
            self.T.pop(); self.X.pop(); self.Y.pop()
        for r in rows[self.n:]:
            if self.T and r[0] <= self.T[-1]:
                self.X[-1], self.Y[-1] = r[1], r[2]
                continue
            self.T.append(r[0]); self.X.append(r[1]); self.Y.append(r[2])
        self.n = len(rows)

    def xy(self, ts):
        i0 = max(bisect.bisect_right(self.T, ts[0]) - 1, 0)
        i1 = min(bisect.bisect_left(self.T, ts[-1]) + 1, len(self.T))
        T = self.T[i0:i1]
        return np.column_stack([np.interp(ts, T, self.X[i0:i1]), np.interp(ts, T, self.Y[i0:i1])])

    def breakpoints(self, t0, t1):
        i0 = bisect.bisect_right(self.T, t0)
        i1 = bisect.bisect_right(self.T, t1)
        return self.T[i0:i1]

    @property
    def end(self):
        return self.T[-1]


# ---------------------------------------------------------------- 계획 궤적
def plan(pos, t0, ch, z, vd, arc_on, via_park=None, cont=False):
    """(시각, x, y) 키포인트 목록과 끝나는 시각. XY 만 필요 (들어올림·내림은 제자리)."""
    pts = [(t0, float(pos[0]), float(pos[1]))]
    t = t0
    zl = z + LIFT
    cur = np.asarray(pos, float)
    if not cont:
        if via_park is not None:
            t += abs(max(cur[2], zl) - cur[2]) / TRAVEL_V
            pts.append((t, cur[0], cur[1]))
            t += float(np.hypot(*(via_park - cur[:2]))) / TRAVEL_V
            cur = np.array([via_park[0], via_park[1], zl])
            pts.append((t, cur[0], cur[1]))
        t += abs(max(cur[2], zl) - cur[2]) / TRAVEL_V
        pts.append((t, cur[0], cur[1]))
        t += float(np.hypot(ch[0][0] - cur[0], ch[0][1] - cur[1])) / TRAVEL_V
        pts.append((t, float(ch[0][0]), float(ch[0][1])))
        t += LIFT / TRAVEL_V + arc_on
        pts.append((t, float(ch[0][0]), float(ch[0][1])))
    for p, q in zip(ch[:-1], ch[1:]):
        d = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        if d < 0.01:
            continue
        t += d / vd
        pts.append((t, float(q[0]), float(q[1])))
    return pts, t


def sample_plan(pts):
    T = np.array([p[0] for p in pts])
    ts = np.arange(T[0], T[-1] + 1e-9, DT)
    if ts[-1] < T[-1]:
        ts = np.append(ts, T[-1])
    ts = np.union1d(ts, T)
    xy = np.column_stack([np.interp(ts, T, [p[1] for p in pts]), np.interp(ts, T, [p[2] for p in pts])])
    return ts, xy


class Checker:
    def __init__(self, robots, required):
        self.robots = robots
        self.tl = [Timeline(rb) for rb in robots]
        self.req = required + MARGIN

    def sync(self, i=None):
        for j, tl in enumerate(self.tl):
            if i is None or j == i:
                tl.sync()

    def conflict(self, i, pts):
        """충돌하는 다른 로봇 번호 목록 (없으면 빈 목록).
        계획 구간뿐 아니라, 계획이 끝난 뒤 '그 자리에 서 있는' 꼬리 구간도
        다른 로봇이 이미 확정한 미래 궤적과 비교한다."""
        ts, xy = sample_plan(pts)
        base_i = self.robots[i].base
        t_end, last = ts[-1], xy[-1]
        hits = []
        for j, rb in enumerate(self.robots):
            if j == i:
                continue
            oxy = self.tl[j].xy(ts)
            d = seg_dist(base_i, xy, rb.base, oxy)
            bad = float(d.min()) < self.req
            if not bad and self.tl[j].end > t_end:        # 꼬리 구간: 나는 서 있고 상대는 움직임
                tt = np.arange(t_end, self.tl[j].end + DT, DT)
                d2 = seg_dist(base_i, np.broadcast_to(last, (len(tt), 2)), rb.base, self.tl[j].xy(tt))
                bad = float(d2.min()) < self.req
            if bad:
                hits.append(j)
        return hits

    def earliest(self, i, make_pts, s0, horizon=900.0):
        """make_pts(s) 계획이 충돌 없는 가장 이른 시작 시각. 못 찾으면 (None, 막은 로봇들)."""
        pts = make_pts(s0)
        hits = self.conflict(i, pts)
        if not hits:
            return s0, []
        cands = set()
        for j in range(len(self.robots)):
            if j != i:
                cands.update(self.tl[j].breakpoints(s0, s0 + horizon))
        cands = sorted(cands)
        if len(cands) > MAX_TRIES:                      # 고르게 추려서 시험
            idx = np.linspace(0, len(cands) - 1, MAX_TRIES).astype(int)
            cands = [cands[k] for k in idx]
        blockers = set(hits)
        for s in cands:
            pts = make_pts(s)
            hits = self.conflict(i, pts)
            if not hits:
                return s, []
            blockers.update(hits)
        # 다른 로봇이 모두 멈춘 뒤(궤적 끝)에도 막히면 무한 대기 → 양보 필요
        s_end = max(self.tl[j].end for j in range(len(self.robots)) if j != i)
        s_end = max(s_end, s0)
        pts = make_pts(s_end)
        hits = self.conflict(i, pts)
        if not hits:
            return s_end, []
        return None, sorted(blockers | set(hits))


# ---------------------------------------------------------------- 본체
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
    chk = Checker(robots, required)
    parked = [False] * 3
    min_iou, first, yields = 1.0, True, 0

    def go_park(j, zl):
        """로봇 j 를 주차점으로 물린다 (충돌 없는 시각을 찾아서)."""
        rb = robots[j]
        if np.hypot(*(rb.pos[:2] - rb.park_xy)) < 1e-6:
            parked[j] = True
            return True
        # 먼저 자기 팔 선분을 따라 베이스 쪽으로 당긴다(팔이 기존 선분의 일부가 되므로 새 충돌이 없음)
        v = rb.pos[:2] - rb.base
        L = float(np.hypot(*v))
        mid = rb.base + v * (RETRACT_R / L) if L > RETRACT_R else rb.pos[:2].copy()

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
            first = False
            chk.sync()
            parked = [True] * 3
        clock = max(rb.t for rb in robots)
        for rb in robots:
            rb.idle_until(clock)
            if abs(rb.pos[2] - zl) > 1e-6:
                rb.travel([rb.pos[0], rb.pos[1], zl])
        clock = max(rb.t for rb in robots)
        for rb in robots:
            rb.idle_until(clock)
        chk.sync()

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

        arc_open = [False] * 3
        last_key = [None] * 3
        while any(queues[i] for i in range(3)):
            best, blocked = None, set()
            for i in range(3):
                if not queues[i]:
                    continue
                rb = robots[i]
                pos = rb.pos.copy()
                for qi in range(min(opts.lookahead, len(queues[i]))):
                    pid, ci, ch = queues[i][qi]
                    cont = (qi == 0 and arc_open[i] and last_key[i] == (pid, ci - 1)
                            and np.hypot(*(pos[:2] - ch[0])) < 1e-6)
                    if cont:
                        pts, _ = plan(pos, rb.t, ch, z, vd, arc_on, cont=True)
                        if not chk.conflict(i, pts):
                            cand = (rb.t, qi, i, "cont")
                            if best is None or cand[:2] < best[:2]:
                                best = cand
                            break
                    off = arc_off if arc_open[i] else 0.0
                    s0 = rb.t + off
                    s, bl = chk.earliest(i, lambda s_, ch=ch: plan(pos, s_, ch, z, vd, arc_on)[0], s0)
                    if s is None:
                        blocked.update(bl)
                        continue
                    cand = (s, qi, i, "direct")
                    if best is None or cand[:2] < best[:2]:
                        best = cand
                    if abs(s - s0) < 1e-9:
                        break

            if best is None:
                # 모두 막힘 → 막고 서 있는 로봇 중 하나를 주차점으로 물린다
                moved = False
                for j in sorted(blocked | {0, 1, 2}, key=lambda j: robots[j].t):
                    if parked[j]:
                        continue
                    if arc_open[j]:
                        robots[j].wait(arc_off)
                        arc_open[j] = False
                    if go_park(j, zl):
                        yields += 1
                        moved = True
                        break
                if not moved:
                    chk.sync()
                    for a_ in range(3):
                        for b_ in range(a_ + 1, 3):
                            d_ = seg_dist(robots[a_].base, robots[a_].pos[:2][None, :], robots[b_].base,
                                          robots[b_].pos[:2][None, :])[0]
                            print(f"  [진단] 로봇{a_ + 1}-{b_ + 1} 현재 팔 거리 {d_:.1f}mm  "
                                  f"pos{a_ + 1}={robots[a_].pos[:2].round(1)} pos{b_ + 1}={robots[b_].pos[:2].round(1)}")
                    info = [(j + 1, parked[j], round(robots[j].t, 1), len(queues[j])) for j in range(3)]
                    raise RuntimeError(f"layer {k}: 교착 상태 — (로봇, 주차, 시각, 남은조각)={info}")
                continue

            s, qi, i, kind = best
            rb = robots[i]
            pid, ci, ch = queues[i].pop(qi)
            if kind != "cont":
                if arc_open[i]:
                    rb.wait(arc_off)
                    arc_open[i] = False
                rb.idle_until(s)
                rb.travel([rb.pos[0], rb.pos[1], max(rb.pos[2], zl)])
                rb.travel([ch[0][0], ch[0][1], zl])
                rb.travel([ch[0][0], ch[0][1], z])
                rb.wait(arc_on)
            t_before_commit = s if kind != "cont" else rb.t
            rb.deposit(ch, z, vd)
            arc_open[i] = True
            parked[i] = False
            last_key[i] = (pid, ci)
            chk.sync(i)
            if opts.debug:                                   # 확정한 실제 궤적을 다시 검사
                tl = chk.tl[i]
                ts = np.arange(t_before_commit, rb.t, DT)
                if len(ts):
                    mine = tl.xy(ts)
                    for j in range(3):
                        if j == i:
                            continue
                        d_ = seg_dist(rb.base, mine, robots[j].base, chk.tl[j].xy(ts))
                        if d_.min() < required:
                            m_ = int(d_.argmin())
                            print(f"  [위반] 층{k} 로봇{i + 1}({kind}) vs 로봇{j + 1}: {d_.min():.1f}mm "
                                  f"t={ts[m_]:.1f} 계획시작={s:.1f} 로봇{j + 1}궤적끝={chk.tl[j].end:.1f}")

        # 층 끝: 아크 끄고 주차점으로
        for i, rb in enumerate(robots):
            if arc_open[i]:
                rb.wait(arc_off)
                arc_open[i] = False
            chk.sync(i)
        for i in sorted(range(3), key=lambda i: robots[i].t):
            if not go_park(i, zl):
                raise RuntimeError(f"layer {k}: 층 끝 주차 실패 (로봇 {i + 1})")

    for rb in robots:
        last = rb.rows[-1]
        rb.rows[-1] = (last[0], last[1], last[2], last[3], "W")
        if len(rb.rows) < 2:
            rb.rows.append((last[0] + 1.0, last[1], last[2], last[3], "W"))
    return robots, max(rb.rows[-1][0] for rb in robots), min_iou, n, yields


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--no-arc", dest="arc", action="store_false")
    ap.add_argument("--chunk", type=float, default=40.0)
    ap.add_argument("--band", type=float, default=60.0)
    ap.add_argument("--lookahead", type=int, default=6)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    job, out = Path(a.job), Path(a.out)
    robots, mk, iou, n, yields = run(job, a)
    nrows = write(job, out, robots)
    print(f"{job.name}: layers={n} rows={nrows} makespan={mk:.1f}s ({mk / 3600:.2f}h) "
          f"self_min_IoU={iou:.3f} yields={yields} gen={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
