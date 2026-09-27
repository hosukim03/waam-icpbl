"""V0: 한 번에 한 로봇만 움직이는 순차 스케줄 -> trajectory.csv

- 층마다 best_paths 로 중심선 생성
- 각 폴리라인을 120도 섹터 경계에서 끊어 섹터 주인 로봇에 배정 (섹터 전체가 리치 안)
- 조각 순서: 층 안에서 최근접 이웃
- 로봇 교대 시 이전 로봇은 홈으로 복귀한 뒤 다음 로봇 출발 -> 동시에 둘 이상 움직이지 않음
"""
from __future__ import annotations
import argparse, math, shutil, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers
from toolpath import best_paths

TRAVEL_V = 140.0   # 상한 150 (허용 165) 보다 여유
LIFT = 10.0        # 층 위 이동 높이 여유

class Robot:
    def __init__(self, rid, base, home):
        self.rid, self.base, self.home = rid, np.array(base[:2]), np.array(home, float)
        self.rows = [(0.0, *self.home, "W")]
        self.pos, self.t = self.home.copy(), 0.0

    def _row(self, t, p, mode):
        last = self.rows[-1]
        if t <= last[0] + 1e-9:
            # 같은 시각이면 마지막 행의 mode 만 갱신
            self.rows[-1] = (last[0], last[1], last[2], last[3], mode)
            return
        self.rows.append((t, float(p[0]), float(p[1]), float(p[2]), mode))

    def start_at(self, t):
        """대기 중이던 로봇이 시각 t 에 움직이기 시작: 현 위치에 T 행."""
        if t > self.t + 1e-9:
            last = self.rows[-1]
            self.rows[-1] = (last[0], last[1], last[2], last[3], "W")   # 주차 중엔 W
            self._row(t, self.pos, "T")
        else:
            self._row(self.t, self.pos, "T")
        self.t = max(self.t, t)

    def travel(self, q):
        q = np.asarray(q, float)
        d = float(np.linalg.norm(q - self.pos))
        if d < 0.01:
            return
        self._row(self.t, self.pos, "T")
        self.t += d / TRAVEL_V
        self.pos = q
        self._row(self.t, self.pos, "T")

    def wait(self, dt):
        self._row(self.t, self.pos, "W")
        self.t += dt
        self._row(self.t, self.pos, "W")

    def deposit(self, pts, z, v):
        self._row(self.t, self.pos, "D")
        for p in pts[1:]:
            q = np.array([p[0], p[1], z])
            d = float(np.linalg.norm(q[:2] - self.pos[:2]))
            if d < 0.01:      # 0.01mm 미만 조각은 건너뜀 (시각 중복 방지)
                continue
            self._row(self.t, self.pos, "D")
            self.t += d / v
            self.pos = q
            self._row(self.t, self.pos, "D")

DIRS = [180.0, 60.0, 300.0]   # main() 에서 config 의 base 방향으로 덮어씀

def sector_owner(p):
    """점의 방위각에 가장 가까운 베이스 방향의 로봇 index (= 120도 섹터)."""
    a = math.degrees(math.atan2(p[1], p[0]))
    return int(np.argmin([abs((a - d + 180) % 360 - 180) for d in DIRS]))

def split_by_sector(poly):
    """폴리라인을 섹터 경계에서 끊는다. 경계를 넘는 구간은 0.25mm 이내 경계점을 찾아 삽입."""
    pts = [np.asarray(poly[0], float)]
    for p in poly[1:]:
        p = np.asarray(p, float); q = pts[-1]
        if sector_owner(q) != sector_owner(p):
            L = float(np.hypot(*(p - q))); n = max(2, int(math.ceil(L / 0.5)))
            prev = q
            for i in range(1, n + 1):
                cur = q + (p - q) * (i / n)
                if sector_owner(cur) != sector_owner(prev):
                    pts.append((prev + cur) / 2.0)
                prev = cur
        pts.append(p)
    pieces, cur, own = [], [pts[0]], sector_owner(pts[0])
    for i in range(1, len(pts)):
        p = pts[i]
        o = sector_owner((pts[i - 1] + p) / 2.0)
        if o != own:
            if len(cur) >= 2: pieces.append((own, np.array(cur)))
            cur, own = [pts[i - 1]], o
        cur.append(p)
    if len(cur) >= 2: pieces.append((own, np.array(cur)))
    return pieces

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("out")
    ap.add_argument("--arc", action="store_true", default=True)
    ap.add_argument("--no-arc", dest="arc", action="store_false")
    a = ap.parse_args()
    job, out = Path(a.job), Path(a.out)
    t0 = time.time()
    cfg = load_config(job / "config.yaml")
    mesh = load_target_mesh(job / "target.stl", cfg)
    pr = cfg.process
    arc_on = (pr.arc_on_time_s or 0.0) if a.arc else 0.0
    arc_off = (pr.arc_off_time_s or 0.0) if a.arc else 0.0
    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    layers = slice_target_layers(mesh, list(range(n)), cfg)
    robots = [Robot(r.id, r.base_xyz_mm, r.home_xyz_mm) for r in sorted(cfg.robots, key=lambda r: r.id)]
    global DIRS
    DIRS = [math.degrees(math.atan2(rb.base[1], rb.base[0])) for rb in robots]
    reach = {r.id: r.xy_reach_radius_mm for r in cfg.robots}
    clock, active, min_iou = 0.0, None, 1.0
    for k in range(n):
        A = layers[k]
        if A.is_empty: continue
        z = pr.build_plane_z_mm + (k + 0.5) * pr.layer_height_mm
        paths, sc, _ = best_paths(A)
        min_iou = min(min_iou, sc["iou"])
        pieces = [pc for P in paths for pc in split_by_sector(P)]
        # 층 안 조각 순서: 로봇별로 묶고(교대 최소화), 묶음 안은 최근접 이웃
        for own in sorted({o for o, _ in pieces}, key=lambda o: (o != active, o)):
            todo = [pc for o, pc in pieces if o == own]
            rb = robots[own]
            if active is not None and active != own:
                prv = robots[active]
                prv.travel([prv.pos[0], prv.pos[1], max(prv.pos[2], z + LIFT)])
                prv.travel(prv.home)
                prv.wait(0.0) if False else None
                clock = prv.t
            if active != own:
                rb.start_at(clock)
            active = own
            while todo:
                cur = rb.pos[:2]
                best_i, best_d, best_rev = 0, 1e18, False
                for i, pc in enumerate(todo):
                    for rev, end in ((False, pc[0]), (True, pc[-1])):
                        d = np.hypot(*(end - cur))
                        if d < best_d: best_i, best_d, best_rev = i, d, rev
                pc = todo.pop(best_i)
                if best_rev: pc = pc[::-1]
                for p in pc:
                    if np.hypot(p[0] - rb.base[0], p[1] - rb.base[1]) > reach[rb.rid] + 1e-6:
                        raise RuntimeError(f"reach violation robot {rb.rid} layer {k}")
                rb.travel([rb.pos[0], rb.pos[1], max(rb.pos[2], z + LIFT)])
                rb.travel([pc[0][0], pc[0][1], z + LIFT])
                rb.travel([pc[0][0], pc[0][1], z])
                if arc_on > 0: rb.wait(arc_on)
                rb.deposit(pc, z, pr.deposition_speed_mm_s)
                if arc_off > 0: rb.wait(arc_off)
                else: rb._row(rb.t, rb.pos, "T")
            clock = rb.t
    if active is not None:
        rb = robots[active]
        rb.travel([rb.pos[0], rb.pos[1], rb.pos[2] + LIFT]); rb.travel(rb.home)
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
                if tr <= prev:          # 반올림 후에도 엄격 증가 보장
                    tr = round(prev + 1e-6, 6)
                prev = tr
                f.write(f"{rb.rid},{tr:.6f},{x:.6f},{y:.6f},{zz:.6f},{m}\n"); nrows += 1
    mk = max(rb.rows[-1][0] for rb in robots)
    print(f"{job.name}: layers={n} rows={nrows} makespan={mk:.1f}s ({mk/3600:.2f}h) self_min_layer_IoU={min_iou:.3f} gen={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
