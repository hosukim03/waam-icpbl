# -*- coding: utf-8 -*-
"""포트폴리오 생성의 총 계산량(CPU 시간)을 잰다 — 코어 수와 무관한 생성 시간 기준.

시계 시간(wall)은 코어가 많을수록 짧아지므로 컴퓨터마다 다르다. 모든 하위 프로세스가 쓴 CPU 시간을
더하면 '코어 1개로 돌렸을 때 걸릴 시간'에 가깝다. 알고리즘은 건드리지 않고, 실행 중 0.1초마다
하위 프로세스를 훑어 프로세스별 마지막 CPU 시간을 기록한 뒤 더한다(끝나기 직전 0.1초 미만은 빠질 수 있다).

usage: python cpu_time.py <형상폴더> <결과이름> [portfolio.py 에 넘길 인자...]   (WAAM venv 파이썬)
  예) python cpu_time.py ../data/check808 cpu_full --full
결과: results/<결과이름>.csv  (job, makespan_s, wall_s, cpu_s, cpu/wall, 최대 동시 프로세스 수, 뽑힌 설정)
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
VPY = ROOT / "WAAM_Validator" / ".venv" / "Scripts" / "python.exe"


def measure(job: Path, out: Path, extra: list[str]) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    p = subprocess.Popen([str(VPY), str(ROOT / "src" / "portfolio.py"), str(job), str(out), *extra],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT / "src")
    root = psutil.Process(p.pid)
    seen: dict[int, float] = {}
    peak = 0
    while p.poll() is None:
        try:
            procs = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            break
        busy = 0
        for q in procs:
            try:
                ct = q.cpu_times()
                seen[q.pid] = ct.user + ct.system
                busy += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak = max(peak, busy)
        time.sleep(0.1)
    return time.perf_counter() - t0, sum(seen.values()), peak


def main() -> None:
    src, name, extra = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3:]
    rows = []
    for d in sorted(x for x in src.iterdir() if (x / "target.stl").exists()):
        out = ROOT / "jobs_eval" / name / d.name
        wall, cpu, peak = measure(d, out, extra)
        mk, best = "", ""
        if (out / "trajectory.csv").exists():
            with open(out / "trajectory.csv", newline="") as f:
                mk = f"{max(float(r['time_s']) for r in csv.DictReader(f)):.1f}"
            best = next((l.split("	")[1].strip() for l in (out / "portfolio.txt").read_text(encoding="utf-8").splitlines()
                         if l.startswith("best")), "")
        rows.append({"job": d.name, "makespan_s": mk, "best": best, "wall_s": f"{wall:.1f}", "cpu_s": f"{cpu:.1f}",
                     "cpu_per_wall": f"{cpu / wall:.2f}", "peak_procs": peak})
        print(f"{name} {d.name} makespan={mk} [{best}] wall={wall:.1f}s cpu={cpu:.1f}s ({cpu / wall:.1f}배) 동시 프로세스 최대 {peak}", flush=True)
    path = ROOT / "results" / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tw = sum(float(r["wall_s"]) for r in rows)
    tc = sum(float(r["cpu_s"]) for r in rows)
    print(f"{name} 합계 wall={tw:.1f}s cpu={tc:.1f}s ({tc / tw:.1f}배) → {path}", flush=True)


if __name__ == "__main__":
    main()
