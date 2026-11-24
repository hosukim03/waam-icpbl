# -*- coding: utf-8 -*-
"""고치기 전(base)과 후(new) solve.py 를 코어 1개에 묶어 형상마다 번갈아 돌리고, 계산 시간과 결과가 같은지 비교한다.

  - 프로세스를 코어 하나에 묶는다(psutil.cpu_affinity). 수치 라이브러리 스레드도 1개로.
  - 형상마다 순서를 바꿔(ABBA) 노트북 상태가 시간에 따라 바뀌는 영향을 줄인다.
  - 결과 비교: 두 trajectory.csv 가 바이트까지 같은지, makespan 이 같은지.

usage: python time_compare.py <base_src> <new_src> <out_root> <job_dir> [<job_dir> ...]   (WAAM venv 파이썬)
"""
import filecmp
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

base_src, new_src, out_root, *jobs = sys.argv[1:]
out_root = Path(out_root)
out_root.mkdir(parents=True, exist_ok=True)
CORE = 2
psutil.Process().cpu_affinity([CORE])   # 이 스크립트부터 묶어 두면 띄우는 프로세스가 처음부터 같은 코어를 물려받는다
env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
PAT = re.compile(r"makespan=([\d.]+)s 계산 ([\d.]+)s \(자르기 ([\d.]+) 특징 ([\d.]+) 경로 ([\d.]+) 시험·계획 ([\d.]+)"
                 r"(?: 국소 탐색 ([\d.]+))?")


def run(src, job, out):
    out.mkdir(parents=True, exist_ok=True)
    log = out.parent / f"{job.name}.log"
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as f:
        p = psutil.Popen([sys.executable, "-X", "utf8", str(Path(src) / "solve.py"), str(job), str(out)], env=env,
                         stdout=f, stderr=subprocess.STDOUT)
        try:
            p.cpu_affinity([CORE])
        except psutil.Error:
            pass
        cpu = 0.0
        while p.poll() is None:
            try:
                ct = p.cpu_times()
                cpu = ct.user + ct.system
            except psutil.Error:
                pass
            time.sleep(0.2)
    wall = time.perf_counter() - t0
    text = log.read_text(encoding="utf-8", errors="replace")
    m = PAT.search(text)
    if not m:
        raise RuntimeError(f"{src} {job}: 출력 해석 실패\n{text[-2000:]}")
    g = [float(x) if x else 0.0 for x in m.groups()]
    return {"wall": wall, "cpu": cpu, "mk": g[0], "total": g[1], "paths": g[4], "plan": g[5], "ls": g[6]}


bat = psutil.sensors_battery()
print(f"코어 {CORE} 하나에 묶음, 전원 {'연결' if bat is None or bat.power_plugged else '배터리'}"
      + (f" {bat.percent:.0f}%" if bat else ""), flush=True)
rows = []
for n, job in enumerate(jobs):
    job = Path(job)
    order = [("base", base_src), ("new", new_src)]
    if n % 2:
        order.reverse()
    r = {}
    for tag, src in order:
        r[tag] = run(src, job, out_root / tag / job.name)
    same = filecmp.cmp(out_root / "base" / job.name / "trajectory.csv", out_root / "new" / job.name / "trajectory.csv",
                       shallow=False)
    b, w = r["base"], r["new"]
    print(f"{job.name}  전 {b['total']:7.1f}s (계획 {b['plan']:6.1f} 국소 {b['ls']:6.1f})  →  "
          f"후 {w['total']:7.1f}s (계획 {w['plan']:6.1f} 국소 {w['ls']:6.1f})  "
          f"{(w['total'] / b['total'] - 1) * 100:+6.1f}%   makespan {b['mk']:.1f} / {w['mk']:.1f}  "
          f"궤적 {'같음' if same else '다름'}", flush=True)
    rows.append((job.name, b, w, same))
tb = sum(b["total"] for _, b, _, _ in rows)
tw = sum(w["total"] for _, _, w, _ in rows)
print(f"합계  전 {tb:.1f}s → 후 {tw:.1f}s ({(tw / tb - 1) * 100:+.1f}%), 궤적 같음 {sum(s for *_, s in rows)}/{len(rows)}")
