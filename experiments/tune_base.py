# -*- coding: utf-8 -*-
"""학습 형상(train_div)에 경로 캐시를 만들고, 세 설정 기본값의 makespan·시간을 잰다(튜닝 기준선)."""
import json, multiprocessing as mp, re, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
VPY = ROOT / "WAAM_Validator/.venv/Scripts/python.exe"
SRC = ROOT / "data" / "train_div"
WORK = ROOT / "tune_work"

def sched(args):
    job, var, extra, tag = args
    script = "schedule_v7.py" if var == "v7" else "schedule_v8.py"
    flags = ["--no-balance"] if var == "v8" else []
    out = WORK / "runs" / tag / var / job
    t0 = time.time()
    p = subprocess.run([str(VPY), str(ROOT / "v0" / script), str(SRC / job), str(out),
                        "--paths", str(WORK / "cache" / f"{job}.pkl"), *flags, *extra],
                       capture_output=True, text=True, encoding="utf-8", cwd=ROOT / "WAAM_Validator")
    m = re.search(r"makespan=([\d.]+)s", p.stdout)
    return job, var, float(m.group(1)) if m else None, time.time() - t0

if __name__ == "__main__":
    jobs = sorted(d.name for d in SRC.iterdir() if (d / "target.stl").exists())
    (WORK / "cache").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for j in jobs:
        c = WORK / "cache" / f"{j}.pkl"
        if not c.exists():
            subprocess.run([str(VPY), str(ROOT / "v0/toolpath_cache.py"), str(SRC / j), str(c)],
                           cwd=ROOT / "WAAM_Validator", capture_output=True)
    print(f"캐시 {len(jobs)}개 {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.map(sched, [(j, v, [], "base") for j in jobs for v in ("v7", "v8", "v8bal")], chunksize=1)
    base = {}
    for j, v, mk, dt in res:
        base.setdefault(j, {})[v] = (mk, round(dt, 1))
    json.dump(base, open(WORK / "base.json", "w"), indent=1)
    for j in jobs:
        print(j, base[j])
    print(f"기준선 {time.time()-t0:.0f}s", flush=True)
