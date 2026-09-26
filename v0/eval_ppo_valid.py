# -*- coding: utf-8 -*-
"""PPO 정책(V8/균형V8 + --policy)을 검증 형상에 돌려, 기존 포트폴리오(pf_valid_c)에 설정으로 추가했을 때 이득을 본다."""
import csv, json, multiprocessing as mp, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
VPY = ROOT / "WAAM_Validator/.venv/Scripts/python.exe"
SRC = ROOT / sys.argv[1]; POL = ROOT / sys.argv[2]; REF = sys.argv[3]
W = ROOT / "tune_work" / ("ppo_eval_" + SRC.name)

def run(a):
    j, bal = a
    out = W / ("bal" if bal else "nb") / j
    p = subprocess.run([str(VPY), str(ROOT / "v0/schedule_v8.py"), str(SRC / j), str(out), "--paths", str(W / "cache" / f"{j}.pkl"),
                        "--policy", str(POL), *([] if bal else ["--no-balance"])], capture_output=True, text=True, encoding="utf-8", cwd=ROOT / "WAAM_Validator")
    m = re.search(r"makespan=([\d.]+)s", p.stdout)
    return j, bal, float(m.group(1)) if m else None

if __name__ == "__main__":
    jobs = sorted(d.name for d in SRC.iterdir() if (d / "target.stl").exists())
    (W / "cache").mkdir(parents=True, exist_ok=True)
    for j in jobs:
        c = W / "cache" / f"{j}.pkl"
        if not c.exists():
            subprocess.run([str(VPY), str(ROOT / "v0/toolpath_cache.py"), str(SRC / j), str(c)], cwd=ROOT / "WAAM_Validator", capture_output=True)
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.map(run, [(j, b) for j in jobs for b in (False, True)], chunksize=1)
    ref = {r["job"]: float(r["makespan_s"]) for r in csv.DictReader(open(ROOT / "v0/results" / f"{REF}.csv", encoding="utf-8-sig"))}
    new = {}
    for j, b, mk in res:
        if mk: new[j] = min(new.get(j, 1e18), mk)
    s0 = sum(ref.values()); s1 = sum(min(ref[j], new.get(j, 1e18)) for j in ref)
    won = sum(new.get(j, 1e18) < ref[j] - 1e-6 for j in ref)
    print(f"{SRC.name}: 기존 포트폴리오 {s0:.0f}s → PPO 설정 추가 시 {s1:.0f}s ({(s1/s0-1)*100:+.2f}%), PPO가 이긴 형상 {won}/{len(ref)}")
    print("PPO 단독(V8·균형V8 중 최소) 합계", round(sum(new.values())), "실패", [j for j in ref if j not in new])
