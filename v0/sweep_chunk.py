# -*- coding: utf-8 -*-
"""학습 형상 30개에서 조각 길이(chunk)를 훑어, 포트폴리오에 넣을 값을 고른다."""
import json, multiprocessing as mp, re, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
VPY = ROOT / "WAAM_Validator/.venv/Scripts/python.exe"
SRC = ROOT / "data" / "train_div"; WORK = ROOT / "tune_work"
CH = [40, 60, 80, 120, 160, 220, 320]

def run(a):
    j, c, bal = a
    out = WORK / "sweep" / f"c{c}" / ("bal" if bal else "nb") / j
    p = subprocess.run([str(VPY), str(ROOT / "v0/schedule_v8.py"), str(SRC / j), str(out),
                        "--paths", str(WORK / "cache" / f"{j}.pkl"), "--chunk", str(c),
                        *([] if bal else ["--no-balance"])], capture_output=True, text=True,
                       encoding="utf-8", cwd=ROOT / "WAAM_Validator")
    m = re.search(r"makespan=([\d.]+)s", p.stdout)
    return j, c, bal, (float(m.group(1)) if m else None)

if __name__ == "__main__":
    base = json.load(open(WORK / "base.json"))
    jobs = sorted(base)
    t0 = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.map(run, [(j, c, b) for j in jobs for c in CH for b in (False, True)], chunksize=1)
    best = {}
    for j, c, b, mk in res:
        if mk: best[(j, c)] = min(best.get((j, c), 1e18), mk)
    ref = {j: min(base[j]["v7"][0], base[j]["v8"][0], base[j]["v8bal"][0]) for j in jobs}
    print("조각길이별 (단독 사용 시 기본 포트폴리오 대비 평균비 / 그 길이가 최선인 형상 수)")
    for c in CH:
        r = [best[(j, c)] / ref[j] for j in jobs if (j, c) in best]
        win = sum(1 for j in jobs if min((best[(j, cc)] for cc in CH if (j, cc) in best), default=9e18) == best.get((j, c), 9e18))
        print(f"  {c:>4}mm  평균비 {np.mean(r):.4f}  최선 {win}개")
    # 조합: 기본 포트폴리오 + 길이 1~2개를 설정으로 추가했을 때
    def add(cs):
        tot = sum(min(ref[j], *[best[(j, c)] for c in cs if (j, c) in best]) for j in jobs)
        return tot / sum(ref[j] for j in jobs)
    print("설정으로 추가했을 때 합계비:")
    for c in CH:
        print(f"  +{c}mm → {add([c]):.4f}")
    pairs = [(a, b) for i, a in enumerate(CH) for b in CH[i + 1:]]
    top = sorted(pairs, key=lambda p: add(list(p)))[:5]
    for p in top:
        print(f"  +{p[0]}·{p[1]}mm → {add(list(p)):.4f}")
    print(f"{time.time()-t0:.0f}s")
