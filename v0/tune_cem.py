# -*- coding: utf-8 -*-
"""CEM(교차 엔트로피법)으로 V8 설정값을 학습 형상(train_div 30개)에서 찾는다.

목적 = 형상별 (min(V7 기본, V8(θ), 균형V8(θ)) / 기본 포트폴리오 makespan) 의 평균
       + 0.01 × max(0, 생성시간 비 − 1)   (느려지는 설정에 작은 벌점)
θ = chunk, band, lookahead, max_tries, retract.  안전 여유(MARGIN)는 건드리지 않는다.
세대마다 tune_work/cem_log.json 에 저장하고, 다시 실행하면 이어서 한다.
"""
import json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
from tune_base import sched, WORK, SRC

PARAMS = [  # 이름, 하한, 상한, 기본, 정수 여부
    ("chunk", 20.0, 80.0, 40.0, False),
    ("band", 30.0, 120.0, 60.0, False),
    ("lookahead", 2, 12, 6, True),
    ("max_tries", 10, 40, 25, True),
    ("retract", 300.0, 800.0, 500.0, False),
]
POP, ELITE, GENS, SIG0 = 10, 3, 5, 0.25


def decode(u):
    th = {}
    for (n, lo, hi, _, isint), x in zip(PARAMS, np.clip(u, 0, 1)):
        v = lo + x * (hi - lo)
        th[n] = int(round(v)) if isint else round(float(v), 1)
    return th


def encode(th):
    return np.array([(th[n] - lo) / (hi - lo) for n, lo, hi, _, _ in PARAMS])


def args_of(th):
    return ["--chunk", str(th["chunk"]), "--band", str(th["band"]), "--lookahead", str(th["lookahead"]),
            "--max-tries", str(th["max_tries"]), "--retract", str(th["retract"])]


def score(th, runs, base, jobs):
    rat, gt, gb = [], 0.0, 0.0
    for j in jobs:
        b = base[j]
        pf0 = min(b["v7"][0], b["v8"][0], b["v8bal"][0])
        cand = [b["v7"][0]] + [runs[(j, v)][0] for v in ("v8", "v8bal") if runs[(j, v)][0]]
        rat.append(min(cand) / pf0)
        gt += max(b["v7"][1], *(runs[(j, v)][1] for v in ("v8", "v8bal")))
        gb += max(b["v7"][1], b["v8"][1], b["v8bal"][1])
    r = float(np.mean(rat))
    return r + 0.01 * max(0.0, gt / gb - 1), r, gt / gb


def main():
    jobs = sorted(d.name for d in SRC.iterdir() if (d / "target.stl").exists())
    base = json.load(open(WORK / "base.json"))
    logp = WORK / "cem_log.json"
    log = json.load(open(logp)) if logp.exists() else {"gens": []}
    rng = np.random.default_rng(7 + len(log["gens"]))
    if log["gens"]:
        mu, sig = np.array(log["gens"][-1]["mu"]), np.array(log["gens"][-1]["sig"])
    else:
        mu = encode({n: d for n, _, _, d, _ in PARAMS}); sig = np.full(len(PARAMS), SIG0)
    for g in range(len(log["gens"]), GENS):
        t0 = time.time()
        U = [mu] + [np.clip(mu + sig * rng.standard_normal(len(mu)), 0, 1) for _ in range(POP - 1)]
        TH = [decode(u) for u in U]
        tasks = [(j, v, args_of(th), f"g{g}_{k}") for k, th in enumerate(TH) for j in jobs for v in ("v8", "v8bal")]
        tasks.sort(key=lambda t: -base[t[0]][t[1]][1])          # 오래 걸리는 것부터(코어 놀림 방지)
        with mp.Pool(mp.cpu_count()) as pool:
            res = pool.map(sched, tasks, chunksize=1)
        by = {}
        for (j, v, _, tag), (_, _, mk, dt) in zip(tasks, res):
            by.setdefault(tag, {})[(j, v)] = (mk, dt)
        rows = []
        for k, th in enumerate(TH):
            f, r, gr = score(th, by[f"g{g}_{k}"], base, jobs)
            rows.append({"theta": th, "obj": f, "ratio": r, "gen_ratio": gr})
        order = np.argsort([x["obj"] for x in rows])
        E = np.array([encode(TH[i]) for i in order[:ELITE]])
        mu = E.mean(0); sig = np.maximum(E.std(0), 0.03)
        log["gens"].append({"rows": rows, "mu": mu.tolist(), "sig": sig.tolist(), "sec": time.time() - t0})
        json.dump(log, open(logp, "w"), indent=1, ensure_ascii=False)
        b = rows[order[0]]
        print(f"세대 {g}: 최선 obj={b['obj']:.4f} (makespan비 {b['ratio']:.4f}, 생성시간비 {b['gen_ratio']:.2f}) "
              f"{b['theta']} | 기본값 obj={rows[0]['obj']:.4f} | {time.time()-t0:.0f}s", flush=True)
        import shutil; shutil.rmtree(WORK / "runs", ignore_errors=True)   # 궤적 파일은 용량만 차지
    print("CEM_DONE", flush=True)


if __name__ == "__main__":
    main()
