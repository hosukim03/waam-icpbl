# -*- coding: utf-8 -*-
"""PPO 재학습 — 지금 스케줄러(나선 경로·주차 생략·조각 잇기·여유 5mm·후보 25) 위에서 후보 선택을 배운다.

예전 학습(train_ppo.py)은 이 개선들이 들어가기 전 스케줄러로 해서 '고를 여지가 없다'는 결론이었다.
그런데 후보 수를 6→25 로 늘리자 04 가 −13% 가 되어, 팔이 엉키는 형상에서는 어떤 후보를 고르느냐가
실제로 중요하다는 게 드러났다. 그래서 가장 자주 이기는 두 설정(r2c160m5·r3c160m5) 위에서 다시 학습한다.

  1) 기준선: 정책 없이(=그리디) 학습 형상 30개 × 두 설정 makespan·시간 → tune_work/ppo2/base.json
  2) PPO: 형상별 SAMPLES 번 확률적으로 돌려 층 단위 보상(층 소요시간, 같은 층 묶음 평균 기준선)으로 갱신
  3) 반복마다 정책 저장, EVAL_EVERY 마다 결정적(argmax) 평가

usage: python train_ppo2.py [--iters 30]      (WAAM venv 파이썬, WAAM_Validator 폴더에서)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ppo_policy import Policy, logp_and_grad  # noqa: E402
from train_ppo import Adam  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "train_div"
CACHE = ROOT / "tune_work" / "cache2"
OUT = ROOT / "tune_work" / "ppo2"
SETTINGS = {                       # portfolio 의 r2c160m5 / r3c160m5 와 같은 인자
    "r2": dict(robots=2, balance=True),
    "r3": dict(robots=3, balance=False),
}
JOBS_PER_IT, SAMPLES, EPOCHS, CLIP, LR = 8, 4, 3, 0.2, 0.003
MAX_BASE_S = 40.0                  # 학습에는 한 번 도는 데 이 시간 이하인 형상만(반복 속도)


def opts_for(job, st):
    s = SETTINGS[st]
    return SimpleNamespace(arc=True, chunk=160.0, band=60.0, lookahead=25, balance=s["balance"],
                           robots=s["robots"], margin=5.0, margin_slow=None, max_tries=25, retract=500.0,
                           park_end=False, link_gap=9.0, order2=False, sweep=False, pair=None,
                           band_dir="out", overhead_w=0.0, rollout=None, bound_file=None, debug=False,
                           paths=str(CACHE / f"{job}_ring.pkl"))


def episode(args):
    job, st, ps, seed = args
    import schedule_v8
    pol = None
    if ps is not None:
        pol = Policy(hidden=len(ps[1]))
        pol.set(ps)
        if seed is not None:
            pol.rng = np.random.default_rng(seed)
            pol.trace = []
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, mk, _, _, _ = schedule_v8.run(SRC / job, opts_for(job, st), pol)
    except Exception as e:
        return job, st, seed, None, None, time.time() - t0, str(e)[:100]
    tr = None
    if pol is not None and pol.trace is not None:
        tr = (pol.trace, {k: pol.layer_t1[k] - pol.layer_t0[k] for k in pol.layer_t1})
    return job, st, seed, mk, tr, time.time() - t0, ""


def evaluate(pool, ps, tasks, base):
    res = pool.map(episode, [(j, st, ps, None) for j, st in tasks], chunksize=1)
    rat = [(mk if mk else 2 * base[j][st][0]) / base[j][st][0] for j, st, _, mk, _, _, _ in res]
    return float(np.mean(rat)), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = sorted(d.name for d in SRC.iterdir() if (d / "target.stl").exists())
    with mp.Pool(mp.cpu_count()) as pool:
        bp = OUT / "base.json"
        if bp.exists():
            base = json.load(open(bp))
        else:
            t0 = time.time()
            res = pool.map(episode, [(j, st, None, None) for j in jobs for st in SETTINGS], chunksize=1)
            base = {}
            for j, st, _, mk, _, dt, err in res:
                base.setdefault(j, {})[st] = (mk, round(dt, 1))
            json.dump(base, open(bp, "w"), indent=1)
            print(f"기준선 {time.time() - t0:.0f}s", flush=True)
        tasks = [(j, st) for j in jobs for st in SETTINGS
                 if base[j][st][0] and base[j][st][1] <= MAX_BASE_S]
        print(f"학습 과제 {len(tasks)}개 (형상×설정)", flush=True)
        pol = Policy(hidden=16, seed=0)
        ps = pol.get()
        opt = Adam(ps, LR)
        rng = np.random.default_rng(0)
        e0, _ = evaluate(pool, ps, tasks, base)
        print(f"반복 0 (그리디와 같은 시작점): 결정적 평가 makespan비 {e0:.4f}", flush=True)
        log = [{"it": 0, "eval_ratio": e0}]
        best = (e0, 0)
        for it in range(1, a.iters + 1):
            t0 = time.time()
            pick = rng.choice(len(tasks), size=min(JOBS_PER_IT, len(tasks)), replace=False)
            run_tasks = [(tasks[p][0], tasks[p][1], ps, int(rng.integers(1 << 30)))
                         for p in pick for _ in range(SAMPLES)]
            res = pool.map(episode, run_tasks, chunksize=1)
            groups, ratios = {}, []
            for j, st, sd, mk, tr, dt, err in res:
                groups.setdefault((j, st), []).append((mk, tr))
                ratios.append((mk if mk else 2 * base[j][st][0]) / base[j][st][0])
            data = []
            for (j, st), eps in groups.items():
                ok = [tr for _, tr in eps if tr]
                if len(ok) < 2:
                    continue
                for k in set.intersection(*[set(lt) for _, lt in ok]):
                    d = np.array([lt[k] for _, lt in ok])
                    if d.std() < 1e-6:
                        continue
                    A = -(d - d.mean()) / (d.std() + 1e-8)
                    for (tr, _), adv in zip(ok, A):
                        step = [t for t in tr if t[3] == k]
                        for X, act, lp, _k in step:
                            data.append((X, act, lp, adv / max(len(step), 1)))
            nclip = 0
            for _ in range(EPOCHS):
                G = [np.zeros_like(p, dtype=float) for p in ps]
                nclip = 0
                for X, act, lp_old, adv in data:
                    lp, g = logp_and_grad(ps, X.astype(float), act)
                    r = np.exp(lp - lp_old)
                    if (adv > 0 and r > 1 + CLIP) or (adv < 0 and r < 1 - CLIP):
                        nclip += 1
                        continue
                    for q in range(4):
                        G[q] += adv * r * g[q]
                ps = opt.step(ps, G)
            pol.set(ps)
            pol.save(OUT / f"policy_it{it:02d}.npz")
            row = {"it": it, "sampled_ratio": float(np.mean(ratios)), "decisions": len(data),
                   "clipped": nclip, "sec": round(time.time() - t0, 1)}
            if it % a.eval_every == 0 or it == a.iters:
                row["eval_ratio"], _ = evaluate(pool, ps, tasks, base)
                if row["eval_ratio"] < best[0]:
                    best = (row["eval_ratio"], it)
                    pol.save(OUT / "policy_best.npz")
            log.append(row)
            json.dump(log, open(OUT / "log.json", "w"), indent=1)
            print(f"반복 {it}: 샘플 평균 {row['sampled_ratio']:.4f} 결정 {len(data)} 클립 {nclip} {row['sec']}s"
                  + (f" | 결정적 평가 {row['eval_ratio']:.4f}" if "eval_ratio" in row else ""), flush=True)
    print(f"최선: 반복 {best[1]} 결정적 평가 {best[0]:.4f}", flush=True)
    print("PPO2_DONE", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
