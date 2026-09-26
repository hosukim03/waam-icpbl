# -*- coding: utf-8 -*-
"""PPO 로 V8 후보 선택 정책을 학습한다 (학습 형상 train_div, 경로 캐시 tune_work/cache 사용).

한 반복(iteration):
  1) 학습 형상 JOBS_PER_IT 개를 뽑아, 형상마다 정책을 SAMPLES 번 확률적으로 돌린다(코어 병렬).
  2) 보상 = -층 소요시간(층 끝 − 층 시작). 같은 형상·같은 층의 SAMPLES 번 평균을 기준선으로 쓴다.
     형상 끝 makespan 하나만 쓰면 수천 개 판단 중 무엇이 좋았는지 구분되지 않아(신용 할당 문제)
     층 단위로 신호를 촘촘히 준다. 층 소요시간의 합이 곧 makespan 이므로 목표는 그대로다.
  3) PPO 클리핑 목적  min(r·A, clip(r, 1-ε, 1+ε)·A),  r = π_new(a|s) / π_old(a|s)  로 EPOCHS 번 갱신(Adam).
매 반복 정책을 tune_work/ppo/policy_itNN.npz 로 저장하고, EVAL_EVERY 마다 결정적(argmax) 성능을 잰다.

usage: python train_ppo.py [--iters 30]      (WAAM venv 파이썬, WAAM_Validator 폴더에서)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ppo_policy import Policy, logp_and_grad  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "train_div"
WORK = ROOT / "tune_work"
OUT = WORK / "ppo"
JOBS_PER_IT, SAMPLES, EPOCHS, CLIP, LR = 8, 4, 4, 0.2, 0.01
MAX_BASE_S = 30.0          # 학습에는 V8 1회가 이 시간 이하인 형상만 사용(반복 속도)


def opts_for(job, bal):
    return SimpleNamespace(arc=True, chunk=40.0, band=60.0, lookahead=6, balance=bal, max_tries=25,
                           retract=500.0, debug=False, paths=str(WORK / "cache" / f"{job}.pkl"))


def episode(args):
    job, bal, ps, seed = args
    import io, contextlib
    import schedule_v8
    pol = Policy(hidden=len(ps[1]))
    pol.set(ps)
    if seed is not None:
        pol.rng = np.random.default_rng(seed)
        pol.trace = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, mk, _, _, _ = schedule_v8.run(SRC / job, opts_for(job, bal), pol)
    except Exception as e:                       # 교착 등 → 큰 벌점
        return job, bal, seed, None, None, str(e)[:100]
    lt = {k: pol.layer_t1[k] - pol.layer_t0[k] for k in pol.layer_t1}
    return job, bal, seed, mk, (pol.trace, lt) if pol.trace is not None else None, ""


class Adam:
    def __init__(self, ps, lr):
        self.m = [np.zeros_like(p, dtype=float) for p in ps]
        self.v = [np.zeros_like(p, dtype=float) for p in ps]
        self.t, self.lr = 0, lr

    def step(self, ps, gs):                      # 경사 상승
        self.t += 1
        out = []
        for k, (p, g) in enumerate(zip(ps, gs)):
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            out.append(p + self.lr * mh / (np.sqrt(vh) + 1e-8))
        return out


def evaluate(pool, ps, jobs, base):
    res = pool.map(episode, [(j, b, ps, None) for j in jobs for b in (False, True)], chunksize=1)
    rat = []
    for j, b, _, mk, _, _ in res:
        ref = base[j]["v8bal" if b else "v8"][0]
        rat.append((mk if mk else 2 * ref) / ref)
    return float(np.mean(rat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.load(open(WORK / "base.json"))
    jobs = [j for j in sorted(base) if max(base[j]["v8"][1], base[j]["v8bal"][1]) <= MAX_BASE_S]
    print(f"학습 형상 {len(jobs)}개: {jobs}", flush=True)
    pol = Policy(hidden=16, seed=0)
    ps = pol.get()
    opt = Adam(ps, LR)
    rng = np.random.default_rng(0)
    log = []
    with mp.Pool(mp.cpu_count()) as pool:
        e0 = evaluate(pool, ps, jobs, base)
        print(f"반복 0 (V8 과 같은 시작점): 결정적 평가 makespan비 {e0:.4f}", flush=True)
        for it in range(1, a.iters + 1):
            t0 = time.time()
            pick = rng.choice(len(jobs), size=min(JOBS_PER_IT, len(jobs)), replace=False)
            tasks = [(jobs[p], bool(rng.random() < 0.5), ps, int(rng.integers(1 << 30)))
                     for p in pick for _ in range(SAMPLES)]
            res = pool.map(episode, tasks, chunksize=1)
            # 층별 기준선 → advantage
            groups = {}
            for j, b, sd, mk, tr, err in res:
                groups.setdefault((j, b), []).append((mk, tr))
            data, ratios = [], []
            for (j, b), eps in groups.items():
                ref = base[j]["v8bal" if b else "v8"][0]
                ratios += [(mk if mk else 2 * ref) / ref for mk, _ in eps]
                ok = [tr for _, tr in eps if tr]
                if len(ok) < 2:
                    continue
                for k in set.intersection(*[set(lt) for _, lt in ok]):   # 두 실행 모두에 있는 층
                    d = np.array([lt[k] for _, lt in ok])
                    if d.std() < 1e-6:
                        continue
                    A = -(d - d.mean()) / (d.std() + 1e-8)      # 그 층을 짧게 끝낸 실행의 선택을 늘린다
                    for (tr, _), adv in zip(ok, A):
                        step = [t for t in tr if t[3] == k]
                        for X, act, lp, _k in step:
                            data.append((X, act, lp, adv / max(len(step), 1)))
            # PPO 갱신
            for _ in range(EPOCHS):
                G = [np.zeros_like(p, dtype=float) for p in ps]
                nclip = 0
                for X, act, lp_old, adv in data:
                    lp, g = logp_and_grad(ps, X.astype(float), act)
                    r = np.exp(lp - lp_old)
                    if (adv > 0 and r > 1 + CLIP) or (adv < 0 and r < 1 - CLIP):
                        nclip += 1
                        continue                         # 클리핑: 이미 충분히 바뀐 선택은 더 밀지 않음
                    for k in range(4):
                        G[k] += adv * r * g[k]
                ps = opt.step(ps, G)
            pol.set(ps)
            pol.save(OUT / f"policy_it{it:02d}.npz")
            row = {"it": it, "mean_ratio_sampled": float(np.mean(ratios)), "decisions": len(data),
                   "clipped_last_epoch": nclip, "sec": round(time.time() - t0, 1)}
            if it % a.eval_every == 0 or it == a.iters:
                row["eval_ratio"] = evaluate(pool, ps, jobs, base)
            log.append(row)
            json.dump(log, open(OUT / "log.json", "w"), indent=1)
            print(f"반복 {it}: 샘플 평균 makespan비 {row['mean_ratio_sampled']:.4f} 결정 {len(data)}개 "
                  f"클립 {nclip} {row['sec']}s" + (f" | 결정적 평가 {row['eval_ratio']:.4f}" if "eval_ratio" in row else ""),
                  flush=True)
    print("PPO_DONE", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
