# -*- coding: utf-8 -*-
"""REINFORCE 로 배차 정책(신경망)을 학습한다.

한 에피소드 = 학습용 형상 하나를 끝까지 스케줄링
보상 = (그리디 makespan − 이번 makespan) / 그리디 makespan     (형상 크기와 무관한 상대 개선율)
기울기 = Σ_t ∇logπ(a_t|s_t) × (보상 − 기준선)                  (기준선 = 형상별 이동평균)

usage: python train_rl.py <학습폴더> [--iters 60] [--hidden 8] [--lr 0.05] [--out rl_policy.npz]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from schedule_rl import Policy, parse, run

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rl_policy.npz")
    a = ap.parse_args()

    jobs = sorted(p for p in Path(a.train_dir).iterdir() if (p / "target.stl").exists())
    opts = parse(["x", "y"])                       # 스케줄러 기본 옵션(조각 40mm, 후보 6개)
    pol = Policy(hidden=a.hidden, seed=a.seed)
    rng = np.random.default_rng(a.seed)

    print("기준선(그리디) 계산 중...")
    base = {}
    for j in jobs:
        t0 = time.time()
        _, mk, _, _ = run(j, opts, policy=None)
        base[j.name] = mk
        print(f"  {j.name}: {mk:8.1f}s ({time.time() - t0:.1f}s)")

    ema = {j.name: 0.0 for j in jobs}              # 형상별 기준선(이동평균)
    best, t_start = -1e9, time.time()
    for it in range(a.iters):
        rewards = []
        for j in jobs:
            trace = []
            _, mk, _, _ = run(j, opts, policy=pol, rng=rng, trace=trace)
            r = (base[j.name] - mk) / base[j.name]
            adv = r - ema[j.name]
            ema[j.name] = 0.8 * ema[j.name] + 0.2 * r
            g = [np.zeros_like(pol.W1), np.zeros_like(pol.b1), np.zeros_like(pol.w2), 0.0]
            for X, H, p, idx in trace:
                gW1, gb1, gw2, gb2 = pol.grad_logp(X, H, p.copy(), idx)
                g[0] += gW1
                g[1] += gb1
                g[2] += gw2
                g[3] += gb2
            scale = a.lr * adv / max(len(trace), 1)
            pol.add(g, scale)
            rewards.append(r)
        m = float(np.mean(rewards))
        print(f"[{it + 1:3d}/{a.iters}] 평균 개선율 {m:+.3%}  최고 {max(rewards):+.2%}  "
              f"최저 {min(rewards):+.2%}  ({time.time() - t_start:.0f}s)")
        if m > best:
            best = m
            pol.save(a.out)
            print(f"      → 저장 {a.out}")
    print(f"학습 끝. 최고 평균 개선율 {best:+.3%}, 총 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
