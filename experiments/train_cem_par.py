# -*- coding: utf-8 -*-
"""CEM 학습 (병렬판). 후보×형상 평가를 여러 코어에 나눠 돌린다.

usage: python train_cem_par.py <학습폴더> [--pop 12] [--elite 4] [--gens 10] [--layers 2]
                               [--workers 10] [--init w.npz] [--out rl_policy.npz]
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_worker as W
from schedule_rl import NFEAT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir")
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    ap.add_argument("--init", default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="후보 1개 평가 시 결정 횟수 상한")
    ap.add_argument("--out", default="rl_policy.npz")
    a = ap.parse_args()

    names = [p.name for p in sorted(Path(a.train_dir).iterdir()) if (p / "target.stl").exists()]
    dim = a.hidden * NFEAT + 2 * a.hidden + 1
    rng = np.random.default_rng(a.seed)
    mu, sigma = np.zeros(dim), np.full(dim, a.sigma)
    if a.init:
        d = np.load(a.init)
        mu = np.concatenate([d["W1"].ravel(), d["b1"], d["w2"], [float(d["b2"])]])
        print(f"이어서 학습: {a.init}", flush=True)

    t0 = time.time()
    with mp.Pool(a.workers, initializer=W.init, initargs=(a.train_dir, a.layers, a.hidden, a.max_steps)) as pool:
        base = {}
        for _, name, mk in pool.imap_unordered(W.work, [(-1, n, None) for n in names]):
            base[name] = mk
        print(f"기준선(그리디) {len(base)}개 완료 ({time.time() - t0:.0f}s)", flush=True)

        best, best_vec = -1e9, mu.copy()
        for g in range(a.gens):
            pops = rng.normal(mu, sigma, (a.pop, dim))
            pops[0] = mu                                    # 현재 평균도 후보에 포함
            tasks = [(i, n, pops[i]) for i in range(a.pop) for n in names]
            got = {i: [] for i in range(a.pop)}
            for i, name, mk in pool.imap_unordered(W.work, tasks, chunksize=1):
                r = -1.0 if mk == float("inf") else (base[name] - mk) / base[name]
                got[i].append(r)
            scores = np.array([float(np.mean(got[i])) for i in range(a.pop)])
            order = np.argsort(scores)
            elite = pops[order[-a.elite:]]
            mu, sigma = elite.mean(axis=0), elite.std(axis=0) + 0.02
            if scores.max() > best:
                best = float(scores.max())
                best_vec = pops[int(np.argmax(scores))].copy()
                W.unpack(best_vec, a.hidden).save(a.out)
            print(f"[{g + 1:2d}/{a.gens}] 최고 {scores.max():+.3%}  평균 {scores.mean():+.3%}  "
                  f"누적최고 {best:+.3%}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"학습 끝. 최고 개선율 {best:+.3%} → {a.out}, 총 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
