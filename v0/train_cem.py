# -*- coding: utf-8 -*-
"""CEM(교차 엔트로피 방법)으로 배차 정책 신경망을 학습한다.

REINFORCE 는 매 결정마다 확률적으로 뽑기 때문에 아크 재점화 손실이 누적되어 실패했다.
CEM 은 '실제로 배포할 결정적(argmax) 정책' 자체를 평가하므로 그 문제가 없다.

  1) 가중치 후보를 정규분포에서 N개 뽑는다
  2) 각 후보로 학습 형상들을 스케줄링해 그리디 대비 개선율을 잰다
  3) 상위 K개(엘리트)의 평균·표준편차로 분포를 갱신한다

usage: python train_cem.py <학습폴더> [--pop 12] [--elite 4] [--gens 12] [--layers 2]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from schedule_rl import NFEAT, Policy, parse, run

sys.stdout.reconfigure(encoding="utf-8")


def unpack(vec, hidden):
    p = Policy(hidden=hidden)
    n1 = hidden * NFEAT
    p.W1 = vec[:n1].reshape(hidden, NFEAT)
    p.b1 = vec[n1:n1 + hidden]
    p.w2 = vec[n1 + hidden:n1 + 2 * hidden]
    p.b2 = float(vec[-1])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir")
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2, help="학습 시 형상당 층 수 제한")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rl_policy.npz")
    ap.add_argument("--init", default=None, help="이어서 학습할 가중치 파일")
    a = ap.parse_args()

    jobs = sorted(p for p in Path(a.train_dir).iterdir() if (p / "target.stl").exists())
    opts = parse(["x", "y", "--max-layers", str(a.layers)])
    rng = np.random.default_rng(a.seed)

    print("기준선(그리디) 계산 중...")
    base = {}
    for j in jobs:
        _, mk, _, _ = run(j, opts, policy=None)
        base[j.name] = mk
    print("  " + ", ".join(f"{k}:{v:.0f}s" for k, v in base.items()))

    dim = a.hidden * NFEAT + a.hidden + a.hidden + 1
    mu, sigma = np.zeros(dim), np.full(dim, a.sigma)
    if a.init:                                          # 저장된 정책에서 이어서 학습
        d = np.load(a.init)
        mu = np.concatenate([d["W1"].ravel(), d["b1"], d["w2"], [float(d["b2"])]])
        print(f"이어서 학습: {a.init}")
    best_score, t0 = -1e9, time.time()

    def score_of(vec):
        pol = unpack(vec, a.hidden)
        rs = []
        for j in jobs:
            _, mk, _, _ = run(j, opts, policy=pol)          # argmax (rng 없음)
            rs.append((base[j.name] - mk) / base[j.name])
        return float(np.mean(rs)), rs

    s0, _ = score_of(np.zeros(dim))
    print(f"가중치 0 (=그리디와 동일) 확인: 개선율 {s0:+.4%}")

    for g in range(a.gens):
        pops = rng.normal(mu, sigma, (a.pop, dim))
        pops[0] = mu                                        # 현재 평균도 함께 평가
        scores = []
        for v in pops:
            sc, _ = score_of(v)
            scores.append(sc)
        scores = np.array(scores)
        elite = pops[np.argsort(scores)[-a.elite:]]
        mu, sigma = elite.mean(axis=0), elite.std(axis=0) + 0.02
        if scores.max() > best_score:
            best_score = float(scores.max())
            unpack(pops[int(np.argmax(scores))], a.hidden).save(a.out)
        print(f"[{g + 1:2d}/{a.gens}] 최고 {scores.max():+.3%}  평균 {scores.mean():+.3%}  "
              f"누적최고 {best_score:+.3%}  ({time.time() - t0:.0f}s)")
    print(f"학습 끝. 최고 개선율 {best_score:+.3%} → {a.out}, 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
