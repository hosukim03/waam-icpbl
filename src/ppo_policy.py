# -*- coding: utf-8 -*-
"""V8 후보 선택 정책 (PPO 로 학습).

V8 은 매 순간 충돌 검사를 통과한 후보들(로봇 i 가 큐의 qi 번째 조각을 시각 s 에 시작) 중
'가장 빨리 시작하는 것'을 고른다. 정책은 이 그리디 점수를 사전 점수로 깔고, 작은 신경망이
보정값만 더한다. 보정이 0 이면 그리디와 같은 선택이므로 학습 시작점이 곧 V8 이다.

score = -PRIOR_WAIT·대기(분) - PRIOR_QUEUE·큐위치 + w2·tanh(W1 x + b1) + b2
"""
from __future__ import annotations

import numpy as np

NFEAT = 10


def features(cands, robots, queues, loads, vd):
    now = min(rb.t for rb in robots)
    rt = np.array([rb.t for rb in robots])
    tcp = np.array([rb.pos[:2] for rb in robots])
    X = np.empty((len(cands), NFEAT))
    for r, (s, qi, i, kind) in enumerate(cands):
        ch = queues[i][qi][2]
        c = ch.mean(axis=0)
        others = [j for j in range(3) if j != i]
        X[r] = (
            min((s - now) / 60.0, 5.0),                                   # 시작까지 대기
            qi / 6.0,                                                     # 큐 위치
            1.0 if kind == "cont" else 0.0,                               # 아크를 이어가는가
            float(np.sum(np.hypot(*np.diff(ch, axis=0).T))) / vd / 30.0,  # 적층 시간
            min(float(np.hypot(*(tcp[i] - ch[0]))) / 500.0, 3.0),         # 접근 거리
            float(np.hypot(*c)) / 700.0,                                  # 조각이 가운데인지 바깥인지
            (rt[i] - now) / 60.0,                                         # 이 로봇이 얼마나 앞서 있나
            (loads[i] - loads.mean()) / 600.0,                            # 누적 부하 편차
            min(len(queues[i]) / 50.0, 3.0),                              # 남은 조각 수
            min(min(float(np.hypot(*(tcp[j] - c))) for j in others) / 500.0, 2.0),  # 다른 토치와 거리
        )
    return X


class Policy:
    TEMP = 0.3            # 학습 중 샘플링 온도
    PRIOR_WAIT = 8.0
    PRIOR_QUEUE = 0.5

    def __init__(self, hidden=16, seed=0, scale=0.1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, scale, (hidden, NFEAT))
        self.b1 = np.zeros(hidden)
        self.w2 = np.zeros(hidden)          # 보정 0 에서 시작 = V8 과 같은 선택
        self.b2 = 0.0
        self.rng = None                     # 설정하면 샘플링(학습), 없으면 argmax(실사용)
        self.trace = None
        self.layer = -1                     # 현재 층(층 단위 보상용)
        self.layer_t0, self.layer_t1 = {}, {}

    # ---- 저장/불러오기 ----
    def get(self):
        return [self.W1.copy(), self.b1.copy(), self.w2.copy(), np.array(self.b2)]

    def set(self, ps):
        self.W1, self.b1, self.w2, self.b2 = ps[0].copy(), ps[1].copy(), ps[2].copy(), float(ps[3])

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        p = cls(hidden=len(d["b1"]))
        p.set([d["W1"], d["b1"], d["w2"], d["b2"]])
        return p

    # ---- 계산 ----
    @staticmethod
    def logits(ps, X):
        W1, b1, w2, b2 = ps
        H = np.tanh(X @ W1.T + b1)
        prior = -Policy.PRIOR_WAIT * X[:, 0] - Policy.PRIOR_QUEUE * X[:, 1]
        return (prior + H @ w2 + float(b2)) / Policy.TEMP, H

    def pick(self, cands, robots, queues, loads, vd):
        X = features(cands, robots, queues, loads, vd)
        z, _ = self.logits(self.get(), X)
        if self.rng is None:
            return int(np.argmax(z))
        p = np.exp(z - z.max())
        p /= p.sum()
        a = int(self.rng.choice(len(p), p=p))
        if self.trace is not None:
            self.trace.append((X.astype(np.float32), a, float(np.log(p[a] + 1e-12)), self.layer))
        return a


def logp_and_grad(ps, X, a):
    """선택 a 의 log 확률과 파라미터 기울기."""
    W1, b1, w2, b2 = ps
    z, H = Policy.logits(ps, X)
    p = np.exp(z - z.max())
    p /= p.sum()
    d = -p
    d[a] += 1.0
    d /= Policy.TEMP                                   # dlogp/dscore
    gw2 = d @ H
    gb2 = d.sum()
    dH = np.outer(d, w2) * (1 - H ** 2)
    return float(np.log(p[a] + 1e-12)), [dH.T @ X, dH.sum(0), gw2, np.array(gb2)]
