# -*- coding: utf-8 -*-
"""V3 결과의 남은 개선 여지 분석.
각 로봇의 시간을 D(적층)/T(이동)/W(대기·아크)로 나누고,
하한 = Σ_층 (그 층 적층 시간 합 / 3)  ← 로봇 3대가 대기·이동 없이 완벽히 나눠 쓴다고 가정
효율 = 하한 / makespan (1에 가까울수록 더 줄일 여지가 적음)"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
LAYER_H = 2.0


def analyze(path):
    rows = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[int(r["robot_id"])].append((float(r["time_s"]), float(r["z_mm"]), r["mode"]))
    tot = defaultdict(float)
    layer_d = defaultdict(float)
    makespan = 0.0
    for rid, rs in rows.items():
        rs.sort()
        makespan = max(makespan, rs[-1][0])
        for (t0, z0, m), (t1, z1, _) in zip(rs, rs[1:]):
            dt = t1 - t0
            tot[(rid, m)] += dt
            if m == "D":
                layer_d[int(z0 // LAYER_H)] += dt
    D = sum(v for (r, m), v in tot.items() if m == "D")
    T = sum(v for (r, m), v in tot.items() if m == "T")
    W = sum(v for (r, m), v in tot.items() if m == "W")
    # 대기(W)에는 아크 on/off 1s 가 섞여 있음
    lb = sum(v / 3 for v in layer_d.values())
    busy = D / (3 * makespan)
    return makespan, D, T, W, lb, busy


if __name__ == "__main__":
    print(f"{'job':6s} {'makespan':>9s} {'하한':>8s} {'효율':>6s} {'로봇가동률(D)':>12s} {'D':>8s} {'T':>7s} {'W':>8s}")
    for group, base in (("public", ROOT / "jobs_v3"), ("bench", ROOT / "bench" / "jobs_v3")):
        for d in sorted(base.iterdir()):
            p = d / "trajectory.csv"
            if not p.exists():
                continue
            ms, D, T, W, lb, busy = analyze(p)
            print(f"{group[0]}{d.name:5s} {ms / 3600:8.2f}h {lb / 3600:7.2f}h {lb / ms:6.0%} {busy:11.0%}"
                  f" {D / 3600:7.2f}h {T / 3600:6.2f}h {W / 3600:7.2f}h")
