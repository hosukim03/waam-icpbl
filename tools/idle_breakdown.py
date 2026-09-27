# -*- coding: utf-8 -*-
"""층 장벽(layer barrier) 때문에 생기는 대기가 얼마나 되는지 측정.

층마다: 로봇별 마지막 작업 종료 시각 vs 그 층의 종료 시각(세 로봇 중 가장 늦은 시각)
        차이 = 그 로봇이 다음 층을 기다리며 놀고 있는 시간
"""
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
    # 층별로 각 로봇의 마지막 적층 종료 시각
    end = defaultdict(dict)          # layer -> rid -> 마지막 D 종료
    for rid, rs in rows.items():
        rs.sort()
        for (t0, z0, m), (t1, _, _) in zip(rs, rs[1:]):
            if m == "D":
                k = int(z0 // LAYER_H)
                end[k][rid] = max(end[k].get(rid, 0.0), t1)
    idle = 0.0
    worst = []
    for k, e in end.items():
        fin = max(e.values())
        gap = sum(fin - v for v in e.values())
        idle += gap
        worst.append((gap, k, fin))
    makespan = max(rs[-1][0] for rs in rows.values())
    worst.sort(reverse=True)
    return makespan, idle, len(end), worst[:3]


if __name__ == "__main__":
    print(f"{'job':6s} {'makespan':>9s} {'층끝대기합':>10s} {'makespan대비':>11s}  가장 큰 층 (대기, 층번호)")
    for name in sorted((ROOT / "jobs_v3").iterdir()):
        p = name / "trajectory.csv"
        if not p.exists():
            continue
        ms, idle, nl, worst = analyze(p)
        top = ", ".join(f"층{k}:{g / 60:.0f}분" for g, k, _ in worst)
        print(f"{name.name:6s} {ms / 3600:8.2f}h {idle / 3600:9.2f}h {idle / (3 * ms):10.0%}  {top}")
