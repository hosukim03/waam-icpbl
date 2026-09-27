# -*- coding: utf-8 -*-
"""한 형상의 궤적을 뜯어 병목을 본다: 모드별 시간, 층별 소요, 로봇 편차, 하한 대비 여유."""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

job = Path(sys.argv[1])
rows = defaultdict(list)
with open(job / "trajectory.csv", newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        rows[int(r["robot_id"])].append((float(r["time_s"]), float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"]), r["mode"]))
mk = max(v[-1][0] for v in rows.values())
tot = defaultdict(float); dep_len = 0.0
zs = set()
for rid, v in rows.items():
    for (t0, x0, y0, z0, m), (t1, *_ ) in zip(v, v[1:]):
        tot[(rid, m)] += t1 - t0
    for (t0, x0, y0, z0, m), (t1, x1, y1, z1, _) in zip(v, v[1:]):
        if m == "D":
            dep_len += float(np.hypot(x1 - x0, y1 - y0))
        zs.add(round(z0, 3))
print(f"{job.name}: makespan {mk:.0f}s ({mk/3600:.2f}h)")
print(f"{'로봇':>4} {'D 적층':>9} {'T 이동':>9} {'W 대기':>9} {'적층 비율':>8}")
for rid in sorted(rows):
    d, t, w = tot[(rid, 'D')], tot[(rid, 'T')], tot[(rid, 'W')]
    print(f"{rid:>4} {d:9.0f} {t:9.0f} {w:9.0f} {d/mk*100:7.1f}%")
D = sum(tot[(r, 'D')] for r in rows)
print(f"적층 시간 합계 {D:.0f}s → 3대 완전 분할 하한 {D/3:.0f}s ({D/3/mk*100:.1f}% of makespan)")
print(f"이동 합계 {sum(tot[(r,'T')] for r in rows):.0f}s, 대기·아크 합계 {sum(tot[(r,'W')] for r in rows):.0f}s")

# 층 경계: 모든 로봇이 z 를 올리는 시각으로 층을 나눈다(적층 z 값 기준)
dz = sorted({round(z, 3) for v in rows.values() for (_, _, _, z, m) in v if m == "D"})
print(f"적층 층 수 {len(dz)}")
lay = []
for z in dz:
    ts = [t for v in rows.values() for (t, _, _, zz, m) in v if m == "D" and round(zz, 3) == z]
    lay.append((z, min(ts), max(ts)))
span = [(b - a) for _, a, b in lay]
print(f"층 소요(적층 첫~끝) 평균 {np.mean(span):.0f}s 최대 {np.max(span):.0f}s(층 {int(np.argmax(span))+1}) 합계 {np.sum(span):.0f}s")
print(f"층 사이 전환(주차·상승 등) 합계 {mk - np.sum(span):.0f}s ({(mk-np.sum(span))/mk*100:.1f}%)")
w = [max(tot[(r, 'W')] for r in rows), min(tot[(r, 'W')] for r in rows)]
print(f"로봇별 적층시간 편차: {[round(tot[(r,'D')]) for r in sorted(rows)]} (최대-최소 {max(tot[(r,'D')] for r in rows)-min(tot[(r,'D')] for r in rows):.0f}s)")
