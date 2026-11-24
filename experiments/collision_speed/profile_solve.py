# -*- coding: utf-8 -*-
"""solve.py 한 번을 cProfile 로 돌려, 계산 시간 중 충돌 검사(Checker._check)가 차지하는 몫을 잰다.

usage: python profile_solve.py <src_dir> <job_dir> <out_dir> <stats_file>   (WAAM venv 파이썬)
"""
import cProfile
import pstats
import sys
import time
from pathlib import Path

src, job, out, stats = sys.argv[1:5]
sys.path.insert(0, str(Path(src).resolve()))
import solve  # noqa: E402

sys.argv = ["solve.py", job, out]
t0 = time.perf_counter()
pr = cProfile.Profile()
pr.runcall(solve.main)
wall = time.perf_counter() - t0
pr.dump_stats(stats)

st = pstats.Stats(pr)
want = {"_check": "충돌 검사 전체(_check)", "seg_dist": "  그중 팔 거리 계산(seg_dist)", "xy": "  그중 남의 위치 찾기(Timeline.xy)",
        "earliest": "가장 이른 시작 찾기(earliest)", "build_paths": "경로 만들기(build_paths)", "run": "계획(schedule_v8.run)"}
rows = {}
for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
    if name in want and ("schedule_v8" in fn or "solve" in fn):
        rows[name] = (ct, nc)
print(f"프로파일 켠 채 전체 {wall:.1f}s")
for k, label in want.items():
    if k in rows:
        ct, nc = rows[k]
        print(f"{label:34s} {ct:8.1f}s  ({ct / wall * 100:5.1f}%)  호출 {nc:,}회")
