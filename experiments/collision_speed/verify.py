# -*- coding: utf-8 -*-
"""성긴 충돌 검사가 모든 표본을 재는 검사와 같은 답을 내는지, 검사할 때마다 둘을 견주며 solve.py 를 돌린다.
다르면 AssertionError 로 멈춘다(schedule_v8.CHECK_VERIFY).

usage: python verify.py <src_dir> <job_dir> <out_dir> [solve.py 옵션 ...]   (WAAM venv 파이썬)
"""
import sys
import time
from pathlib import Path

src, job, out, *rest = sys.argv[1:]
sys.path.insert(0, str(Path(src).resolve()))
import schedule_v8 as s8  # noqa: E402
import solve  # noqa: E402

s8.CHECK_VERIFY = True
calls = {"n": 0}
orig = s8.Checker._check_all


def counted(self, *a):
    calls["n"] += 1
    return orig(self, *a)


s8.Checker._check_all = counted
sys.argv = ["solve.py", job, out, *rest]
t0 = time.perf_counter()
solve.main()
print(f"검사 {calls['n']:,}번 모두 같음 ({time.perf_counter() - t0:.1f}s)")
