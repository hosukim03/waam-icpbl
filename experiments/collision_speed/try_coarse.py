# -*- coding: utf-8 -*-
"""성긴 검사 간격(COARSE)을 바꿔 가며 solve.py 를 돌려 계산 시간을 견준다(0 = 모든 표본을 잼).

usage: python try_coarse.py <src_dir> <job_dir> <out_root> <COARSE> [<COARSE> ...]   (WAAM venv 파이썬)
"""
import sys
from pathlib import Path

src, job, out_root, *vals = sys.argv[1:]
sys.path.insert(0, str(Path(src).resolve()))
import psutil  # noqa: E402

psutil.Process().cpu_affinity([3])
import schedule_v8 as s8  # noqa: E402
import solve  # noqa: E402

for v in vals:
    s8.COARSE = int(v)
    sys.argv = ["solve.py", job, str(Path(out_root) / f"c{v}")]
    print(f"COARSE={v:>3}: ", end="", flush=True)
    solve.main()
