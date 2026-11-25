# -*- coding: utf-8 -*-
"""solve.py 를 그대로 돌리면서 계획 한 번(짧은 시험·끝까지 계획·국소 탐색)마다 걸린 시간과 충돌 검사 횟수를
적는다(계산량 한도 시험, WAAM venv 파이썬, 코어 1개로 하나씩 돌릴 것).

usage: python experiments/ls_budget/measure_solve.py <log.csv> <묶음 폴더> <형상> <출력 폴더> [solve.py 옵션 ...]
"""
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "WAAM_Validator" / "src"))

import schedule_v8 as s8  # noqa: E402
import solve  # noqa: E402

CNT = {"checks": 0}
LOG = []
_orig_check = s8.Checker._check
_orig_run = s8.run


def _check(self, i, rel_ts, rel_xy, starts):
    CNT["checks"] += 1
    return _orig_check(self, i, rel_ts, rel_xy, starts)


def _run(job, opts, policy=None, pre=None, path_cache=None, only=None, bound=None):
    c0, t0 = CNT["checks"], time.perf_counter()
    kind = "trial" if only is not None else ("ls" if int(getattr(opts, "ls", 0) or 0) > 0 else "full")
    name = getattr(opts, "_variant", "?")
    status = "done"
    try:
        r = _orig_run(job, opts, policy, pre, path_cache, only, bound)
        mk = r[1]
        return r
    except s8.Pruned:
        status, mk = "pruned", float("nan")
        raise
    finally:
        LOG.append({"kind": kind, "variant": name, "status": status, "makespan": mk,
                    "secs": round(time.perf_counter() - t0, 2), "checks": CNT["checks"] - c0})


_orig_options = solve.options


def _options(name, job, out):
    o = _orig_options(name, job, out)
    o._variant = name
    return o


s8.Checker._check = _check
s8.run = _run
solve.options = _options


def main() -> None:
    log, set_dir, shape, out, *rest = sys.argv[1:]
    Path(out).mkdir(parents=True, exist_ok=True)
    sys.argv = ["solve.py", str(ROOT / set_dir / shape), out, *rest]
    t0 = time.perf_counter()
    solve.main()
    wall = time.perf_counter() - t0
    new = not Path(log).exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["set", "shape", "kind", "variant", "status", "makespan", "secs", "checks"])
        if new:
            w.writeheader()
        for r in LOG:
            w.writerow({"set": Path(set_dir).name, "shape": shape, **r})
        w.writerow({"set": Path(set_dir).name, "shape": shape, "kind": "wall", "variant": "", "status": "",
                    "makespan": "", "secs": round(wall, 2), "checks": CNT["checks"]})
    print(f"{shape}: 벽시계 {wall:.1f} s, 충돌 검사 {CNT['checks']}", flush=True)


if __name__ == "__main__":
    main()
