# -*- coding: utf-8 -*-
"""국소 탐색 계산량 한도를 바꿔 가며 makespan 을 잰다(WAAM venv 파이썬). 충돌 검사 횟수는 어느 컴퓨터에서나
같으므로 여러 개를 동시에 돌려도 된다(시간은 참고만).

usage: python experiments/ls_budget/try_budget.py <log.csv> <묶음 폴더> <형상> <한도|none> [--spread --base N]
  한도   국소 탐색에서 다시 계획하는 데 쓸 충돌 검사 횟수
  --spread  아래층부터 쓰지 않고 층마다 나눠 준다(층 몫 = 그 층 기준 계획 검사 횟수 × 한도 / --base)
  --base    고른 설정을 그냥 계획할 때의 충돌 검사 횟수(--spread 에 필요, 없으면 여기서 계획해 잰다)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "WAAM_Validator" / "src"))

import schedule_v8 as s8  # noqa: E402
import solve  # noqa: E402
import toolpath  # noqa: E402
from portfolio import VARIANTS  # noqa: E402

USED = {"tries": 0, "checks": 0}
_orig_spend = s8._ls_spend


def _spend(ctx, c0):
    USED["tries"] += 1
    USED["checks"] += s8.CHECKS - c0
    return _orig_spend(ctx, c0)


s8._ls_spend = _spend


def best_of(name: str, set_dir: str, shape: str) -> str:
    p = ROOT / "runs" / "evals" / name / Path(set_dir).name / shape / "solve.txt"
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("best\t"):
            return line.split("\t")[1]
    raise SystemExit(f"best 없음: {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("set_dir")
    ap.add_argument("shape")
    ap.add_argument("budget")
    ap.add_argument("--spread", action="store_true")
    ap.add_argument("--base", type=float, default=None)
    ap.add_argument("--ls", type=int, default=2)
    ap.add_argument("--best-from", default="1110a_ls2_adapt뺌")
    a = ap.parse_args()
    toolpath.FAST_VORONOI = True
    job = ROOT / a.set_dir / a.shape
    best = best_of(a.best_from, a.set_dir, a.shape)
    pre = s8.prepare(job)
    cfg, n, layers = pre
    s8.apply_config(cfg, False)
    cache = solve.build_paths(layers, n, {VARIANTS[best][2]})[VARIANTS[best][2]]
    out = ROOT / "runs" / "tmp_profile"
    budget = None if a.budget == "none" else float(a.budget)
    base = a.base
    if a.spread and base is None:
        c = s8.CHECKS
        s8.run(job, solve.options(best, job, out), pre=pre, path_cache=cache)
        base = s8.CHECKS - c
    o = solve.options(best, job, out)
    o.ls = a.ls
    o.ls_budget = budget
    o.ls_rate = (budget / base) if (a.spread and budget is not None) else None
    c, t = s8.CHECKS, time.perf_counter()
    _, mk, *_ = s8.run(job, o, pre=pre, path_cache=cache)
    row = {"set": Path(a.set_dir).name, "shape": a.shape, "best": best, "budget": a.budget,
           "mode": "spread" if a.spread else "bottom", "makespan": round(mk, 3), "ls_checks": s8.CHECKS - c,
           "try_checks": USED["checks"], "tries": USED["tries"], "secs": round(time.perf_counter() - t, 1)}
    new = not Path(a.log).exists()
    with open(a.log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
