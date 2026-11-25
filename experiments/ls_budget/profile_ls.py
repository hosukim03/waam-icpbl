# -*- coding: utf-8 -*-
"""국소 탐색이 어디에 계산을 쓰는지 잰다(계산량 한도의 단위를 고르기 위한 시험, WAAM venv 파이썬).

고른 설정(평가 기록의 best)을 그대로 계획(기준)한 뒤 국소 탐색으로 한 번 더 계획하며, 층 계획 한 번마다
걸린 시간·충돌 검사 횟수·거리 계산 수를 적는다. 시간 대신 셀 수 있는 양 가운데 시간과 가장 고르게
비례하는 것을 한도의 단위로 쓰려는 것이다.

usage: python experiments/ls_budget/profile_ls.py <out.csv> <묶음 폴더> <형상> [<형상> ...] [--best-from 실험 이름]
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

CNT = {"checks": 0, "starts": 0, "sd": 0, "rows": 0}
PHASE = ["?"]
LOG = []

_orig_check = s8.Checker._check
_orig_sd = s8.seg_dist
_orig_place = s8._place_layer


def _check(self, i, rel_ts, rel_xy, starts):
    CNT["checks"] += 1
    CNT["starts"] += len(starts)
    return _orig_check(self, i, rel_ts, rel_xy, starts)


def _sd(a0, a1, b0, b1):
    CNT["sd"] += 1
    CNT["rows"] += len(a1)
    return _orig_sd(a0, a1, b0, b1)


def _place(ctx, k, paths, z, zl, P, policy=None):
    c = dict(CNT)
    t0 = time.perf_counter()
    kind = "try" if P.get("shift") else "base"
    ok = 1
    try:
        return _orig_place(ctx, k, paths, z, zl, P, policy)
    except s8.OutOfReach:
        ok = 0
        raise
    finally:
        LOG.append({"phase": PHASE[0], "layer": k, "kind": kind, "ok": ok,
                    "secs": time.perf_counter() - t0, "segs": sum(len(Q) for Q in paths),
                    **{key: CNT[key] - c[key] for key in CNT}})


s8.Checker._check = _check
s8.seg_dist = _sd
s8._place_layer = _place


def best_of(name: str, set_dir: str, shape: str) -> str:
    p = ROOT / "runs" / "evals" / name / Path(set_dir).name / shape / "solve.txt"
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("best\t"):
            return line.split("\t")[1]
    raise SystemExit(f"best 없음: {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("set_dir")
    ap.add_argument("shapes", nargs="+")
    ap.add_argument("--best-from", default="1110a_ls2_adapt뺌")
    ap.add_argument("--ls", type=int, default=2)
    a = ap.parse_args()
    toolpath.FAST_VORONOI = True
    rows = []
    for shape in a.shapes:
        job = ROOT / a.set_dir / shape
        best = best_of(a.best_from, a.set_dir, shape)
        if not solve.uses_sectors(best):
            print(f"{shape}: {best} 는 부채꼴 배정을 안 써서 국소 탐색 없음", flush=True)
            continue
        pre = s8.prepare(job)
        cfg, n, layers = pre
        s8.apply_config(cfg, False)
        caches = solve.build_paths(layers, n, {VARIANTS[best][2]})
        out = ROOT / "runs" / "tmp_profile"
        LOG.clear()
        PHASE[0] = "plain"
        t = time.perf_counter()
        _, mk0, *_ = s8.run(job, solve.options(best, job, out), pre=pre, path_cache=caches[VARIANTS[best][2]])
        t_plain = time.perf_counter() - t
        o = solve.options(best, job, out)
        o.ls = a.ls
        PHASE[0] = "ls"
        t = time.perf_counter()
        _, mk1, *_ = s8.run(job, o, pre=pre, path_cache=caches[VARIANTS[best][2]])
        t_ls = time.perf_counter() - t
        for r in LOG:
            rows.append({"set": Path(a.set_dir).name, "shape": shape, "best": best, **r})
        tot = {ph: {k: sum(r[k] for r in LOG if r["phase"] == ph) for k in ("secs", "checks", "starts", "sd", "rows")}
               for ph in ("plain", "ls")}
        print(f"{shape}: {best} 층 {n} 기준 {mk0:.1f} → 국소 탐색 {mk1:.1f} ({(mk1 / mk0 - 1) * 100:+.2f}%) | "
              f"기준 계획 {t_plain:.1f} s 검사 {tot['plain']['checks']} | 국소 탐색 {t_ls:.1f} s 검사 {tot['ls']['checks']} "
              f"거리 {tot['ls']['rows']}", flush=True)
    keys = ["set", "shape", "best", "phase", "layer", "kind", "ok", "secs", "segs", "checks", "starts", "sd", "rows"]
    new = not Path(a.out).exists()
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
