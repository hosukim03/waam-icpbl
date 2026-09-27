# -*- coding: utf-8 -*-
"""알고리즘 포트폴리오 실행기 — 여러 스케줄링 방식을 코어별로 동시에 돌려 makespan 이 가장 짧은 결과를 낸다.

형상마다 유리한 방식이 다르기 때문이다(검증용 20개: V7 6개, V8 4개, 균형 V8 10개가 각각 최선).
코어를 나눠 쓰므로 생성 시간은 '가장 느린 변형' 수준이고, 순차 실행처럼 합쳐지지 않는다.

기본 변형
  v7      : V7 (볼록 영역 예약, 조각 40mm, 후보 6)
  v8      : V8 (채점 기준 그대로 충돌 판정) + 기존 부채꼴 배정
  v8bal   : V8 + 부하 균형 배정(층 무게중심 기준 경계각 최적화)

usage:
  python portfolio.py <job_dir> <out_dir> [--variants v7,v8,v8bal] [--workers 3] [--keep]
결과: out_dir 에 가장 짧은 변형의 trajectory.csv/config.yaml/target.stl (--keep 이면 변형별 결과도 남김)
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VPY = ROOT / "WAAM_Validator" / ".venv" / "Scripts" / "python.exe"

VARIANTS = {
    "v7":    ("schedule_v7.py", []),
    "v8":    ("schedule_v8.py", ["--no-balance"]),
    "v8bal": ("schedule_v8.py", []),
}


def makespan_of(csv_path: Path) -> float:
    last = 0.0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            last = max(last, float(row["time_s"]))
    return last


def run_variant(args):
    name, job, out_root, cache = args
    script, extra = VARIANTS[name]
    extra = list(extra) + (["--paths", str(cache)] if cache else [])
    out = Path(out_root) / name
    t0 = time.time()
    p = subprocess.run([str(VPY), str(ROOT / "v0" / script), str(job), str(out), *extra],
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=ROOT / "WAAM_Validator")
    dt = time.time() - t0
    traj = out / "trajectory.csv"
    if p.returncode != 0 or not traj.exists():
        return name, None, dt, (p.stderr or "")[-300:]
    return name, makespan_of(traj), dt, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--variants", default="v7,v8,v8bal")
    ap.add_argument("--workers", type=int, default=0, help="동시 실행 개수 (0 = 변형 수와 코어 수 중 작은 값)")
    ap.add_argument("--keep", action="store_true", help="변형별 결과 폴더도 남김")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="층별 경로를 변형마다 따로 계산(예전 방식)")
    a = ap.parse_args()
    job, out = Path(a.job), Path(a.out)
    names = [v for v in a.variants.split(",") if v in VARIANTS]
    work = Path(str(out) + "_variants")
    work.mkdir(parents=True, exist_ok=True)
    nw = a.workers or min(len(names), mp.cpu_count())

    t0 = time.time()
    cache = None
    if a.cache:                                   # 층별 경로를 코어별로 나눠 한 번만 계산해 공유
        cache = work / "paths.pkl"
        c = subprocess.run([str(VPY), str(ROOT / "v0" / "toolpath_cache.py"), str(job), str(cache)],
                           capture_output=True, text=True, encoding="utf-8", cwd=ROOT / "WAAM_Validator")
        if c.returncode != 0 or not cache.exists():
            print(f"  경로 캐시 실패 → 변형별 계산: {(c.stderr or '')[-120:]}", flush=True)
            cache = None
        else:
            print(f"  {c.stdout.strip()}", flush=True)
    with mp.Pool(nw) as pool:
        res = pool.map(run_variant, [(n, job, work, cache) for n in names])
    ok = [r for r in res if r[1] is not None]
    for name, mk, dt, err in res:
        tag = f"{mk:.1f}s ({mk / 3600:.2f}h)" if mk else f"실패: {err[:80]}"
        print(f"  {name:6s} {tag}  gen={dt:.1f}s", flush=True)
    if not ok:
        print("모든 변형 실패")
        sys.exit(1)
    best = min(ok, key=lambda r: r[1])
    out.mkdir(parents=True, exist_ok=True)
    for f in ("trajectory.csv", "config.yaml", "target.stl"):
        shutil.copyfile(work / best[0] / f, out / f)
    (out / "portfolio.txt").write_text(
        "\n".join(f"{n}\t{'' if m is None else f'{m:.3f}'}\t{d:.1f}s" for n, m, d, _ in res)
        + f"\nbest\t{best[0]}\n", encoding="utf-8")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    print(f"{job.name}: 선택 {best[0]} makespan={best[1]:.1f}s ({best[1] / 3600:.2f}h) "
          f"총 {time.time() - t0:.1f}s (병렬 {nw})", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
