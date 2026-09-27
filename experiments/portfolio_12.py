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

# tune_cem.py 로 학습 형상(train_div 30개)에서 찾은 V8 설정값. 이 값만 단독으로 쓰면 처음 보는
# 형상에서 오히려 나쁘지만(검증 +0.3%), 변형으로 '추가'하면 이 설정이 잘 맞는 형상에서만 뽑혀
# 검증 20개 −0.93%. 그래서 기본 변형은 그대로 두고 v8t·v8balt 를 더해 5개를 돌린다.
V8_TUNED = ["--chunk", "41.4", "--band", "49.4", "--lookahead", "5", "--max-tries", "28", "--retract", "598.4"]
VARIANTS = {
    "v7":    ("schedule_v7.py", []),
    "v8":    ("schedule_v8.py", ["--no-balance"]),
    "v8bal": ("schedule_v8.py", []),
    "v8t":    ("schedule_v8.py", ["--no-balance", *V8_TUNED]),
    "v8balt": ("schedule_v8.py", V8_TUNED),
    # 긴 조각: 아크 재점화(끄기1s+켜기1s)가 makespan 의 6% 를 차지해(05 분석) 조각을 길게 잡아 줄인다.
    # 대신 충돌을 피할 여지가 줄어 형상에 따라 유불리가 갈리므로 변형으로만 둔다(sweep_chunk.py: 학습형상에서 80·220mm 조합이 최선).
    "v8c":    ("schedule_v8.py", ["--no-balance", "--chunk", "80"]),
    "v8balc": ("schedule_v8.py", ["--chunk", "80"]),
    "v8c160":    ("schedule_v8.py", ["--no-balance", "--chunk", "160"]),
    "v8balc160": ("schedule_v8.py", ["--chunk", "160"]),
    "v8c2":    ("schedule_v8.py", ["--no-balance", "--chunk", "220"]),
    "v8balc2": ("schedule_v8.py", ["--chunk", "220"]),
    # 좁은 형상: 3대가 서로 비켜 주는 시간이 분담 이득보다 커서 1~2대가 더 빠르다(01 −11%, 06 −15%).
    "solo":   ("schedule_v8.py", ["--robots", "1", "--chunk", "160"]),
    "duo":    ("schedule_v8.py", ["--robots", "2", "--chunk", "160"]),
}


# 코어가 모자랄 때 남길 순서(36개 형상에서 '한 개씩 더했을 때 합계가 가장 줄어드는' 순서로 계산).
# 3개만 써도 10개 전부 쓴 값 대비 +0.13%, 4개면 +0.07% 수준이다.
PRIORITY = ["v8balc2", "v8c160", "v8balc", "v8t", "v8balc160", "v8c2", "v8c", "v8balt", "v7", "v8"]


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
    ap.add_argument("--variants", default="v7,v8,v8t,v8balt,v8c,v8balc,v8c160,v8balc160,v8c2,v8balc2,solo,duo")
    ap.add_argument("--workers", type=int, default=0, help="동시 실행 개수 (0 = 변형 수와 코어 수 중 작은 값)")
    ap.add_argument("--all", action="store_true",
                    help="코어가 모자라도 변형을 줄이지 않고 전부 실행(순차 대기)")
    ap.add_argument("--keep", action="store_true", help="변형별 결과 폴더도 남김")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="층별 경로를 변형마다 따로 계산(예전 방식)")
    a = ap.parse_args()
    job, out = Path(a.job), Path(a.out)
    names = [v for v in a.variants.split(",") if v in VARIANTS]
    work = Path(str(out) + "_variants")
    work.mkdir(parents=True, exist_ok=True)
    nw = a.workers or min(len(names), mp.cpu_count())
    if not a.all and len(names) > nw:                 # 코어가 모자라면 기여 큰 변형만 남긴다
        keep = [n for n in PRIORITY if n in names][:max(nw, 2)]
        keep += [n for n in names if n not in keep][:max(0, max(nw, 2) - len(keep))]
        if len(keep) < len(names):
            print(f"  동시 실행 {nw}개(코어 {mp.cpu_count()}) → 변형 {len(names)}개 중 "
                  f"{len(keep)}개만 실행: {','.join(keep)}", flush=True)
        names = keep

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
