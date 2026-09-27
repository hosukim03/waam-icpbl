# -*- coding: utf-8 -*-
"""알고리즘 포트폴리오 실행기 — 여러 스케줄링 방식을 코어별로 동시에 돌려 makespan 이 가장 짧은 결과를 낸다.

형상마다 유리한 방식이 다르기 때문이다(검증용 20개: V7 6개, V8 4개, 균형 V8 10개가 각각 최선).
코어를 나눠 쓰므로 생성 시간은 '가장 느린 설정' 수준이고, 순차 실행처럼 합쳐지지 않는다.

기본 설정
  r1c160 / r2c160 : 일감을 로봇 1대·2대에게만 준다 (좁은 형상은 서로 비켜 주는 시간이 더 커서 유리)
  r3c80 / r3c160 / r3c220 : 로봇 3대, 용접선을 80·160·220mm 조각으로 끊는다 (조각이 길수록 아크 재점화가 준다)
  ...bal  : 부하 균형 배정(층 무게중심 기준 경계각 최적화). 없으면 작업대 중심 부채꼴
  r3tuned : CEM 으로 찾은 설정값(조각 41.4mm 등)

usage:
  python portfolio.py <job_dir> <out_dir> [--variants v7,v8,v8bal] [--workers 3] [--keep]
결과: out_dir 에 가장 짧은 설정의 trajectory.csv/config.yaml/target.stl (--keep 이면 설정별 결과도 남김)
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
# 형상에서 오히려 나쁘지만(검증 +0.3%), 설정으로 '추가'하면 이 설정이 잘 맞는 형상에서만 뽑혀
# 검증 20개 −0.93%. 그래서 설정(r3tuned)으로만 둔다.
# 기본으로 붙이는 인자
NP = ["--no-park-layer", "--link-gap", "9"]
# 후보 25: 막혔을 때 큐 앞 25개까지 훑어 바로 시작할 조각을 찾는다(04 −13.2%). 전 설정에 넣으면
# 15개 동시 실행 중 코어 경쟁으로 생성시간이 2.7배가 되어, 가장 자주 뽑히는 m5 설정 3개에만 붙인다.
LA = ["--lookahead", "25"]
V8_TUNED = ["--chunk", "41.4", "--band", "49.4", "--lookahead", "5", "--max-tries", "28", "--retract", "598.4"]
# 이름 규칙: r=로봇 수, c=조각 길이(mm), bal=부하 균형 배정, sp=나선 경로(고리를 이어 붙인 것)
# 값 = (스크립트, 추가 인자, 경로 묶음)  경로 묶음: "sp"=나선 포함, "ring"=동심원 고리만
VARIANTS = {
    # 기본: 층 끝에 주차하지 않고 그 자리에서 다음 층 시작(왕복 제거, 05 −9.5%),
    #       배정된 조각 사이가 9mm 안이면 아크를 끄지 않고 이어 긋기(05 −2.8%, 06 −4.9%).
    # 예비로 주차하는 설정(...park)과 잇지 않는 설정(...nolink)도 남긴다.
    "r1c160":      ("schedule_v8.py", ["--robots", "1", "--chunk", "160", *NP], "ring"),
    "r2c160":      ("schedule_v8.py", ["--robots", "2", "--chunk", "160", *NP], "ring"),
    "r3c80":       ("schedule_v8.py", ["--no-balance", "--chunk", "80", *NP], "ring"),
    "r3c160":      ("schedule_v8.py", ["--no-balance", "--chunk", "160", *NP], "ring"),
    "r3c220bal":   ("schedule_v8.py", ["--chunk", "220", *NP], "ring"),
    "r3tuned":     ("schedule_v8.py", ["--no-balance", *V8_TUNED, *NP], "ring"),
    "sp_r1c160":   ("schedule_v8.py", ["--robots", "1", "--chunk", "160", *NP], "sp"),
    "sp_r3c160":   ("schedule_v8.py", ["--no-balance", "--chunk", "160", *NP], "sp"),
    "sp_r3c160o2": ("schedule_v8.py", ["--no-balance", "--chunk", "160", "--order2", *NP], "sp"),
    "sp_r3c220bal": ("schedule_v8.py", ["--chunk", "220", *NP], "sp"),
    "r3c160nolink": ("schedule_v8.py", ["--no-balance", "--chunk", "160", "--no-park-layer"], "ring"),
    "sp_r3c160park": ("schedule_v8.py", ["--no-balance", "--chunk", "160"], "sp"),
    # m5 = 충돌 판정 여유 20→5mm(규칙 250 위에 얹는 값). 팔이 15mm 더 붙을 수 있어 좁은 형상에 유리
    # (04 −2.3%, 채점기 최소 팔거리 254.7mm 로 PASS). 이동 중엔 여유가 빠듯할 수 있어 제출기가
    # 채점기로 검증하고 실패하면 다음 설정으로 넘어간다(export_waam.py).
    "r2c160m5":     ("schedule_v8.py", ["--robots", "2", "--chunk", "160", "--margin", "5", *NP, *LA], "ring"),
    "r3c160m5":     ("schedule_v8.py", ["--no-balance", "--chunk", "160", "--margin", "5", *NP, *LA], "ring"),
    "sp_r3c220balm5": ("schedule_v8.py", ["--chunk", "220", "--margin", "5", *NP, *LA], "sp"),
}


# 코어가 모자랄 때 남길 순서(36개 형상에서 '한 개씩 더했을 때 합계가 가장 줄어드는' 순서로 계산).
# 2개면 전부 대비 +1.18%, 3개 +0.50%, 4개 +0.26%, 5개 +0.10% 수준이다.
PRIORITY = ["sp_r3c160", "r2c160", "r1c160", "sp_r1c160", "r2c160m5", "sp_r3c160o2", "r3c220bal",
            "sp_r3c220bal", "sp_r3c220balm5", "r3c160", "r3c160m5", "r3tuned", "r3c80",
            "sp_r3c160park", "r3c160nolink"]


def makespan_of(csv_path: Path) -> float:
    last = 0.0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            last = max(last, float(row["time_s"]))
    return last


def run_variant(args):
    name, job, out_root, caches = args
    script, extra, tag = VARIANTS[name]
    cache = (caches or {}).get(tag)
    extra = list(extra) + (["--paths", str(cache)] if cache else [])
    out = Path(out_root) / name
    t0 = time.time()
    p = subprocess.run([str(VPY), str(ROOT / "v0" / script), str(job), str(out), *extra],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
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
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=0, help="동시 실행 개수 (0 = 설정 수와 코어 수 중 작은 값)")
    ap.add_argument("--all", action="store_true",
                    help="코어가 모자라도 설정을 줄이지 않고 전부 실행(순차 대기)")
    ap.add_argument("--keep", action="store_true", help="설정별 결과 폴더도 남김")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="층별 경로를 설정마다 따로 계산(예전 방식)")
    a = ap.parse_args()
    job, out = Path(a.job), Path(a.out)
    names = [v for v in a.variants.split(",") if v in VARIANTS]
    work = Path(str(out) + "_variants")
    work.mkdir(parents=True, exist_ok=True)
    nw = a.workers or min(len(names), mp.cpu_count())
    if not a.all and len(names) > nw:                 # 코어가 모자라면 기여 큰 설정만 남긴다
        keep = [n for n in PRIORITY if n in names][:max(nw, 2)]
        keep += [n for n in names if n not in keep][:max(0, max(nw, 2) - len(keep))]
        if len(keep) < len(names):
            print(f"  동시 실행 {nw}개(코어 {mp.cpu_count()}) → 설정 {len(names)}개 중 "
                  f"{len(keep)}개만 실행: {','.join(keep)}", flush=True)
        names = keep

    t0 = time.time()
    caches = {}
    if a.cache:            # 층별 경로를 코어별로 나눠 한 번만 계산해 공유(나선 포함/미포함 두 묶음)
        need = {VARIANTS[n][2] for n in names}
        for tag in sorted(need):
            cpath = work / f"paths_{tag}.pkl"
            c = subprocess.run([str(VPY), str(ROOT / "v0" / "toolpath_cache.py"), str(job), str(cpath),
                                *([] if tag == "sp" else ["--no-spiral"])],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT / "WAAM_Validator")
            if c.returncode == 0 and cpath.exists():
                caches[tag] = cpath
                msg = (c.stdout or "").strip().encode("ascii", "ignore").decode()   # 자식 출력 인코딩이 섞인다
                print(f"  [{tag}] 경로 캐시 준비 {msg[-40:]}", flush=True)
            else:
                print(f"  [{tag}] 경로 캐시 실패 → 설정별 계산: {(c.stderr or '')[-120:]}", flush=True)
    with mp.Pool(nw) as pool:
        res = pool.map(run_variant, [(n, job, work, caches) for n in names])
    ok = [r for r in res if r[1] is not None]
    for name, mk, dt, err in res:
        tag = f"{mk:.1f}s ({mk / 3600:.2f}h)" if mk else f"실패: {err[:80]}"
        print(f"  {name:6s} {tag}  gen={dt:.1f}s", flush=True)
    if not ok:
        print("모든 설정 실패")
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
