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
import json
import csv
import multiprocessing as mp
import os
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
    # 1대 설정은 띠(반지름 60mm, 바깥부터) 나누기가 필요 없다: 이웃 고리가 다른 띠로 갈리면 잇지 못한다(−0.1~−0.9%)
    # 1대는 잇기 기준도 13mm(9→13: 벌집 −1.1%·공개06 −0.7%, 채점 IoU·넘침 변화 없음)
    "r1c160":      ("schedule_v8.py", ["--robots", "1", "--chunk", "160", "--band", "99999", *NP, "--link-gap", "13"], "ring"),
    "r2c160":      ("schedule_v8.py", ["--robots", "2", "--chunk", "160", *NP], "ring"),
    "r3c80":       ("schedule_v8.py", ["--no-balance", "--chunk", "80", *NP], "ring"),
    "r3c160":      ("schedule_v8.py", ["--no-balance", "--chunk", "160", *NP], "ring"),
    "r3c220bal":   ("schedule_v8.py", ["--chunk", "220", *NP], "ring"),
    "r3tuned":     ("schedule_v8.py", ["--no-balance", *V8_TUNED, *NP], "ring"),
    "sp_r1c160":   ("schedule_v8.py", ["--robots", "1", "--chunk", "160", "--band", "99999", *NP, "--link-gap", "13"], "sp"),
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
    # --islands: 층에 떨어진 섬이 여럿이면 각도순으로 이웃한 섬끼리 묶어 로봇에 배분(505-03 섬 −8.3%).
    # 섬이 하나인 층은 원래대로 부채꼴이라 섬 없는 형상은 결과가 같다.
    "sp_r3c220balm5": ("schedule_v8.py", ["--chunk", "220", "--margin", "5", "--islands", *NP, *LA], "sp"),
    # 나선 경로 + 2대 + 로봇마다 띠 방향 엇갈리기: 폭 300~400mm 격자형에서 두 팔이 가운데서 동시에
    # 만나는 걸 줄인다(505-02 벌집 −3.4%, 505-15 삼각격자 −1.7%)
    # rp = 아래층 경로 재사용(--reuse-prev). 같은 설정이라도 경로가 조금 다르면 결과가 ±3~6% 흔들려서,
    # 자주 이기는 설정 3개를 재사용 쪽으로도 돌려 형상마다 좋은 쪽을 고른다(81형상 추정 −0.55%)
    "sp_r3c220balm5rp": ("schedule_v8.py", ["--chunk", "220", "--margin", "5", "--islands", "--reuse-prev", *NP, *LA], "sp"),
    "sp_r1c160rp": ("schedule_v8.py", ["--robots", "1", "--chunk", "160", "--band", "99999", "--reuse-prev", *NP,
                                       "--link-gap", "13"], "sp"),
    "r2c160m5rp":  ("schedule_v8.py", ["--robots", "2", "--chunk", "160", "--margin", "5", "--reuse-prev", *NP, *LA], "ring"),
    "sp_r2c160m5arp": ("schedule_v8.py", ["--robots", "2", "--chunk", "160", "--margin", "5", "--band-dir", "alt",
                                          "--reuse-prev", *NP, *LA], "sp"),
    "r1c160rp":    ("schedule_v8.py", ["--robots", "1", "--chunk", "160", "--band", "99999", "--reuse-prev", *NP,
                                       "--link-gap", "13"], "ring"),
    "sp_r2c160m5a": ("schedule_v8.py", ["--robots", "2", "--chunk", "160", "--margin", "5", "--band-dir", "alt",
                                        *NP, *LA], "sp"),
    # 위와 같고 조각만 80 mm. 아크 대기를 뺀 뒤로는 같은 줄의 다음 조각을 끊지 않고 이어 그리므로 잘게 잘라도 손해가
    # 거의 없고, 로봇이 작은 단위로 일을 맡아 막혔을 때 덜 기다린다(공개 04: 160 mm 28,876 s → 80 mm 27,853 s)
    "sp_r2c80m5a": ("schedule_v8.py", ["--robots", "2", "--chunk", "80", "--margin", "5", "--band-dir", "alt",
                                       *NP, *LA], "sp"),
}


# 기본으로 돌릴 설정 8개: 51개 형상에서 하나씩 더해 가며 합계가 가장 줄어드는 순서로 고른 상위 8개.
# 12개 전부 대비 +0.002%(사실상 동일)이고 프로세스가 줄어 코어 경쟁·생성시간이 준다.
# 81형상 기여도로 r2c160m5·sp_r3c160o2(빼도 손해 0%)를 빼고, 가장 중요한 설정들의 재사용 버전을 넣었다
DEFAULT = ["sp_r3c220balm5", "sp_r1c160", "r3c160m5", "r3c220bal", "r3tuned", "r1c160",
           "sp_r2c160m5a", "sp_r3c220balm5rp", "sp_r1c160rp", "r2c160m5rp", "sp_r2c160m5arp", "r1c160rp"]

# ---------------------------------------------------------------- 형상을 보고 돌릴 설정 고르기
# 설정 12개를 전부 돌리면 코어가 많은 컴퓨터에서만 빠르다(총 계산량은 시계 시간의 약 6배).
# 형상 특징(features.py, 1초 안팎)을 보고 필요한 설정만 돌려 총 계산량을 줄인다.
# 근거(2026-10-02, 공개 6 + check808 21개, 가지치기 없이 설정별 makespan 을 전부 구한 표):
#   12개 전부 대비 makespan +0.09%, 설정 계산량 34%, 가장 손해 본 형상 +1.2%.
# 두 로봇이 동시에 설 수 있는 조합이 없으면(pair_max≈0) 한 번에 한 대만 적층할 수 있는 형상이다.
PAIR_EPS = 0.01
SOLO = ["sp_r1c160", "sp_r1c160rp"]                       # 한 대로 다 닿는다 → 1대 설정
TURN = ["sp_r2c160m5arp", "sp_r2c160m5a", "sp_r3c220balm5rp"]   # 한 대로는 안 닿는다 → 번갈아 적층
MULTI = ["sp_r2c160m5a", "sp_r3c220balm5rp", "r3tuned", "sp_r1c160"]   # 여러 대가 나눠 적층


def select_variants(job: Path) -> tuple[list[str], dict]:
    """형상 특징으로 돌릴 설정 목록을 정한다. 특징 계산이 실패하면 여러 대 설정으로 간다."""
    c = subprocess.run([str(VPY), str(ROOT / "src" / "features.py"), str(job)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT / "WAAM_Validator")
    try:
        f = json.loads(c.stdout.strip().splitlines()[-1])
    except Exception:
        return list(MULTI), {}
    if f["pair_max"] < PAIR_EPS:
        return list(SOLO if f["reach1"] >= 0.999 else TURN), f
    return list(MULTI), f


# 코어가 모자랄 때 남길 순서(36개 형상에서 '한 개씩 더했을 때 합계가 가장 줄어드는' 순서로 계산).
# 2개면 전부 대비 +1.18%, 3개 +0.50%, 4개 +0.26%, 5개 +0.10% 수준이다.
PRIORITY = list(DEFAULT)


def makespan_of(csv_path: Path) -> float:
    last = 0.0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            last = max(last, float(row["time_s"]))
    return last


def _write_bound(work: Path, bound: Path, value: float) -> None:
    """최선 makespan 을 bound 파일에 쓴다. 참고용이라 실패해도 절대 예외를 내지 않는다.

    윈도우에서는 스케줄러가 파일을 읽는 찰나에 바꿔치기하면 '액세스 거부'가 난다(검증 19 에서 생성 실패).
    몇 번 재시도하고, 끝내 못 쓰면 이번 갱신은 건너뛴다(가지치기가 조금 늦어질 뿐 결과는 같다).
    """
    tmp = work / f"best_makespan.{os.getpid()}.tmp"
    for _ in range(20):
        try:
            tmp.write_text(f"{value:.6f}")
            os.replace(tmp, bound)            # 읽는 쪽이 반쯤 쓴 값을 보지 않도록 바꿔치기
            return
        except OSError:
            time.sleep(0.05)
    try:
        tmp.unlink()
    except OSError:
        pass


def run_variant(args):
    name, job, out_root, caches, bound = args
    script, extra, tag = VARIANTS[name]
    cache = (caches or {}).get(tag)
    extra = list(extra) + (["--paths", str(cache)] if cache else [])
    if bound:
        extra += ["--bound-file", str(bound)]
    out = Path(out_root) / name
    t0 = time.time()
    p = subprocess.run([str(VPY), str(ROOT / "src" / script), str(job), str(out), *extra],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=ROOT / "WAAM_Validator")
    dt = time.time() - t0
    traj = out / "trajectory.csv"
    if p.returncode == 3:                       # 가지치기로 중단: 이미 다른 설정보다 느림
        return name, None, dt, "pruned"
    if p.returncode != 0 or not traj.exists():
        return name, None, dt, (p.stderr or "")[-300:]
    return name, makespan_of(traj), dt, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--variants", default=None,
                    help="돌릴 설정을 직접 지정(쉼표로 구분). 생략하면 형상 특징으로 고른다")
    ap.add_argument("--full", action="store_true",
                    help="형상을 보고 고르지 않고 기본 설정 12개를 전부 돌린다(예전 방식)")
    ap.add_argument("--workers", type=int, default=0, help="동시 실행 개수 (0 = 설정 수와 코어 수 중 작은 값)")
    ap.add_argument("--all", action="store_true",
                    help="코어가 모자라도 설정을 줄이지 않고 전부 실행(순차 대기)")
    ap.add_argument("--no-prune", dest="prune", action="store_false",
                    help="이미 진 설정을 중간에 멈추지 않는다")
    ap.add_argument("--keep", action="store_true", help="설정별 결과 폴더도 남김")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="층별 경로를 설정마다 따로 계산(예전 방식)")
    a = ap.parse_args()
    job, out = Path(a.job).resolve(), Path(a.out).resolve()   # 하위 프로세스는 채점기 폴더에서 실행된다
    if a.variants:
        names = [v for v in a.variants.split(",") if v in VARIANTS]
    elif a.full:
        names = list(DEFAULT)
    else:
        names, feat = select_variants(job)
        print(f"  형상 특징 {{{', '.join(f'{k}={v:.3g}' for k, v in feat.items())}}} → 설정 {len(names)}개: {','.join(names)}",
              flush=True)
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
        # 두 묶음이 다 필요하면 한 번에 계산한다(고리 후보를 두 번 평가하지 않음)
        # (경로 재사용 --reuse 는 쓰지 않는다: 모양이 거의 같은 층이라도 경로 시작점·순서가 조금 바뀌면
        #  로봇 순서 전체가 달라져 형상마다 makespan 이 ±3~6% 흔들렸다 — 404 +2.0%, 404-04 +6.5%)
        jobs_ = ([("sp", ["--both", str(work / "paths_ring.pkl")])] if need == {"sp", "ring"}
                 else [(t, [] if t == "sp" else ["--no-spiral"]) for t in sorted(need)])
        for tag, extra in jobs_:
            cpath = work / f"paths_{tag}.pkl"
            c = subprocess.run([str(VPY), str(ROOT / "src" / "toolpath_cache.py"), str(job), str(cpath), *extra],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT / "WAAM_Validator")
            if c.returncode == 0 and cpath.exists():
                caches[tag] = cpath
                if extra and extra[0] == "--both" and Path(extra[1]).exists():
                    caches["ring"] = Path(extra[1])
                msg = (c.stdout or "").strip().encode("ascii", "ignore").decode()   # 자식 출력 인코딩이 섞인다
                print(f"  [{'+'.join(sorted(need)) if extra[:1] == ['--both'] else tag}] 경로 캐시 준비 {msg[-40:]}", flush=True)
            else:
                print(f"  [{tag}] 경로 캐시 실패 → 설정별 계산: {(c.stderr or '')[-120:]}", flush=True)
    # 가지치기: 설정이 하나 끝날 때마다 지금까지 최선 makespan 을 파일에 적는다. 아직 도는 설정은
    # 층마다 이 값을 읽어, 자기 로봇 시각이 이미 넘었으면 멈춘다(결과는 바뀌지 않고 시간만 준다).
    bound = work / "best_makespan.txt" if a.prune else None
    if bound:
        bound.write_text("inf")
    res, best_mk = [], float("inf")
    with mp.Pool(nw) as pool:
        for r in pool.imap_unordered(run_variant, [(n, job, work, caches, bound) for n in names]):
            res.append(r)
            if bound and r[1] is not None and r[1] < best_mk:
                best_mk = r[1]
                _write_bound(work, bound, best_mk)
    ok = [r for r in res if r[1] is not None]
    for name, mk, dt, err in res:
        tag = f"{mk:.1f}s ({mk / 3600:.2f}h)" if mk else ("중단(이미 짐)" if err == "pruned" else f"실패: {err[:80]}")
        print(f"  {name:6s} {tag}  gen={dt:.1f}s", flush=True)
    if not ok:
        print("모든 설정 실패")
        sys.exit(1)
    order = {n: i for i, n in enumerate(names)}           # 동점이면 설정 목록 순서로(끝난 순서와 무관하게 같은 답)
    best = min(ok, key=lambda r: (round(r[1], 6), order[r[0]]))
    res.sort(key=lambda r: order[r[0]])
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
