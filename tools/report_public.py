# -*- coding: utf-8 -*-
"""공개 모델의 makespan 과 계산 시간을 한 화면 표로 보여 준다(보고·캡처용).

형상마다
  1) 우리 알고리즘(기본 src/solve.py, --solver portfolio 면 예전 src/portfolio.py)으로 trajectory.csv 를
     만들며 시계 시간을 잰다. = 계산 시간: 형상을 넣은 순간부터 경로 파일이 나올 때까지. 실행 중 학습은 없다.
  2) 교수님 채점기(waam-validator run)로 채점해 판정과 makespan 을 받는다.
     표의 makespan 은 채점기가 낸 값이다(채점에 걸린 시간은 계산 시간에 넣지 않는다).
마지막에 표와 합계를 출력하고 results/report_public_<날짜시각>.csv 로 남긴다.

usage: python tools/report_public.py [--solver solve|portfolio] [--models 01,02] [--cpus 2] [--args "..."]   (WAAM venv 파이썬)
  --solver  solve = 코어 1개용 풀이(기본), portfolio = 설정 여러 개를 코어별로 동시에 돌리는 예전 방식
  --full    (portfolio 전용) 형상을 보고 고르지 않고 설정 12개를 전부 돌린다
  --models  일부 형상만(기본 01~06 전부)
  --cpus    계산(풀이 프로그램과 그 하위 프로세스)을 이 논리 CPU 들로만 돌린다. 코어 수도 그 개수로 알린다
            (PYTHON_CPU_COUNT). 코어 1개 컴퓨터와 같은 조건을 만든다. 채점은 제한하지 않는다.
  --args    풀이 프로그램에 그대로 넘길 인자
"""
import argparse
import csv
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
VPY = ROOT / "WAAM_Validator" / ".venv" / "Scripts" / "python.exe"
TESTS = ROOT / "WAAM_Validator" / "tests"


def width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, w: int, right: bool = False) -> str:
    gap = " " * max(0, w - width(s))
    return gap + s if right else s + gap


def hms(sec: float) -> str:
    h, rem = divmod(int(round(sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}시간 {m:02d}분 {s:02d}초" if h else f"{m}분 {s:02d}초"


def machine() -> str:
    cpu = platform.processor()
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    ram = psutil.virtual_memory().total / 2 ** 30
    bat = psutil.sensors_battery()
    power = "충전기 연결" if bat is None or bat.power_plugged else "배터리(계산 시간이 느리게 나올 수 있음)"
    return (f"{cpu} (코어 {psutil.cpu_count(logical=False)}개) · RAM {ram:.1f} GB(사용 가능) · "
            f"{platform.system()} {platform.release()} ({platform.version()}) · 전원: {power}")


def code_version() -> str:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rev = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "src")
    return rev + (" + 커밋 전 변경(src)" if dirty else "")


def keep_awake(on: bool) -> None:
    """측정 중 절전(모던 스탠바이)으로 들어가 계산 시간이 부풀지 않게 한다. 설정은 바꾸지 않는다."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["solve", "portfolio"], default="solve")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--models", default="01,02,03,04,05,06")
    ap.add_argument("--cpus", default="", help="계산을 돌릴 논리 CPU 번호(쉼표로 구분). 비우면 제한 없음")
    ap.add_argument("--args", default="", help="풀이 프로그램에 그대로 넘길 인자")
    a = ap.parse_args()
    models = a.models.split(",")
    cpus = [int(c) for c in a.cpus.split(",") if c.strip()]
    stamp = datetime.now()
    work = ROOT / "jobs_report" / stamp.strftime("%Y%m%d_%H%M%S")
    if a.solver == "solve":
        mode = "solve.py(코어 하나로 후보를 차례로 계획, 가지치기)"
    else:
        mode = "portfolio.py " + ("설정 12개 전부(--full)" if a.full else "형상을 보고 설정 고르기")
    if a.args:
        mode += f" · 인자 {a.args}"
    limit = f"코어 {len(cpus)}개로 제한(CPU {','.join(map(str, cpus))}번)" if cpus else "제한 없음"
    print(f"공개 모델 {len(models)}개 실행 시작 ({mode} · {limit}) …", flush=True)
    keep_awake(True)

    me = psutil.Process()
    all_cpus = me.cpu_affinity()
    env = dict(os.environ, **({"PYTHON_CPU_COUNT": str(len(cpus))} if cpus else {}))
    rows = []
    for m in models:
        out = work / m
        if cpus:
            me.cpu_affinity(cpus)          # 하위 프로세스는 만들어질 때 이 제한을 물려받는다
        t0 = time.perf_counter()
        g = subprocess.run([str(VPY), str(ROOT / "src" / f"{a.solver}.py"), str(TESTS / m), str(out),
                            *(["--full"] if a.full and a.solver == "portfolio" else []), *a.args.split()],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT / "src", env=env)
        gen = time.perf_counter() - t0
        me.cpu_affinity(all_cpus)          # 채점은 계산 시간에 넣지 않으므로 제한 없이 돌린다
        row = {"model": m, "gen_s": gen, "status": "생성 실패", "makespan_s": None, "coll": "", "iou": ""}
        if g.returncode == 0 and (out / "trajectory.csv").exists():
            runs = work / "_validator" / m
            v = subprocess.run([str(VPY), "-m", "waam_validator.cli", "run", str(out), "--output", str(runs), "--json"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT / "WAAM_Validator")
            try:
                s = json.loads(v.stdout)
                row.update(status=s["status"], makespan_s=float(s["schedule"]["makespan_s"]),
                           coll=s["collision"]["collision_event_count"], iou=f"{s['shape']['iou']:.4f}")
            except Exception:
                row["status"] = "채점 오류"
        rows.append(row)
        mk = f"{row['makespan_s']:.1f} s" if row["makespan_s"] else "-"
        print(f"  {m}: 계산 {gen:.1f} s → {row['status']}, makespan {mk}", flush=True)

    # ---------------------------------------------------------------- 결과 표
    W = [6, 15, 20, 15, 22]
    head = ["모델", "makespan [s]", "makespan", "계산 시간 [s]", "채점기 판정"]
    line = "=" * (sum(W) + 2 * len(W))
    print("\n" + line)
    print(" WAAM 3로봇 경로계획 — 공개 모델 결과")
    print(f" 측정 일시 : {stamp:%Y-%m-%d %H:%M}")
    print(f" 컴퓨터    : {machine()}")
    print(f" 코드      : {code_version()} · {mode}")
    print(f" 코어 제한 : {limit}")
    print(" 계산 시간 : 형상을 넣은 순간부터 경로 파일이 나올 때까지의 시계 시간(학습 시간 없음, 채점 시간 제외)")
    print(" makespan  : 교수님 채점기(WAAM Validator)가 계산한 값")
    print(line)
    print("  ".join(pad(h, w, i in (1, 3)) for i, (h, w) in enumerate(zip(head, W))))
    print("-" * len(line))
    tot_mk = tot_gen = 0.0
    for r in rows:
        mk = r["makespan_s"]
        tot_gen += r["gen_s"]
        tot_mk += mk or 0.0
        judge = f"{r['status']} (충돌 {r['coll']}, IoU {r['iou']})" if mk else r["status"]
        cells = [r["model"], f"{mk:,.1f}" if mk else "-", hms(mk) if mk else "-", f"{r['gen_s']:,.1f}", judge]
        print("  ".join(pad(c, w, i in (1, 3)) for i, (c, w) in enumerate(zip(cells, W))))
    print("-" * len(line))
    npass = sum(r["status"] == "PASS" for r in rows)
    cells = ["합계", f"{tot_mk:,.1f}", hms(tot_mk), f"{tot_gen:,.1f}", f"PASS {npass}/{len(rows)}"]
    print("  ".join(pad(c, w, i in (1, 3)) for i, (c, w) in enumerate(zip(cells, W))))
    print(line)

    keep_awake(False)
    path = ROOT / "results" / f"report_public_{stamp:%Y%m%d_%H%M}{f'_{len(cpus)}core' if cpus else ''}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "makespan_s", "gen_s", "status", "collisions", "iou"])
        for r in rows:
            w.writerow([r["model"], f"{r['makespan_s']:.3f}" if r["makespan_s"] else "", f"{r['gen_s']:.1f}",
                        r["status"], r["coll"], r["iou"]])
    print(f"(기록: {path})")
    shutil.rmtree(work / "_validator", ignore_errors=True)        # 이 스크립트가 만든 채점 결과 폴더만 정리
    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
