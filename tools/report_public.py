# -*- coding: utf-8 -*-
"""공개 모델의 makespan 과 계산 시간을 한 화면 표로 보여 준다(보고·캡처용).

형상마다
  1) 우리 알고리즘(src/portfolio.py)으로 trajectory.csv 를 만들며 시계 시간을 잰다.
     = 계산 시간: 형상을 넣은 순간부터 경로 파일이 나올 때까지. 실행 중 학습은 없다.
  2) 교수님 채점기(waam-validator run)로 채점해 판정과 makespan 을 받는다.
     표의 makespan 은 채점기가 낸 값이다(채점에 걸린 시간은 계산 시간에 넣지 않는다).
마지막에 표와 합계를 출력하고 results/report_public_<날짜시각>.csv 로 남긴다.

usage: python tools/report_public.py [--full] [--models 01,02]   (WAAM venv 파이썬)
  --full    형상을 보고 고르지 않고 설정 12개를 전부 돌린다(예전 방식)
  --models  일부 형상만(기본 01~06 전부)
"""
import argparse
import csv
import json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--models", default="01,02,03,04,05,06")
    a = ap.parse_args()
    models = a.models.split(",")
    stamp = datetime.now()
    work = ROOT / "jobs_report" / stamp.strftime("%Y%m%d_%H%M%S")
    mode = "설정 12개 전부(--full)" if a.full else "형상을 보고 설정 고르기"
    print(f"공개 모델 {len(models)}개 실행 시작 ({mode}) …", flush=True)

    rows = []
    for m in models:
        out = work / m
        t0 = time.perf_counter()
        g = subprocess.run([str(VPY), str(ROOT / "src" / "portfolio.py"), str(TESTS / m), str(out),
                            *(["--full"] if a.full else [])],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT / "src")
        gen = time.perf_counter() - t0
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

    path = ROOT / "results" / f"report_public_{stamp:%Y%m%d_%H%M}.csv"
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
