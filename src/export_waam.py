# -*- coding: utf-8 -*-
"""Exporter — 시나리오 폴더를 받아 제출용 작업 폴더(trajectory.csv + config.yaml + target.stl)를 만든다.

저장소 문서 '10. Exporter 납품 인터페이스' 의 권장 형태를 따른다.
우리 알고리즘은 할당·순서만 내놓는 것이 아니라 시간이 지정된 TCP 경로까지 직접 만들므로
별도의 algorithm_result.json 없이 시나리오만으로 동작한다(있으면 --result 로 설정을 덮어쓸 수 있다).

  python export_waam.py --scenario tests/01 --output out/01
  python export_waam.py --scenario tests/01 --output out/01 --variants r1c160,r3c160 --no-check

동작
  1) 시나리오의 config.yaml·target.stl 을 읽는다.
  2) 포트폴리오(여러 스케줄링 설정을 코어별 병렬 실행)를 돌려 makespan 이 가장 짧은 결과를 고른다.
  3) 고정 스키마 trajectory.csv 를 쓰고, config.yaml·target.stl 을 그대로 복사한다.
  4) 안전망: 채점기 전체 검증(run)을 돌려 FAIL 이면 makespan 순으로 다음 설정 결과를 대신 쓴다.
     (충돌 여유를 줄인 m5 설정처럼 공격적인 설정이 혹시 실패해도 제출물은 항상 PASS 상태로 남긴다)
  5) waam-validator check 로 입력 인터페이스 검사를 돌리고 결과를 알린다(--no-check 로 생략).
종료 코드: 0 = 성공(검사까지 통과), 1 = 생성 실패, 2 = check 실패, 3 = 모든 설정이 채점기 FAIL
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VPY = ROOT / "WAAM_Validator" / ".venv" / "Scripts" / "python.exe"
HEADER = ["robot_id", "time_s", "x_mm", "y_mm", "z_mm", "mode"]


def makespan_of(p: Path) -> float:
    with open(p, newline="", encoding="utf-8") as f:
        return max(float(r["time_s"]) for r in csv.DictReader(f))


def grade(job_dir: Path, out_dir: Path) -> tuple[bool, str]:
    """채점기 전체 검증. (PASS 여부, 한 줄 요약)"""
    shutil.rmtree(out_dir, ignore_errors=True)
    r = subprocess.run([str(VPY), "-m", "waam_validator.cli", "run", str(job_dir), "--output", str(out_dir), "--json"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=ROOT / "WAAM_Validator")
    try:
        d = json.loads(r.stdout)
    except Exception:
        return False, f"채점기 실행 실패: {(r.stderr or '')[-160:]}"
    ok = d.get("status") == "PASS"
    c = d.get("collision", {})
    msg = (f"{d.get('status')} 충돌 {c.get('collision_event_count')} IoU {d.get('shape', {}).get('iou', 0):.4f}"
           + (f" 사유 {d.get('failure_reasons')}" if not ok else ""))
    return ok, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="config.yaml·target.stl 이 있는 시나리오 폴더")
    ap.add_argument("--output", required=True, help="제출용 작업 폴더(없으면 만든다)")
    ap.add_argument("--result", default=None, help="선택: 설정·워커 수를 담은 JSON")
    ap.add_argument("--variants", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-check", dest="check", action="store_false", help="waam-validator check 생략")
    ap.add_argument("--no-grade", dest="grade", action="store_false", help="채점기 전체 검증·대체 생략")
    a = ap.parse_args()

    # 하위 프로세스가 작업 위치를 채점기 폴더로 바꿔 실행하므로, 상대 경로는 여기서 절대 경로로 바꾼다
    scen, out = Path(a.scenario).resolve(), Path(a.output).resolve()
    for f in ("config.yaml", "target.stl"):
        if not (scen / f).exists():
            print(f"[오류] 시나리오에 {f} 가 없다: {scen}", file=sys.stderr)
            return 1
    opt = json.loads(Path(a.result).read_text(encoding="utf-8")) if a.result else {}
    variants = a.variants or opt.get("variants")
    workers = a.workers or int(opt.get("workers", 0))

    cmd = [str(VPY), str(ROOT / "src" / "portfolio.py"), str(scen), str(out), "--keep"]
    if variants:
        cmd += ["--variants", variants]
    if workers:
        cmd += ["--workers", str(workers)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT / "WAAM_Validator")
    traj = out / "trajectory.csv"
    if p.returncode != 0 or not traj.exists():
        print("[오류] 궤적 생성 실패", file=sys.stderr)
        return 1
    with open(traj, newline="", encoding="utf-8") as f:
        head = next(csv.reader(f))
    if head != HEADER:
        print(f"[오류] CSV 헤더가 규격과 다르다: {head}", file=sys.stderr)
        return 1
    gen = time.time() - t0
    print(f"생성 완료: {out}  makespan={makespan_of(traj):.1f}s ({makespan_of(traj)/3600:.2f}h)  생성시간={gen:.1f}s")

    work = Path(str(out) + "_variants")
    if a.grade and (out / "portfolio.txt").exists() and work.exists():
        # makespan 순으로 설정을 훑어 채점기 PASS 인 첫 결과를 제출물로 삼는다
        rows = []
        for line in (out / "portfolio.txt").read_text(encoding="utf-8").splitlines():
            f = line.split("\t")
            if len(f) == 3 and f[1] and (work / f[0] / "trajectory.csv").exists():
                rows.append((float(f[1]), f[0]))
        rows.sort()
        chosen = None
        for mk, name in rows:
            ok, msg = grade(work / name, work / f"_grade_{name}")
            print(f"  검증 {name:14s} {mk:9.1f}s → {msg}")
            if ok:
                chosen = name
                break
        if chosen is None:
            print("[오류] 모든 설정이 채점기 FAIL", file=sys.stderr)
            return 3
        for f in ("trajectory.csv", "config.yaml", "target.stl"):
            shutil.copyfile(work / chosen / f, out / f)
        print(f"제출 선택: {chosen}  makespan={makespan_of(out / 'trajectory.csv'):.1f}s  (채점기 PASS 확인)")
        shutil.rmtree(work, ignore_errors=True)

    if not a.check:
        return 0
    c = subprocess.run([str(VPY), "-m", "waam_validator.cli", "check", str(out)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=ROOT / "WAAM_Validator")
    print(c.stdout.strip() or c.stderr.strip())
    if c.returncode != 0:
        print("[오류] 입력 인터페이스 검사 실패", file=sys.stderr)
        return 2
    print("입력 인터페이스 검사 통과 (exit 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
