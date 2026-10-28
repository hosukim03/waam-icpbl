# -*- coding: utf-8 -*-
"""Exporter — 시나리오 폴더를 받아 제출용 작업 폴더(trajectory.csv + config.yaml + target.stl)를 만든다.

저장소 문서 '10. Exporter 납품 인터페이스' 의 권장 형태를 따른다.
우리 알고리즘은 할당·순서만 내놓는 것이 아니라 시간이 지정된 TCP 경로까지 직접 만들므로
별도의 algorithm_result.json 없이 시나리오만으로 동작한다(있으면 --result 로 설정을 덮어쓸 수 있다).

  python export_waam.py --scenario tests/01 --output out/01
  python export_waam.py --scenario tests/01 --output out/01 --variants sp_r1c160,sp_r2c160m5a --no-check

동작
  1) 시나리오의 config.yaml·target.stl 을 읽는다.
  2) solve.py(설정 후보를 코어 하나로 차례로 계획해 makespan 이 가장 짧은 것)로 궤적을 만든다.
  3) 고정 스키마 trajectory.csv 를 쓰고, config.yaml·target.stl 을 그대로 복사한다.
  4) 안전망: 채점기 전체 검증(run)을 돌려 FAIL 이면 그 설정을 빼고 다시 풀어 다음으로 좋은 결과를 쓴다.
     (충돌 여유를 줄인 m5 설정처럼 공격적인 설정이 혹시 실패해도 제출물은 PASS 인 결과로 남긴다)
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
    ap.add_argument("--result", default=None, help="선택: 설정 후보(variants)를 담은 JSON")
    ap.add_argument("--variants", default=None, help="설정 후보를 직접 지정(쉼표로 구분). 생략하면 형상 특징으로 고른다")
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

    def solve(names):
        """solve.py 로 풀고 (고른 설정, 후보 목록)을 돌려준다. 실패하면 None."""
        cmd = [str(VPY), str(ROOT / "src" / "solve.py"), str(scen), str(out)]
        if names:
            cmd += ["--variants", ",".join(names) if isinstance(names, list) else names]
        p = subprocess.run(cmd, cwd=ROOT / "WAAM_Validator")
        traj = out / "trajectory.csv"
        if p.returncode != 0 or not traj.exists():
            return None
        info = {"names": []}
        for line in (out / "solve.txt").read_text(encoding="utf-8").splitlines():
            f = line.split("\t")
            if f[0] == "best":
                info["best"] = f[1]
            elif f[0] not in ("features", "sample", "time", "ls"):
                info["names"].append(f[0])
        return info

    t0 = time.time()
    info = solve(variants)
    traj = out / "trajectory.csv"
    if info is None:
        print("[오류] 궤적 생성 실패", file=sys.stderr)
        return 1
    with open(traj, newline="", encoding="utf-8") as f:
        head = next(csv.reader(f))
    if head != HEADER:
        print(f"[오류] CSV 헤더가 규격과 다르다: {head}", file=sys.stderr)
        return 1
    gen = time.time() - t0
    print(f"생성 완료: {out}  설정 {info['best']}  makespan={makespan_of(traj):.1f}s ({makespan_of(traj)/3600:.2f}h)  "
          f"생성시간={gen:.1f}s")

    if a.grade:
        # 채점기가 FAIL 이면 그 설정을 빼고 남은 후보로 다시 푼다(PASS 가 나올 때까지)
        failed = []
        while True:
            ok, msg = grade(out, Path(str(out) + "_grade"))
            print(f"  검증 {info['best']:18s} {makespan_of(out / 'trajectory.csv'):9.1f}s → {msg}")
            if ok:
                break
            failed.append(info["best"])
            rest = [nm for nm in info["names"] if nm not in failed]
            if not rest:
                print("[오류] 모든 설정이 채점기 FAIL", file=sys.stderr)
                return 3
            info2 = solve(rest)
            if info2 is None:
                print("[오류] 궤적 생성 실패", file=sys.stderr)
                return 1
            info = {"best": info2["best"], "names": info["names"]}
        shutil.rmtree(Path(str(out) + "_grade"), ignore_errors=True)
        print(f"제출 선택: {info['best']}  makespan={makespan_of(out / 'trajectory.csv'):.1f}s  (채점기 PASS 확인)")

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
