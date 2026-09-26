# -*- coding: utf-8 -*-
"""형상 묶음 하나를 특정 스케줄러로 전부 돌리고 채점해 표(CSV)로 남긴다.

usage: python eval_set.py <스케줄러.py> <형상폴더> <결과이름>
  예) python eval_set.py schedule_v8.py ../valid_div v8_valid
결과: v0/results/<결과이름>.csv  (job, 난이도, 설명, 판정, makespan_s, 생성시간_s, IoU, 충돌수, 실패사유)
기본 파이썬으로 실행 (스케줄러·채점기는 WAAM venv 파이썬으로 호출)
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VPY = ROOT / "WAAM_Validator" / ".venv" / "Scripts" / "python.exe"
sched, src, name = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3]
jobs_dir = ROOT / "jobs_eval" / name
runs_dir = ROOT / "runs_eval" / name
out_csv = ROOT / "v0" / "results" / f"{name}.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
env = {"PYTHONIOENCODING": "utf-8", **__import__("os").environ}

rows = []
for d in sorted(p for p in src.iterdir() if (p / "target.stl").exists()):
    desc = (d / "desc.txt").read_text(encoding="utf-8") if (d / "desc.txt").exists() else ""
    level = desc.split()[0] if desc else ""
    job = jobs_dir / d.name
    t0 = time.time()
    if sched == "portfolio.py":          # 여러 방식을 동시에 돌려 가장 짧은 것 선택 (기본 파이썬)
        g = subprocess.run([sys.executable, str(ROOT / "v0" / "portfolio.py"), str(d), str(job)],
                           capture_output=True, text=True, encoding="utf-8", env=env,
                           cwd=ROOT / "v0")
    else:
        g = subprocess.run([str(VPY), str(ROOT / "v0" / sched), str(d), str(job)],
                           capture_output=True, text=True, encoding="utf-8", env=env,
                           cwd=ROOT / "WAAM_Validator")
    gen = time.time() - t0
    row = {"job": d.name, "level": level, "desc": desc, "status": "GEN_FAIL",
           "makespan_s": "", "gen_s": f"{gen:.1f}", "iou": "", "collisions": "", "reasons": ""}
    if g.returncode == 0:
        run = runs_dir / d.name
        if run.exists():
            import shutil
            shutil.rmtree(run)                          # 이 스크립트가 만든 이전 채점 결과만 비움
        v = subprocess.run([str(VPY), "-m", "waam_validator.cli", "run", str(job), "--output", str(run), "--json"],
                           capture_output=True, text=True, encoding="utf-8", env=env,
                           cwd=ROOT / "WAAM_Validator")
        try:
            s = json.loads(v.stdout)
            row.update(status=s["status"], makespan_s=f"{s['schedule']['makespan_s']:.1f}",
                       iou=f"{s['shape']['iou']:.3f}", collisions=s["collision"]["collision_event_count"],
                       reasons="; ".join(s["failure_reasons"])[:200])
        except Exception:
            row.update(status=f"VAL_ERR{v.returncode}", reasons=(v.stderr or "")[-200:])
    else:
        row["reasons"] = (g.stderr or "")[-200:].replace("\n", " ")
    rows.append(row)
    print(f"{name} {d.name} [{level}] {row['status']} makespan={row['makespan_s']} gen={row['gen_s']}s "
          f"coll={row['collisions']} {row['reasons'][:80]}", flush=True)

with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{name} 완료 → {out_csv}", flush=True)
