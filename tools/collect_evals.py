# -*- coding: utf-8 -*-
"""평가 결과에서 대시보드(dashboard/)가 쓰는 작은 파일만 runs/ 로 모은다(WAAM venv 파이썬).

eval  : 묶음 평가 결과 폴더(형상마다 solve.txt, _val/<형상>/summary.json·layer_metrics.csv)를 runs/evals/<이름>/ 로
        복사하고 runs/evals/index.json 에 이름·묶음·설명을 적는다. 경로(trajectory.csv)는 커서 뺀다.
        --bound 를 주면 묶음의 '이론상 최소 시간'을 형상마다 계산해 runs/evals/bounds/<묶음 폴더 이름>.json 에 둔다.
timing: 코어 1개 계산 시간 측정 로그(형상 방법 벽시계 … 꼴)를 runs/timing/<이름>.log 로 복사하고 index.json 에 적는다.

usage: python tools/collect_evals.py eval <결과 폴더> <이름> <묶음 폴더> "<설명>" [--log <요약 로그>] [--bound]
       python tools/collect_evals.py timing <로그 파일> <이름> <묶음 폴더> "<설명>"
"""
import argparse
import datetime
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def _index(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _put(path: Path, entry: dict) -> None:
    rows = [r for r in _index(path) if r["name"] != entry["name"]]
    rows.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def bounds(set_dir: Path) -> dict:
    """형상마다 한 대 시간(0.98 × 넓이 / 비드 폭 / 8.8 mm/s 를 층마다 더함)과 이론상 최소 시간
    (층의 가장 먼 두 점 거리가 200 mm 이하면 한 대만, 아니면 세 대가 나눠 쌓는다고 봄)."""
    from waam_validator.config.loader import load_config
    from waam_validator.shape.target import load_target_mesh, slice_target_layers
    out = {}
    for job in sorted(p for p in set_dir.iterdir() if (p / "target.stl").exists()):
        cfg = load_config(job / "config.yaml")
        mesh = load_target_mesh(job / "target.stl", cfg)
        pr = cfg.process
        n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
        L = slice_target_layers(mesh, list(range(n)), cfg)
        t1 = lb = 0.0
        for k in range(n):
            tk = 0.98 * L[k].area / pr.bead_width_mm / 8.8
            hull = L[k].convex_hull
            h = np.asarray(hull.exterior.coords) if hull.geom_type == "Polygon" else np.zeros((1, 2))
            t1 += tk
            lb += tk if np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max() <= 200.0 else tk / 3
        b = mesh.bounds
        out[job.name] = {"layers": n, "one_robot_s": t1, "bound_s": lb,
                         "size_mm": [round(float(b[1, i] - b[0, i])) for i in range(3)]}
        print(" 이론상 최소", job.name, f"{lb:.0f} s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["eval", "timing"])
    ap.add_argument("src")
    ap.add_argument("name")
    ap.add_argument("set_dir")
    ap.add_argument("desc")
    ap.add_argument("--log", default=None)
    ap.add_argument("--bound", action="store_true")
    a = ap.parse_args()
    src, set_dir = Path(a.src), Path(a.set_dir)
    rel = set_dir.resolve().relative_to(ROOT).as_posix() if set_dir.resolve().is_relative_to(ROOT) else str(set_dir)
    entry = {"name": a.name, "set_dir": rel, "desc": a.desc, "date": datetime.date.today().isoformat(), "source": str(src)}
    if a.kind == "timing":
        dst = RUNS / "timing" / f"{a.name}.log"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        _put(RUNS / "timing" / "index.json", entry)
        print("모음", dst)
        return
    dst = RUNS / "evals" / a.name
    if dst.exists():
        shutil.rmtree(dst)
    n = 0
    for job in sorted(p for p in src.iterdir() if p.is_dir() and (p / "solve.txt").exists()):
        d = dst / job.name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(job / "solve.txt", d / "solve.txt")
        for f in ("summary.json", "layer_metrics.csv", "robot_metrics.csv"):
            if (src / "_val" / job.name / f).exists():
                shutil.copyfile(src / "_val" / job.name / f, d / f)
        n += 1
    if a.log and Path(a.log).exists():
        shutil.copyfile(a.log, dst / "eval.log")
    entry["shapes"] = n
    _put(RUNS / "evals" / "index.json", entry)
    print("모음", dst, n, "개 형상")
    if a.bound:
        bp = RUNS / "evals" / "bounds" / f"{set_dir.name}.json"
        if not bp.exists():
            bp.parent.mkdir(parents=True, exist_ok=True)
            bp.write_text(json.dumps(bounds(set_dir), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
