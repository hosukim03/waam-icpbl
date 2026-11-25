# -*- coding: utf-8 -*-
"""대시보드 '층별 점검'의 형상 그림 자료: 채점기와 같은 방법으로 층마다 목표 단면·쌓인 면·빈틈·넘침을 만든다(WAAM venv 파이썬).

runs/evals/<실험>/<묶음>/<형상>/ 에 궤적(trajectory.csv.gz, collect_evals --traj)이 있어야 한다. 결과는 같은 폴더의
layer_shapes.zip 이다. 채점기 함수(목표 STL 자르기, 쌓은 구간을 비드 폭으로 부풀려 합치기, 정리)를 그대로 불러 쓰므로
층마다 겹침 점수가 채점기의 layer_metrics.csv 와 같다(다르면 멈춘다).

zip 안:
  summary.json  층마다 높이·점수·넓이·큰 조각 목록, 3D 용 빈틈·넘침 삼각형(1 mm² 넘는 조각), 기준 미달 층의 목표 테두리
  L0000.json …  층 하나의 단면 그림: 목표 단면, 쌓인 면, 빈틈·넘침 조각(0.3 mm² 넘는 것), 불 켜고 지나간 길
대시보드는 summary 와 고른 층 하나만 읽는다(층이 350개인 공개 05 도 가볍게).

usage: python tools/shape_diff.py <실험 이름> <묶음 폴더> [형상 ...] [--force]
       예) python tools/shape_diff.py 1110a_ls2_adapt뺌 data/novel1027 14
       형상을 안 주면 그 묶음에서 궤적이 있는 형상 전부(이미 만든 것은 건너뜀).
"""
import argparse
import csv
import gzip
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "WAAM_Validator" / "src"))
from waam_validator.config.loader import load_config  # noqa: E402
from waam_validator.constants import MODE_D  # noqa: E402
from waam_validator.shape.deposition import build_deposited_layers  # noqa: E402
from waam_validator.shape.layer_index import determine_layer_index, layer_bounds  # noqa: E402
from waam_validator.shape.polygon_utils import normalize_polygon, polygon_components  # noqa: E402
from waam_validator.shape.target import determine_evaluation_layers, load_target_mesh, slice_target_layers  # noqa: E402
from waam_validator.trajectory.loader import load_trajectory_csv  # noqa: E402

OUT_NAME = "layer_shapes.zip"
MIN_2D = 0.3           # 단면 그림에 넣는 빈틈·넘침 조각의 최소 넓이(mm²). 넓이 합은 모든 조각으로 센다
MIN_3D = 1.0           # 3D 에 넣는 조각의 최소 넓이(mm²)
TOP = 12               # 층마다 큰 조각 목록 길이


def rings(geom, tol: float, min_area: float = 0.0) -> list:
    """다각형들 → [[바깥 고리, 구멍 고리, ...], ...], 고리는 [x0, y0, x1, y1, ...] (0.01 mm 반올림)."""
    out = []
    for p in polygon_components(geom):
        if p.area < min_area:
            continue
        if tol > 0:
            q = p.simplify(tol, preserve_topology=True)
            p = q if q.geom_type == "Polygon" and not q.is_empty else p
        rs = [p.exterior] + list(p.interiors)
        out.append([np.round(np.asarray(r.coords)[:-1], 2).ravel().tolist() for r in rs])
    return out


def pieces(geom, kind: str) -> tuple[list, list]:
    """3D 용 삼각형([x,y]×3 를 펼친 6개씩, 1 mm² 넘는 조각)과 큰 조각 목록(넓이·가운데·범위·대략 굵기)."""
    tris, info = [], []
    for p in polygon_components(geom):
        if p.area < MIN_2D:
            continue
        x0, y0, x1, y1 = p.bounds
        info.append({"kind": kind, "area": round(p.area, 2), "cx": round(p.centroid.x, 1), "cy": round(p.centroid.y, 1),
                     "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                     "w": round(2 * p.area / max(p.length, 1e-9), 2)})
        if p.area >= MIN_3D:
            q = p.simplify(0.05, preserve_topology=True)
            q = q if q.geom_type == "Polygon" and not q.is_empty else p
            for t in shapely.constrained_delaunay_triangles(q).geoms:
                tris.append(np.round(np.asarray(t.exterior.coords)[:3], 2).ravel().tolist())
    return tris, info


def torch_paths(traj, cfg) -> dict:
    """층마다 불 켜고(D) 지나간 길: 이어진 구간은 한 줄로, 줄마다 [x0, y0, x1, y1, ...]."""
    out: dict[int, list] = {}
    for tr in traj.robots:
        xyz, m = tr.xyz_mm.astype(float), np.asarray(tr.mode)
        cur, cur_l = None, None
        for i in range(len(m) - 1):
            if int(m[i]) != int(MODE_D):
                cur = None
                continue
            a, b = xyz[i], xyz[i + 1]
            L = determine_layer_index(float((a[2] + b[2]) / 2), cfg)
            if cur is None or cur_l != L or cur[-2:] != [round(a[0], 2), round(a[1], 2)]:
                cur = [round(a[0], 2), round(a[1], 2)]
                cur_l = L
                out.setdefault(L, []).append(cur)
            cur += [round(b[0], 2), round(b[1], 2)]
    return out


def one(name: str, set_dir: str, shape: str) -> str:
    ev = ROOT / "runs" / "evals" / name / shape
    src = ROOT / set_dir / shape
    cfg = load_config(ev / "config.yaml" if (ev / "config.yaml").exists() else src / "config.yaml")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        csv_p = Path(tmp) / "trajectory.csv"
        with gzip.open(ev / "trajectory.csv.gz", "rb") as f, open(csv_p, "wb") as g:
            shutil.copyfileobj(f, g)
        traj = load_trajectory_csv(csv_p, cfg)
    mesh = load_target_mesh(src / "target.stl", cfg)
    dep = build_deposited_layers(traj, cfg)
    order = determine_evaluation_layers(mesh, dep, cfg)
    tgt = slice_target_layers(mesh, order, cfg)
    sv = cfg.shape_validation
    paths = torch_paths(traj, cfg)
    ref = {}
    lm = ev / "layer_metrics.csv"
    if lm.exists():
        with open(lm, encoding="utf-8") as f:
            ref = {int(r["layer_index"]): float(r["iou"]) for r in csv.DictReader(f)}
    summary, files = [], {}
    for L in order:
        t = normalize_polygon(tgt.get(L, shapely.Polygon()), sv.polygon_snap_tolerance_mm, sv.area_epsilon_mm2)
        d = normalize_polygon(dep.get(L, shapely.Polygon()), sv.polygon_snap_tolerance_mm, sv.area_epsilon_mm2)
        ta, da = float(t.area), float(d.area)
        if ta <= sv.area_epsilon_mm2 and da <= sv.area_epsilon_mm2:
            continue
        under, over = t.difference(d), d.difference(t)
        inter = float(t.intersection(d).area)
        iou = 0.0 if ta <= sv.area_epsilon_mm2 or da <= sv.area_epsilon_mm2 else min(1.0, max(0.0, inter / (ta + da - inter)))
        if L in ref and abs(ref[L] - iou) > 1e-6:
            raise SystemExit(f"{shape} {L}층 겹침 점수가 채점기와 다름: {iou} vs {ref[L]}")
        z0, z1, _ = layer_bounds(L, cfg)
        passed = iou >= sv.minimum_layer_iou
        ut, ui = pieces(under, "under")
        ot, oi = pieces(over, "over")
        summary.append({"L": L, "z0": z0, "z1": z1, "iou": iou, "passed": passed,
                        "t_area": round(ta, 2), "d_area": round(da, 2),
                        "under_area": round(float(under.area), 2), "over_area": round(float(over.area), 2),
                        "big": sorted(ui + oi, key=lambda b: -b["area"])[:TOP], "under_tri": ut, "over_tri": ot,
                        "outline": [] if passed else rings(t, 0.2)})
        files[f"L{L:04d}.json"] = {"L": L, "target": rings(t, 0.01), "dep": rings(d, 0.05),
                                   "under": rings(under, 0.01, MIN_2D), "over": rings(over, 0.01, MIN_2D),
                                   "path": paths.get(L, [])}
    head = {"version": 2, "shape": shape, "set_dir": set_dir, "bead_mm": cfg.process.bead_width_mm,
            "layer_h": cfg.process.layer_height_mm, "iou_min": sv.minimum_layer_iou,
            "secs": round(time.perf_counter() - t0, 1), "layers": summary}
    tmp = ev / (OUT_NAME + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("summary.json", json.dumps(head, separators=(",", ":")))
        for k, v in files.items():
            z.writestr(k, json.dumps(v, separators=(",", ":")))
    tmp.replace(ev / OUT_NAME)
    return f"{shape}: {len(summary)}층, {head['secs']} s, 기준 미달 {sum(not x['passed'] for x in summary)}층"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="실험 이름(예: 1110a_ls2_adapt뺌) 또는 index.json 의 이름(실험/묶음)")
    ap.add_argument("set_dir")
    ap.add_argument("shapes", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    name = a.name if "/" in a.name else f"{a.name}/{Path(a.set_dir).name}"
    base = ROOT / "runs" / "evals" / name
    shapes = a.shapes or sorted(p.name for p in base.iterdir() if (p / "trajectory.csv.gz").exists())
    for s in shapes:
        if not a.force and (base / s / OUT_NAME).exists():
            print(f"{s}: 이미 있음", flush=True)
            continue
        print(one(name, a.set_dir, s), flush=True)


if __name__ == "__main__":
    main()
