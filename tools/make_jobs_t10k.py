# -*- coding: utf-8 -*-
"""새 봉인 묶음 만들기(2026-11-28): Thingi10K 공개 모델(출처 표시·변경 허용 라이선스, 깨끗한 메시)을 WAAM 작업으로 바꾼다.
모델마다 (1) 바닥을 작업대에 놓고 (2) 가로 폭을 150~450 mm 중 하나로 키우되 높이 400 mm·부피 4.5e6 mm³(연습 2·3 과
같은 상한)를 넘지 않게 줄이고 (3) 작업대 가운데 근처에 놓은 뒤, 채점기 로더로 읽어 층마다 검사한다.
  - 모든 층이 비어 있지 않음(채점기 validator_accepts 와 같음), 층 10~250 개
  - 비드(폭 8 mm)로 넘치지 않게 채울 수 있는 넓이 99% 이상(연습 3 과 같은 기준)
  - 처마: 층마다 아래층을 2 mm 부풀린 범위 밖으로 나간 넓이가 그 층의 1% 이하(연습 2·3 의 한 층 2 mm 한도, 공중에 뜬 조각 없음)
통과한 모델 중 분류마다 최대 3개씩 시드 1128 순서로 21개를 고른다. 봉인 묶음이므로 모양을 보거나 풀어 보지 않는다.
usage: (WAAM venv) python -I t10k_build.py <t10k_data 폴더> <출력 폴더> [개수]"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"c:\projects\waam")
sys.path.insert(0, str(ROOT / "tools"))
from make_jobs_diverse import RMAX, SRC_CFG  # noqa: E402
from waam_validator.config.loader import load_config  # noqa: E402
from waam_validator.shape.target import load_target_mesh, slice_target_layers  # noqa: E402

RB, STEP, VCAP, FILL_MIN, H = 4.0, 2.0, 4.5e6, 0.99, 2.0
src, out = Path(sys.argv[1]), Path(sys.argv[2])
NEED = int(sys.argv[3]) if len(sys.argv) > 3 else 21
r = np.random.default_rng(1128)
rows = list(csv.DictReader(open(src / "pool.csv", encoding="utf-8")))
tmp = out.parent / (out.name + "_tmp")
tmp.mkdir(parents=True, exist_ok=True)
shutil.copyfile(SRC_CFG, tmp / "config.yaml")
cfg = load_config(tmp / "config.yaml")
table = Point(0, 0).buffer(RMAX * 0.98, 128)


def check(mesh_path):
    """채점기 로더로 읽고 층마다 검사. 통과하면 (층 수, 이유 없음), 아니면 (None, 이유)."""
    try:
        m = load_target_mesh(mesh_path, cfg)
    except Exception as e:                                           # noqa: BLE001
        return None, f"읽기 실패 {type(e).__name__}"
    n = int(np.ceil((m.bounds[1, 2] - cfg.process.build_plane_z_mm) / cfg.process.layer_height_mm - 1e-9))
    if not 10 <= n <= 250:
        return None, f"층 {n}"
    try:
        L = slice_target_layers(m, list(range(n)), cfg)
    except Exception as e:                                           # noqa: BLE001  채점기도 못 자르는 모델은 쓸 수 없다
        return None, f"자르기 실패 {type(e).__name__}"
    prev = None
    for k in range(n):
        g = L[k]
        if g.is_empty or g.area < 50:
            return None, f"빈 층 {k}"
        if not table.contains(g):
            return None, "작업대 밖"
        if g.buffer(-RB * 0.999, 8).buffer(RB * 0.999, 8).area < FILL_MIN * g.area:
            return None, f"가는 곳 층 {k}"
        if prev is not None and g.difference(prev.buffer(STEP, 8)).area > 0.01 * g.area:
            return None, f"처마 층 {k}"
        prev = g
    return n, ""


passed, why = [], {}
for i, row in enumerate(rows):
    f = src / "npz" / f"{row['ID']}.npz"
    if not f.exists():
        continue
    with np.load(f, allow_pickle=False) as z:
        v, fc = np.asarray(z["vertices"], float), np.asarray(z["facets"])
    if fc.ndim != 2 or fc.shape[1] != 3:
        why["삼각형 아님"] = why.get("삼각형 아님", 0) + 1
        continue
    mesh = trimesh.Trimesh(v, fc, process=True)
    if not mesh.is_watertight or mesh.volume <= 0:
        why["닫히지 않음"] = why.get("닫히지 않음", 0) + 1
        continue
    b = mesh.bounds
    w, h = float(max(b[1, 0] - b[0, 0], b[1, 1] - b[0, 1])), float(b[1, 2] - b[0, 2])
    s = r.uniform(150, 450) / w
    s = min(s, 400.0 / h, (VCAP / mesh.volume) ** (1 / 3))
    mesh.apply_translation([-(b[0, 0] + b[1, 0]) / 2, -(b[0, 1] + b[1, 1]) / 2, -b[0, 2]])
    mesh.apply_scale(s)
    w2 = w * s
    lim = max(0.0, RMAX * 0.9 - w2 / 2)
    mesh.apply_translation([r.uniform(-1, 1) * lim * 0.5, r.uniform(-1, 1) * lim * 0.5, cfg.process.build_plane_z_mm])
    p = tmp / "target.stl"
    mesh.export(p)
    n, reason = check(p)
    if n is None:
        key = reason.split()[0] if reason else "?"
        why[key] = why.get(key, 0) + 1
        continue
    passed.append({**row, "scale": round(s, 4), "layers": n, "width": round(w2), "height": round(h * s)})
    shutil.copyfile(p, tmp / f"ok_{row['ID']}.stl")
    if (i + 1) % 50 == 0:
        print(f"{i + 1}개 검사, 통과 {len(passed)}", flush=True)
print("검사 끝: 통과", len(passed), "| 탈락 이유별", why, flush=True)

order = list(r.permutation(len(passed)))
pick, per = [], {}
for j in order:
    c = passed[j].get("Category") or "기타"
    if per.get(c, 0) >= 3:
        continue
    pick.append(passed[j])
    per[c] = per.get(c, 0) + 1
    if len(pick) == NEED:
        break
out.mkdir(parents=True, exist_ok=True)
src_lines = ["# 봉인 묶음 출처(Thingi10K, 2026-11-28)", "", "모델은 아래 저작자가 Thingiverse 에 올린 것을 크기만 바꿔 썼다. 라이선스는 모델마다 다르다.", "",
             "| 작업 | file_id | 이름 | 저작자 | 라이선스 | 링크 |", "|---|---|---|---|---|---|"]
for k, it in enumerate(pick):
    d = out / f"{k + 1:02d}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC_CFG, d / "config.yaml")
    shutil.copyfile(tmp / f"ok_{it['ID']}.stl", d / "target.stl")
    (d / "desc.txt").write_text(f"thingi10k/{it['ID']} 배율{it['scale']} {it['layers']}층", encoding="utf-8")
    src_lines.append(f"| {k + 1:02d} | {it['ID']} | {it.get('Name', '')} | {it.get('Author', '')} | {it.get('License', '')} | {it.get('Link', '')} |")
(out / "SOURCES.md").write_text("\n".join(src_lines) + "\n", encoding="utf-8")
(out / "selection.json").write_text(json.dumps({"seed": 1128, "pool": len(rows), "passed": len(passed), "picked": len(pick),
                                                "why": why, "per_category": per}, ensure_ascii=False, indent=1), encoding="utf-8")
print("뽑음", len(pick), "분류별", per)
