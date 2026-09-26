# -*- coding: utf-8 -*-
"""공개 05번 형상 분석: 층별 면적·조각 수·구멍 수·크기, 여러 높이의 단면 좌표 저장."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

sys.stdout.reconfigure(encoding="utf-8")
d = Path("tests/05")
cfg = load_config(d / "config.yaml")
mesh = load_target_mesh(d / "target.stl", cfg)
pr = cfg.process
n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
print("bounds", mesh.bounds.round(1).tolist(), "layers", n, "faces", len(mesh.faces))
ks = list(range(0, n, 10)) + [n - 1]
L = slice_target_layers(mesh, ks, cfg)
out = {}
for k in ks:
    A = L[k]
    gs = list(getattr(A, "geoms", [A]))
    holes = sum(len(g.interiors) for g in gs)
    x0, y0, x1, y1 = A.bounds
    # 벽 두께 추정: 면적 / (둘레/2)
    wall = 2 * A.area / A.length if A.length else 0
    print(f"층{k:3d} 면적 {A.area / 100:6.0f}cm2 조각 {len(gs)} 구멍 {holes:2d} "
          f"크기 {x1 - x0:4.0f}x{y1 - y0:4.0f} 둘레 {A.length:6.0f} 평균벽두께 {wall:5.1f}")
    out[k] = [{"ext": np.asarray(g.exterior.coords).round(1).tolist(),
               "holes": [np.asarray(r.coords).round(1).tolist() for r in g.interiors]} for g in gs]
json.dump(out, open("../v0/shape05.json", "w"), separators=(",", ":"))
