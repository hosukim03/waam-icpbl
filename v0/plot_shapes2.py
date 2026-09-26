# -*- coding: utf-8 -*-
"""학습 형상 v2(train_jobs2)의 첫/중간/마지막 층 단면을 JSON 으로 저장 (WAAM venv)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

out = {}
for d in sorted(Path("../train_jobs2").iterdir()):
    if not (d / "target.stl").exists():
        continue
    cfg = load_config(d / "config.yaml")
    mesh = load_target_mesh(d / "target.stl", cfg)
    pr = cfg.process
    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    ks = [0, n // 2, n - 1]
    L = slice_target_layers(mesh, ks, cfg)
    polys = {}
    for k in ks:
        A = L[k]
        polys[str(k)] = [{"ext": np.asarray(g.exterior.coords).round(1).tolist(),
                          "holes": [np.asarray(r.coords).round(1).tolist() for r in g.interiors]}
                         for g in getattr(A, "geoms", [A]) if not A.is_empty]
    out[d.name] = {"n": n, "layers": polys}
json.dump(out, open("../v0/shapes2.json", "w"), separators=(",", ":"))
print("ok", len(out))
