# -*- coding: utf-8 -*-
"""형상 묶음(공개/벤치마크/학습)의 첫 층 단면을 좌표로 뽑아 JSON 저장. (WAAM venv 에서 실행)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

out = {}
for label, base in [("bench", Path("../data/bench/tests")), ("train", Path("../data/train_jobs"))]:
    for d in sorted(base.iterdir()):
        if not (d / "target.stl").exists():
            continue
        cfg = load_config(d / "config.yaml")
        mesh = load_target_mesh(d / "target.stl", cfg)
        A = slice_target_layers(mesh, [0], cfg)[0]
        polys = []
        for g in getattr(A, "geoms", [A]):
            polys.append({"ext": np.asarray(g.exterior.coords).round(1).tolist(),
                          "holes": [np.asarray(r.coords).round(1).tolist() for r in g.interiors]})
        out[f"{label}/{d.name}"] = polys
        print(label, d.name, len(polys), "조각")
json.dump(out, open("../v0/shapes.json", "w"), separators=(",", ":"))
