# -*- coding: utf-8 -*-
"""층마다 단면이 얼마나 바뀌는지: 층별 면적과 조각 수의 변화를 본다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

sys.stdout.reconfigure(encoding="utf-8")
print(f"{'형상':12s} {'층수':>4s} {'면적 최소~최대(cm2)':>22s} {'변화폭':>7s} {'조각수':>8s}")
for base, tag in [(Path("tests"), "public"), (Path("../bench/tests"), "bench"), (Path("../train_jobs"), "train")]:
    for d in sorted(base.iterdir()):
        if not (d / "target.stl").exists():
            continue
        cfg = load_config(d / "config.yaml")
        mesh = load_target_mesh(d / "target.stl", cfg)
        pr = cfg.process
        n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
        ks = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
        L = slice_target_layers(mesh, ks, cfg)
        A = list(L.values()) if isinstance(L, dict) else [L[k] for k in ks]
        ar = [a.area / 100 for a in A if not a.is_empty]
        ng = [len(getattr(a, "geoms", [a])) for a in A if not a.is_empty]
        if not ar:
            continue
        chg = (max(ar) - min(ar)) / max(ar)
        print(f"{tag}/{d.name:6s} {n:4d} {min(ar):9.0f} ~ {max(ar):9.0f} {chg:6.0%} {min(ng)}~{max(ng)}")
