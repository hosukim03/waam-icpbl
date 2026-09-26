"""경로 생성 비교: legacy(IoU 최대) vs new(기준 만족 중 최단 시간). 모델별 전 층."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers

from toolpath import best_paths, est_time

ROOT = Path.home() / "waam_icpbl/WAAM_Validator/tests"
jobs = sys.argv[1:] or ["01", "02", "03", "04", "05", "06"]
for m in jobs:
    job = Path(m) if Path(m).is_dir() else ROOT / m
    cfg = load_config(job / "config.yaml")
    mesh = load_target_mesh(job / "target.stl", cfg)
    n = int(np.ceil(mesh.bounds[1, 2] / cfg.process.layer_height_mm - 1e-9))
    idx = list(range(n)) if n <= 60 else list(range(0, n, 10))
    L = slice_target_layers(mesh, idx, cfg)
    res = {}
    for mode in ("legacy", "new"):
        t0 = time.time()
        ious, T, lens, pcs, names = [], 0.0, 0.0, 0, {}
        for k in idx:
            if L[k].is_empty:
                continue
            P, sc, name = best_paths(L[k], legacy=(mode == "legacy"))
            ious.append(sc["iou"]); T += est_time(sc); lens += sc["length"]; pcs += sc["pieces"]
            names[name] = names.get(name, 0) + 1
        res[mode] = (min(ious), float(np.mean(ious)), T, lens, pcs, time.time() - t0, names)
    lg, nw = res["legacy"], res["new"]
    print(f"[{job.name}] layers_eval={len(idx)}")
    for tag, r in (("legacy", lg), ("new   ", nw)):
        print(f"   {tag}: IoU min {r[0]:.3f} mean {r[1]:.3f} | est deposit+arc {r[2]:.0f}s | length {r[3]/1000:.1f}m pieces {r[4]} | path-gen {r[5]:.1f}s | {r[6]}")
    print(f"   -> time {100*(nw[2]-lg[2])/lg[2]:+.1f}%  path-gen {nw[5]-lg[5]:+.1f}s")
