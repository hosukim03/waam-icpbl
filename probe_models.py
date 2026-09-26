import numpy as np
from pathlib import Path
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh, slice_target_layers
for m in ["01","02","03","04","05","06"]:
    job = Path(f"tests/{m}")
    cfg = load_config(job / "config.yaml")
    mesh = load_target_mesh(job / "target.stl", cfg)[0] if isinstance(load_target_mesh(job / "target.stl", cfg), tuple) else load_target_mesh(job / "target.stl", cfg)
    zmax = mesh.bounds[1, 2]
    n = int(np.ceil((zmax - cfg.process.build_plane_z_mm) / cfg.process.layer_height_mm - 1e-9))
    idx = sorted(set([0, n // 2, n - 1]))
    L = slice_target_layers(mesh, idx, cfg)
    print(f"\n[{m}] layers={n} xy={np.round(mesh.bounds[:, :2], 0).tolist()}")
    for k in idx:
        P = L[k]
        if P.is_empty: print(f"  L{k}: empty"); continue
        geoms = list(getattr(P, "geoms", [P]))
        holes = sum(len(g.interiors) for g in geoms)
        thick = 2 * P.area / P.length
        inner = P.buffer(-4.0)
        print(f"  L{k}: polys={len(geoms)} holes={holes} area={P.area:.0f} perim={P.length:.0f} ~wall={thick:.1f}mm  after_-4mm_area={inner.area:.0f}")
