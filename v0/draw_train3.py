# -*- coding: utf-8 -*-
"""학습 형상 v3(train_jobs3) 3D 그림 (기본 파이썬)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
R = Path(__file__).resolve().parents[1]
L = np.array([0.4, -0.6, 0.8]) / np.linalg.norm([0.4, -0.6, 0.8])
dirs = sorted(p for p in (R / "train_jobs3").iterdir() if (p / "target.stl").exists())
fig = plt.figure(figsize=(20, 12.5), dpi=110)
for n, d in enumerate(dirs):
    m = trimesh.load(d / "target.stl")
    ax = fig.add_subplot(3, 5, n + 1, projection="3d")
    tri = m.vertices[m.faces]
    sh = 0.3 + 0.7 * np.clip(m.face_normals @ L, 0, 1)
    col = [0.20, 0.33, 0.58] if n % 3 != 2 else [0.55, 0.40, 0.20]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(np.outer(sh, col) + 0.12, 0, 1), edgecolors="none"))
    lo, hi = m.bounds
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(30, -55); ax.set_axis_off()
    ext = hi - lo
    kind = "꽃병형(05 계열)" if n % 3 != 2 else "판+기둥형(04 계열)"
    ax.set_title(f"{d.name} {kind}\n{ext[0]:.0f}×{ext[1]:.0f}×{ext[2]:.0f}mm", fontsize=11)
fig.suptitle("학습 형상 v3 (15개) — 공개 05·04번을 닮은 비선형·무작위 형상", fontsize=16)
fig.tight_layout()
fig.savefig(R / "v0" / "shapes_train3.png", facecolor="white")
print("saved")
