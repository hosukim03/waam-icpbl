# -*- coding: utf-8 -*-
"""공개 모델 6개 target.stl 3D 그림 (기본 파이썬)."""
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
INFO = {"01": ("6층", "0.39h"), "02": ("8층", "0.61h"), "03": ("6층", "0.58h"),
        "04": ("50층", "12.05h"), "05": ("350층", "36.05h"), "06": ("51층", "2.39h")}
L = np.array([0.4, -0.6, 0.8]) / np.linalg.norm([0.4, -0.6, 0.8])
fig = plt.figure(figsize=(18, 11), dpi=120)
for n, j in enumerate(sorted(INFO)):
    m = trimesh.load(R / "WAAM_Validator" / "tests" / j / "target.stl")
    ax = fig.add_subplot(2, 3, n + 1, projection="3d")
    tri = m.vertices[m.faces]
    sh = 0.3 + 0.7 * np.clip(m.face_normals @ L, 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(np.outer(sh, [0.20, 0.33, 0.58]) + 0.12, 0, 1),
                                         edgecolors="none"))
    lo, hi = m.bounds
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(28, -55)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.tick_params(labelsize=7)
    ext = hi - lo
    ax.set_title(f"공개 {j}   {ext[0]:.0f}×{ext[1]:.0f}×{ext[2]:.0f} mm   {INFO[j][0]}   V7 makespan {INFO[j][1]}",
                 fontsize=12)
fig.suptitle("교수님 공개 모델 6개 (target.stl)", fontsize=16)
fig.tight_layout()
fig.savefig(R / "tools" / "public_models.png", facecolor="white")
print("saved")
