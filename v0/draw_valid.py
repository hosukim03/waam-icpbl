# -*- coding: utf-8 -*-
"""학습 형상 v3(train_div) 3D 그림 (기본 파이썬)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
NAMES = {'line': '막대', 'ring': '원형 벽', 'rect': '사각 벽', 'block': '속찬 블록', 'wave': '물결선', 'hexgrid': '벌집 격자', 'sqgrid': '사각 격자', 'trigrid': '삼각 격자', 'spiral': '나선', 'truss': '트러스', 'islands': '떨어진 섬', 'star': '별', 'plate': '판+기둥', 'vase': '꽃병', 'bracket': '브래킷'}
R = Path(__file__).resolve().parents[1]
L = np.array([0.4, -0.6, 0.8]) / np.linalg.norm([0.4, -0.6, 0.8])
dirs = sorted(p for p in (R / "data" / "valid_div").iterdir() if (p / "target.stl").exists())
fig = plt.figure(figsize=(19, 15.5), dpi=110)
for n, d in enumerate(dirs):
    m = trimesh.load(d / "target.stl")
    ax = fig.add_subplot(4, 5, n + 1, projection="3d")
    tri = m.vertices[m.faces]
    sh = 0.3 + 0.7 * np.clip(m.face_normals @ L, 0, 1)
    desc = (d / "desc.txt").read_text(encoding="utf-8")
    lvl = desc.split()[0]
    col = {"쉬움": [0.30, 0.55, 0.35], "중간": [0.20, 0.33, 0.58], "어려움": [0.60, 0.35, 0.20]}[lvl]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(np.outer(sh, col) + 0.12, 0, 1), edgecolors="none"))
    lo, hi = m.bounds
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(30, -55); ax.set_axis_off()
    ext = hi - lo
    toks = desc.split()
    key = next((t.split("/")[0] for t in toks if t.split("/")[0] in NAMES), "vase")
    mode = next((t.split("/")[1] for t in toks if "/" in t and t.split("/")[0] in NAMES), "")
    ax.set_title(f"{d.name} [{lvl}] {NAMES[key]} {mode}" + chr(10) + f"{ext[0]:.0f}×{ext[1]:.0f}×{ext[2]:.0f}mm", fontsize=11)
fig.suptitle("검증용 형상 (20개, 학습에 안 씀) — 14개 계열 (초록 쉬움 / 파랑 중간 / 주황 어려움)", fontsize=16)
fig.tight_layout()
fig.savefig(R / "v0" / "shapes_valid.png", facecolor="white")
print("saved")
