# -*- coding: utf-8 -*-
"""shapes2.json → 학습 형상 v2 그림 (첫 층 연파랑 / 중간 층 파랑 / 마지막 층 남색 선)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Malgun Gothic"
D = Path(__file__).parent
data = json.load(open(D / "shapes2.json", encoding="utf-8"))
KIND = {"01": "곡선벽·기둥", "02": "격자·기둥", "03": "살+테두리·기둥", "04": "링 반지름 변화",
        "05": "나선·비틀림", "06": "막대·기둥", "07": "이중 링·원뿔", "08": "링 반지름 변화",
        "09": "격자·기둥", "10": "살+테두리·기둥", "11": "별·원뿔", "12": "나선·기둥",
        "13": "링 반지름 변화", "14": "이중 링·비틀림"}
COL = ["#9DBBE3", "#4F7CC0", "#1B2F57"]
keys = sorted(data)
ncol = 5
nrow = -(-len(keys) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.5 * nrow), dpi=130)
for ax, k in zip(axes.ravel(), keys):
    lay = data[k]["layers"]
    w = 1.0
    for c, (lk, polys) in zip(COL, sorted(lay.items(), key=lambda x: int(x[0]))):
        for p in polys:
            for ring in [p["ext"], *p["holes"]]:
                xs, ys = zip(*ring)
                ax.plot(xs, ys, color=c, lw=1.4)
                w = max(w, max(map(abs, xs)), max(map(abs, ys)))
    w *= 1.12
    ax.set_xlim(-w, w); ax.set_ylim(-w, w); ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(f"{k}  {KIND.get(k, '')}  ({data[k]['n']}층)", fontsize=10.5)
for ax in axes.ravel()[len(keys):]:
    ax.set_axis_off()
handles = [Line2D([], [], color=c, lw=2) for c in COL]
fig.legend(handles, ["첫 층", "중간 층", "마지막 층"], loc="lower right", frameon=False, fontsize=11)
fig.suptitle("학습 형상 v2 (14개) — 선 색으로 층별 단면 비교, 겹치면 층마다 같은 기둥형", fontsize=14)
fig.tight_layout()
fig.savefig(D / "shapes_train2.png", facecolor="white")
print("saved shapes_train2.png")
