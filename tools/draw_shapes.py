# -*- coding: utf-8 -*-
"""shapes.json 을 읽어 벤치마크 10개 / 학습 12개 단면 그림으로 저장. (기본 파이썬)"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
D = Path(__file__).parent
data = json.load(open(D / "shapes.json", encoding="utf-8"))


def draw(prefix, title, out, ncol=5):
    keys = sorted(k for k in data if k.startswith(prefix))
    nrow = -(-len(keys) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow), dpi=130)
    for ax, k in zip(axes.ravel(), keys):
        ax.add_patch(Circle((0, 0), 700, fill=False, ls="--", lw=0.8, color="#b0b6bd"))  # 작업대 경계
        for poly in data[k]:
            xs, ys = zip(*poly["ext"])
            ax.fill(xs, ys, color="#21365E", alpha=0.85, lw=0)
            for h in poly["holes"]:
                hx, hy = zip(*h)
                ax.fill(hx, hy, color="white", lw=0)
        w = max(max(abs(x) for p in data[k] for x, _ in p["ext"]),
                max(abs(y) for p in data[k] for _, y in p["ext"])) * 1.15
        ax.set_xlim(-w, w); ax.set_ylim(-w, w); ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(f"{k.split('/')[1]}   폭 {2 * w / 1.15:.0f}mm", fontsize=11)
    for ax in axes.ravel()[len(keys):]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(D / out, facecolor="white")
    print("saved", out)


draw("bench/", "벤치마크 10개 — 학습에 쓰지 않고 검증만 (처음 보는 형상 대용), 첫 층 단면", "shapes_bench.png")
draw("train/", "학습 형상 12개 — RL 학습 전용 (공개·벤치와 겹치지 않음), 첫 층 단면", "shapes_train.png", ncol=4)
