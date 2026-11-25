import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import eval_index, motion, run_table, traj_shapes
from pick import kept, pick
from theme import chart, header, pal, shape_note, shape_option, style

header("작업 일정(간트)", "로봇마다 시간에 따라 무엇을 했는지 보여 줍니다. 쌓기, 이동하기(불꽃 없이 옮겨 가기), "
       "기다리기(다른 로봇 팔과 부딪히지 않으려고 제자리에 선 시간), 층 끝 기다리기(그 층을 먼저 끝내고 다른 로봇을 "
       "기다린 시간)로 나눴습니다. 세로선은 층이 시작하는 시각입니다. 실험 기록을 모을 때 궤적을 같이 넣은 실험만 "
       "보입니다(tools/collect_evals.py --traj).")
idx = eval_index()
idx = idx[idx["name"].map(lambda n: bool(traj_shapes(n)))] if not idx.empty else idx
if idx.empty:
    st.info("궤적을 같이 모은 실험이 없습니다. tools/collect_evals.py eval … --traj 로 모으세요.")
    st.stop()
P = pal()
KINDS = [("D", "쌓기", P["series"][1]), ("T", "이동하기", P["series"][0]),
         ("W", "기다리기", P["muted"]), ("E", "층 끝 기다리기", P["deemph"])]

sets = list(dict.fromkeys(idx["set"]))
labels = dict(zip(idx["set"], idx["set_label"]))
f1, f2, f3 = st.columns([1, 2, 2])
s = pick(f1, "묶음", sets, "set", "gantt_set", format_func=labels.get)
recs = {r["exp"]: r for r in idx[idx["set"] == s].sort_values("name").to_dict("records")}
e = recs[pick(f2, "실험", list(recs), "exp", f"gantt_exp_{s}", default=len(recs) - 1,
              format_func=lambda x: recs[x]["exp_label"])]
if kept("exp") not in recs:
    f2.caption(f"다른 화면에서 고른 실험({dict(zip(eval_index()['exp'], eval_index()['exp_label'])).get(kept('exp'))})은 "
               "경로를 모아 두지 않아 여기서는 볼 수 없습니다.")
t = run_table(e["name"], e["set_dir"]).set_index("형상")
shapes = traj_shapes(e["name"])
sh = pick(f3, "형상", shapes, "shape", f"gantt_shape_{e['name']}",
          format_func=lambda k: shape_option(k, t.loc[k]) if k in t.index else k)
if sh in t.index:
    f3.caption(shape_note(t.loc[sh]))
M = motion(e["name"], sh, e["set_dir"])
if M is None:
    st.warning("궤적이 없습니다.")
    st.stop()
seg, span, n, robots = M["seg"], M["span"], M["n"], M["robots"]


@st.cache_data(ttl=600, max_entries=6, show_spinner=False)
def runs(name: str, shape: str, set_dir: str) -> pd.DataFrame:
    """같은 종류로 이어진 구간을 하나로 합친다(쌓기는 꺾인 선 한 줄이 수십 구간이라 그대로 그리면 무겁다)."""
    sg = motion(name, shape, set_dir)["seg"].sort_values(["robot", "t0"])
    out = []
    for rid, gr in sg.groupby("robot", sort=True):
        k, t0, t1, lay = (gr[c].to_numpy() for c in ("kind", "t0", "t1", "layer"))
        new = np.r_[True, (k[1:] != k[:-1]) | (np.abs(t0[1:] - t1[:-1]) > 1e-6)]
        out.append(pd.DataFrame({"robot": rid, "kind": k[new], "layer": lay[new],
                                 "t0": t0[new], "t1": np.maximum.reduceat(t1, np.nonzero(new)[0])}))
    return pd.concat(out, ignore_index=True)


R = runs(e["name"], sh, e["set_dir"])
c1, c2 = st.columns([1, 3])
view = c1.radio("보기", ["층 범위", "전체"], horizontal=True)
if view == "층 범위" and n > 1:
    a, b = c2.slider("층 범위", 1, n, (1, min(n, 3)))
else:
    a, b = 1, n
L0, L1 = a - 1, b - 1
x0 = float(span.loc[L0, "t0"]) if L0 in span.index else float(R["t0"].min())
x1 = float(span.loc[L1, "t1"]) if L1 in span.index else float(R["t1"].max())
pad = (x1 - x0) * 0.01
W = R[(R["t1"] > x0) & (R["t0"] < x1)]

ids = [r["id"] for r in robots]
ypos = {rid: len(ids) - i for i, rid in enumerate(ids)}          # 로봇 1 이 맨 위
fig = go.Figure()
style(fig, height=180 + 70 * len(ids), legend=True)
for k, name, col in KINDS:
    w = W[W["kind"] == k]
    if w.empty:
        continue
    xs = np.c_[np.maximum(w["t0"], x0), np.minimum(w["t1"], x1), np.full(len(w), np.nan)].ravel()
    ys = np.c_[w["robot"].map(ypos), w["robot"].map(ypos), np.full(len(w), np.nan)].ravel()
    dur = np.repeat((w["t1"] - w["t0"]).to_numpy(), 3)
    fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=name, line=dict(color=col, width=26),
                               customdata=dur, hovertemplate=f"{name} %{{customdata:.1f}} s<br>%{{x:.1f}} s<extra></extra>"))
# 층 시작 세로선(층이 적을 때만 번호를 단다)
starts = span.loc[(span.index >= L0) & (span.index <= L1), "t0"]
vx = np.c_[starts, starts, np.full(len(starts), np.nan)].ravel()
vy = np.c_[np.full(len(starts), 0.45), np.full(len(starts), len(ids) + 0.55), np.full(len(starts), np.nan)].ravel()
fig.add_trace(go.Scatter(x=vx, y=vy, mode="lines", line=dict(color=P["axis"], width=1), hoverinfo="skip",
                         showlegend=False))
if len(starts) <= 25:
    for L, sx in starts.items():
        fig.add_annotation(x=sx, y=len(ids) + 0.55, text=f"{L + 1}층", showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(size=11, color=P["muted"]))
fig.update_yaxes(tickvals=[ypos[r] for r in ids], ticktext=[f"로봇 {r}" for r in ids], range=[0.4, len(ids) + 0.9],
                 showgrid=False)
fig.update_xaxes(range=[x0 - pad, x1 + pad], title="시각 (s)")
fig.update_layout(legend=dict(traceorder="normal"))
chart(fig)

# 고른 범위의 로봇별 시간
rows = []
for rid in ids:
    w = W[W["robot"] == rid]
    d = (np.minimum(w["t1"], x1) - np.maximum(w["t0"], x0)).clip(lower=0)
    tot = {name: float(d[w["kind"] == k].sum()) for k, name, _ in KINDS}
    allt = sum(tot.values())
    row = {"로봇": f"로봇 {rid}"}
    for name, v in tot.items():
        row[f"{name}(s)"] = round(v, 1)
    row["기다리기 비율"] = f"{(tot['기다리기'] + tot['층 끝 기다리기']) / allt * 100:.1f}%" if allt else "-"
    rows.append(row)
st.caption(f"{a}–{b}층, {x1 - x0:,.0f} s 동안 로봇별 시간(기다리기 비율 = 기다리기 + 층 끝 기다리기)")
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
