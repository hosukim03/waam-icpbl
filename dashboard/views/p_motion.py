import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import WAIT_MIN, eval_index, layer_frames, layer_view, motion, run_table, timeline3d, traj_shapes
from theme import badge, chart, header, pal, style

header("움직임 그림", "한 층에서 로봇들이 실제로 움직인 모습입니다(위에서 본 그림). 실선은 쌓은 선, 점선은 이동한 길, "
       f"×는 다른 로봇 팔과 부딪히지 않으려고 기다린 곳({WAIT_MIN:.0f}초 이상), ○는 층 끝에서 다른 로봇이 끝나기를 기다린 곳, "
       "▲는 길을 터 주려고 주차점으로 비켜선 곳입니다. 실험 기록을 모을 때 궤적을 같이 넣은 실험만 보입니다"
       "(tools/collect_evals.py --traj).")
idx = eval_index()
idx = idx[idx["name"].map(lambda n: bool(traj_shapes(n)))] if not idx.empty else idx
if idx.empty:
    st.info("궤적을 같이 모은 실험이 없습니다. tools/collect_evals.py eval … --traj 로 모으세요.")
    st.stop()
P = pal()
sets = list(dict.fromkeys(idx["set"]))
labels = dict(zip(idx["set"], idx["set_label"]))
f1, f2, f3, f4 = st.columns([1, 2, 2, 1.3])
s = f1.selectbox("묶음", sets, format_func=labels.get)
g = idx[idx["set"] == s].sort_values("name").to_dict("records")
e = f2.selectbox("실험", g, index=len(g) - 1, format_func=lambda x: f"{x['exp']} — {x['desc']}", key=f"exp_{s}")
t = run_table(e["name"], e["set_dir"]).set_index("형상")
shapes = traj_shapes(e["name"])
sh = f3.selectbox("형상", shapes, format_func=lambda k: f"{k}  {badge(t.loc[k, '통과'])}  {t.loc[k, '고른 설정']}  {t.loc[k, '설명']}"
                  if k in t.index else k)
view = f4.radio("보기", ["층 넘기며 재생", "한 층 자세히", "3D 시간 재생"], horizontal=False)
M = motion(e["name"], sh, e["set_dir"])
if M is None:
    st.warning("궤적이 없습니다.")
    st.stop()
seg, span, n, robots = M["seg"], M["span"], M["n"], M["robots"]
dur = span["t1"] - span["t0"]
COL = [P["series"][i % 3] for i in range(len(robots))]
R = M["work_r"]
lim = max(R, max(float(np.hypot(*r["park"])) for r in robots)) + 90
th = np.linspace(0, 2 * np.pi, 241)


def base_figure(title: str) -> go.Figure:
    """바뀌지 않는 바탕: 작업대, 로봇마다 팔이 닿는 범위(옅게)와 주차점. 선 굵기는 style 뒤에 넣어야 남는다."""
    fig = go.Figure()
    style(fig, height=780, legend=True, title=title)
    fig.add_scatter(x=R * np.cos(th), y=R * np.sin(th), mode="lines", line=dict(color=P["axis"], width=1.5, dash="dash"),
                    hoverinfo="skip", showlegend=False)
    for r, col in zip(robots, COL):
        bx, by = r["base"]
        fig.add_scatter(x=bx + r["reach"] * np.cos(th), y=by + r["reach"] * np.sin(th), mode="lines",
                        line=dict(color=col, width=1), opacity=0.25, hoverinfo="skip", showlegend=False,
                        legendgroup=str(r["id"]))
        fig.add_scatter(x=[r["park"][0]], y=[r["park"][1]], mode="markers+text", text=[f"주차점 {r['id']}"],
                        textposition="middle left" if r["park"][0] > 0 else "middle right",
                        textfont=dict(color=P["muted"], size=11), hovertemplate=f"로봇 {r['id']} 주차점<extra></extra>",
                        marker=dict(symbol="square", size=10, color=col, line=dict(width=2, color=P["surface"])),
                        showlegend=False, legendgroup=str(r["id"]))
    fig.update_xaxes(range=[-lim, lim], title="X (mm)", constrain="domain")
    fig.update_yaxes(range=[-lim, lim], title="Y (mm)", scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(t=88), legend=dict(y=1.01))
    return fig


def robot_traces(v: dict, col: str) -> list[go.Scatter]:
    """로봇 하나의 움직임 다섯 겹(쌓은 선·옮긴 길·기다림·층 끝 기다림·비켜섬). 재생할 때 층마다 이 다섯 개만 바뀐다."""
    rid, grp = v["id"], str(v["id"])
    D, T, W, E, K = v["D"], v["T"], v["W"], v["E"], v["K"]
    ring = dict(width=1, color=P["surface"])
    return [
        go.Scatter(x=D[0], y=D[1], customdata=D[2], mode="lines", name=f"로봇 {rid}", legendgroup=grp,
                   line=dict(color=col, width=2.2),
                   hovertemplate=f"로봇 {rid} 쌓기<br>층 시작 뒤 %{{customdata:.1f}} s<extra></extra>"),
        go.Scatter(x=T[0], y=T[1], customdata=T[2], mode="lines", legendgroup=grp, showlegend=False, opacity=0.8,
                   line=dict(color=col, width=1.3, dash="dash"),
                   hovertemplate=f"로봇 {rid} 이동<br>층 시작 뒤 %{{customdata:.1f}} s<extra></extra>"),
        go.Scatter(x=W[0], y=W[1], customdata=np.c_[W[2], W[3]] if W[0] else [], mode="markers", legendgroup=grp,
                   showlegend=False, marker=dict(symbol="x", size=10, color=col, line=ring),
                   hovertemplate=f"로봇 {rid} 기다림 %{{customdata[0]:.1f}} s<br>층 시작 뒤 %{{customdata[1]:.1f}} s<extra></extra>"),
        go.Scatter(x=E[0], y=E[1], customdata=E[2], mode="markers", legendgroup=grp, showlegend=False,
                   marker=dict(symbol="circle-open", size=12, color=col, line=dict(width=2)),
                   hovertemplate=f"로봇 {rid} 층 끝에서 기다림 %{{customdata:.1f}} s<extra></extra>"),
        go.Scatter(x=K[0], y=K[1], customdata=K[2], mode="markers", legendgroup=grp, showlegend=False,
                   marker=dict(symbol="triangle-up", size=12, color=col, line=dict(width=2, color=P["surface"])),
                   hovertemplate=f"로봇 {rid} 주차점으로 비켜섬<br>층 시작 뒤 %{{customdata:.1f}} s<extra></extra>"),
    ]


def summary(views: list[dict]) -> str:
    """층 제목 끝에 붙는 로봇별 한 줄: 쌓기·기다리기 시간."""
    parts = [f"로봇 {v['id']} 쌓기 {v['sum']['D']:.0f}·기다림 {v['sum']['W'] + v['sum']['E']:.0f} s"
             for v in views if v["sum"]["D"] > 0]
    return "  |  ".join(parts) if parts else "쌓은 로봇 없음"


def title_of(L: int, views: list[dict]) -> str:
    return (f"{labels[s]} {sh}번 · {L + 1}/{n}층 · {dur.get(L, 0.0):,.1f} s<br>"
            f"<span style='font-size:12px'>{summary(views)}</span>")


if view == "3D 시간 재생":
    import json
    from pathlib import Path

    import streamlit.components.v1 as components
    D3 = timeline3d(e["name"], sh, e["set_dir"])
    if D3 is None:
        st.warning("궤적이 없습니다.")
        st.stop()
    page = (Path(__file__).resolve().parent.parent / "assets" / "motion3d.html").read_text(encoding="utf-8")
    for k, v in {"__INK__": P["ink"], "__INK2__": P["ink2"], "__MUTED__": P["muted"], "__GRID__": P["grid"],
                 "__SURFACE__": P["surface"], "__ACCENT__": P["series"][0]}.items():
        page = page.replace(k, v)
    page = (page.replace("__COLORS__", json.dumps(COL)).replace("__PAL__", json.dumps(P))
            .replace("__DATA__", json.dumps(D3, separators=(",", ":"))))
    # 그림 라이브러리는 인터넷에서 받지 않고, 설치된 plotly 파이썬 패키지에 든 plotly.min.js 를 그대로 넣는다
    import plotly
    js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")
    page = page.replace("__PLOTLY__", "<script>" + js.replace("</script", "<\\/script") + "</script>")
    st.markdown(f"**{labels[s]} {sh}번** · {n}층 · makespan {D3['T']:,.0f} s")
    components.html(page, height=860, scrolling=False)
    st.caption("마우스로 끌면 돌아가고, 휠로 확대·축소합니다. ▶ 재생을 누르면 1초 간격 위치로 움직이며, 속도 30배면 "
               "한 장에 1초씩 넘어갑니다. 막대를 끌거나 '층으로 가기'로 원하는 때로 갈 수 있습니다. 굵은 선은 지금 층에서 "
               "쌓은 선, 옅은 선은 지난 층들입니다. 기둥은 로봇 베이스, 기둥 꼭대기에서 토치까지의 선은 팔을 단순하게 "
               "그린 것입니다(채점기도 위에서 본 베이스–토치 선분으로 충돌을 검사합니다). 토치 표시: ● 쌓기, ◇ 이동, "
               "× 기다리기, □ 끝남. 쌓은 선은 가볍게 하려고 5 mm 간격으로 줄였습니다.")
    st.stop()

if view == "층 넘기며 재생":
    frames_v = layer_frames(e["name"], sh, e["set_dir"])
    if not frames_v:
        st.warning("층이 없습니다.")
        st.stop()
    fig = base_figure(title_of(0, frames_v[0]))
    n_static = len(fig.data)
    for v, col in zip(frames_v[0], COL):
        fig.add_traces(robot_traces(v, col))
    dyn = list(range(n_static, len(fig.data)))
    fig.frames = [go.Frame(name=str(L + 1), traces=dyn,
                           data=[tr for v, col in zip(views, COL) for tr in robot_traces(v, col)],
                           layout=go.Layout(title_text=title_of(L, views)))
                  for L, views in enumerate(frames_v) if views]
    play = dict(frame=dict(duration=900, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")
    btn = dict(bgcolor=P["surface"], bordercolor=P["axis"], font=dict(color=P["ink"]))
    fig.update_layout(
        margin=dict(t=96, b=140),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.11, xanchor="left", yanchor="top", showactive=False,
                          pad=dict(t=4, r=8), **btn,
                          buttons=[dict(label="▶ 재생", method="animate", args=[None, play]),
                                   dict(label="⏸ 멈춤", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.16, y=-0.11, len=0.84, xanchor="left", yanchor="top", pad=dict(t=4),
                      currentvalue=dict(prefix="층 ", font=dict(color=P["ink2"])),
                      font=dict(color=P["muted"] if n <= 30 else "rgba(0,0,0,0)", size=10),   # 층이 많으면 눈금 글자는 감춘다
                      bgcolor=P["grid"], bordercolor=P["axis"], activebgcolor=P["series"][0], tickcolor=P["axis"],
                      steps=[dict(label=f.name, method="animate",
                                  args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode="immediate",
                                                       transition=dict(duration=0))]) for f in fig.frames])])
    fig.update_xaxes(title=None)                     # X축 이름이 재생 단추와 겹쳐 뺀다(단위는 아래 설명에)
    chart(fig)
    st.caption("가로·세로 모두 mm. ▶ 재생을 누르면 1층부터 한 층씩(0.9초마다) 넘어갑니다. 아래 막대를 끌어 원하는 층으로 갈 수 있습니다. "
               "재생 그림은 가볍게 하려고 쌓은 선의 점을 3 mm 간격으로 줄였습니다(모양은 같습니다). "
               "점선 원은 작업대(반지름 700 mm), 옅은 원은 팔이 닿는 범위, 네모는 주차점입니다.")

    # 층마다 걸린 시간: 어느 층에서 시간이 새는지
    rows = []
    for L, views in enumerate(frames_v):
        row = {"층": L + 1, "걸린 시간(s)": float(dur.get(L, 0.0))}
        for v in views:
            row[f"로봇 {v['id']} 기다림(s)"] = v["sum"]["W"] + v["sum"]["E"]
            row[f"로봇 {v['id']} 비켜섬"] = v["n_park"]
        rows.append(row)
    lt = pd.DataFrame(rows)
    wait_cols = [c for c in lt.columns if c.endswith("기다림(s)")]
    fig2 = go.Figure()
    style(fig2, height=280, legend=False, title="층마다 걸린 시간 (s)")
    fig2.add_bar(x=lt["층"], y=lt["걸린 시간(s)"], marker_color=P["series"][0],
                 customdata=lt[wait_cols].sum(axis=1),
                 hovertemplate="%{x}층: %{y:,.1f} s<br>로봇들이 기다린 시간 합 %{customdata:,.1f} s<extra></extra>")
    fig2.update_xaxes(title="층 (아래부터)", showgrid=False)
    chart(fig2)
    with st.expander("층마다 표"):
        st.dataframe(lt, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ["걸린 시간(s)", *wait_cols]})
    st.stop()

# ---------------------------------------------------------------- 한 층 자세히
default = int(dur.idxmax()) + 1 if len(dur) else 1
L = (st.slider("층 (처음엔 가장 오래 걸린 층)", 1, n, default, key=f"layer_{e['name']}_{sh}") if n > 1 else 1) - 1
views = layer_view(seg[seg["layer"] == L], robots, float(span.loc[L, "t0"]))
c = st.columns(4)
c[0].metric("이 층에 걸린 시간", f"{dur.get(L, 0.0):,.1f} s", border=True)
c[1].metric("층", f"{L + 1} / {n}", border=True)
c[2].metric("일한 로봇", f"{sum(v['sum']['D'] > 0 for v in views)}대", border=True)
c[3].metric("기다린 시간 합", f"{sum(v['sum']['W'] + v['sum']['E'] for v in views):,.1f} s", border=True,
            help="로봇마다 이 층에서 기다린 시간(부딪히지 않으려고 + 층 끝에서)을 더한 값")
fig = base_figure(title_of(L, views))
for v, col in zip(views, COL):
    fig.add_traces(robot_traces(v, col))
chart(fig)
st.caption("점선 원은 작업대(반지름 700 mm), 옅은 원은 로봇마다 팔이 닿는 범위(1,500 mm)입니다. 베이스는 그림 밖 1,400 mm 에 있고, "
           "네모가 주차점입니다. 첫 층의 ▲는 시작할 때 주차점으로 가는 움직임입니다. 범례를 누르면 그 로봇만 감출 수 있습니다.")
st.dataframe(pd.DataFrame([{"로봇": f"로봇 {v['id']}", "쌓기(s)": v["sum"]["D"], "이동하기(s)": v["sum"]["T"],
                            "기다리기(s)": v["sum"]["W"], "층 끝 기다리기(s)": v["sum"]["E"],
                            f"{WAIT_MIN:.0f}초 넘게 기다린 번 수": v["n_wait"], "비켜선 번 수": v["n_park"]} for v in views]),
             hide_index=True, width="stretch",
             column_config={k: st.column_config.NumberColumn(format="%.1f")
                            for k in ("쌓기(s)", "이동하기(s)", "기다리기(s)", "층 끝 기다리기(s)")})
