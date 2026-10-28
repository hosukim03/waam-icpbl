import plotly.graph_objects as go
import streamlit as st

from data import timing_index, timing_table
from theme import chart, header, pal, style

header("계산 시간 (코어 1개)",
       "풀이 하나를 고성능 코어 2번 하나에만 돌게 고정하고(PYTHON_CPU_COUNT=1), 비교할 방법들을 형상마다 번갈아 풀어 잰 "
       "벽시계 시간입니다. 같은 코드라도 노트북 상태(열, 충전기)에 따라 회차마다 합계가 10–20% 달라질 수 있어, "
       "비교는 한 기록 안에서만 합니다.")
idx = timing_index()
if idx.empty:
    st.info("runs/timing 에 모은 측정 기록이 없습니다(tools/collect_evals.py timing).")
    st.stop()
P = pal()
e = st.selectbox("측정 기록", idx.sort_values("name", ascending=False).to_dict("records"),
                 format_func=lambda r: f"{r['name']} — {r['desc']}")
t = timing_table(e["name"])
if t.empty:
    st.warning("기록을 읽지 못했습니다.")
    st.stop()
ways = list(dict.fromkeys(t["방법"]))[:3]                 # 색 세 칸까지만(검증한 범위)
t = t[t["방법"].isin(ways)]
tot = t.groupby("방법", sort=False)[["계산 시간(s)", "makespan(s)"]].sum().reindex(ways)
c = st.columns(max(len(ways), 3))
for i, w in enumerate(ways):
    d = None if i == 0 else f"{(tot.loc[w, '계산 시간(s)'] / tot.loc[ways[0], '계산 시간(s)'] - 1) * 100:+.1f}%"
    c[i].metric(f"{w} · 계산 시간 합계", f"{tot.loc[w, '계산 시간(s)']:,.1f} s", d, delta_color="inverse", border=True,
                help=f"makespan 합계 {tot.loc[w, 'makespan(s)']:,.1f} s" + ("" if i == 0 else f" · 첫 방법({ways[0]}) 대비"))
fig = go.Figure()
for i, w in enumerate(ways):
    d = t[t["방법"] == w]
    fig.add_bar(x=d["형상"], y=d["계산 시간(s)"], name=w, marker_color=P["series"][i],
                hovertemplate=f"{w}<br>형상 %{{x}}: %{{y:.1f}} s<extra></extra>")
style(fig, height=380, legend=True, title=f"형상별 계산 시간 (s) · {e['set_label']}")
fig.update_xaxes(type="category", showgrid=False)
chart(fig)
pv = t.pivot_table(index="형상", columns="방법", values=["계산 시간(s)", "makespan(s)"], aggfunc="first")
pv.columns = [f"{b} {a}" for a, b in pv.columns]
pv = pv[[f"{w} 계산 시간(s)" for w in ways] + [f"{w} makespan(s)" for w in ways]].reset_index()
st.dataframe(pv, hide_index=True, width="stretch", height=35 * (len(pv) + 1) + 4,
             column_config={k: st.column_config.NumberColumn(format="%.1f") for k in pv.columns if k != "형상"})
