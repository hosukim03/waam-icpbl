import plotly.graph_objects as go
import streamlit as st

from data import SET_NOTE, eval_index, run_table
from theme import badge, chart, header, pal, style

header("실험 비교", "같은 묶음에서 두 실험을 형상별로 비교합니다. 기준 실험 대비 비교 실험이 얼마나 바뀌었는지 봅니다.")
idx = eval_index()
if idx.empty or idx.groupby("set").size().max() < 2:
    st.info("같은 묶음에 실험이 둘 이상 있어야 비교할 수 있습니다.")
    st.stop()
P = pal()
sets = [s for s, n in idx.groupby("set", sort=False).size().items() if n >= 2]
labels = dict(zip(idx["set"], idx["set_label"]))
fmt = lambda r: f"{r['exp']} — {r['desc']}"  # noqa: E731
f1, f2, f3 = st.columns([2, 3, 3])
s = f1.radio("묶음", sets, format_func=labels.get, horizontal=True)
g = idx[idx["set"] == s].sort_values("name").to_dict("records")
a = f2.selectbox("기준 실험", g, index=len(g) - 2, format_func=fmt)
b = f3.selectbox("비교 실험", g, index=len(g) - 1, format_func=fmt)
st.caption(f"**{labels[s]}**: {SET_NOTE.get(s, '')}")
ta, tb = run_table(a["name"], a["set_dir"]), run_table(b["name"], b["set_dir"])
m = ta.merge(tb, on="형상", suffixes=(" 기준", " 비교"))
m["차이(s)"] = m["makespan(s) 비교"] - m["makespan(s) 기준"]
m["차이(%)"] = 100 * m["차이(s)"] / m["makespan(s) 기준"]

c = st.columns(4)
sa, sb = m["makespan(s) 기준"].sum(), m["makespan(s) 비교"].sum()
c[0].metric("makespan 합계", f"{sb:,.1f} s", f"{(sb / sa - 1) * 100:+.2f}%", delta_color="inverse", border=True)
na, nb = int((m["통과 기준"] == "PASS").sum()), int((m["통과 비교"] == "PASS").sum())
c[1].metric("통과", f"{nb} / {len(m)}", f"{nb - na:+d}" if nb != na else None, border=True)
c[2].metric("줄어든 형상", f"{int((m['차이(s)'] < -0.05).sum())}", border=True)
c[3].metric("늘어난 형상", f"{int((m['차이(s)'] > 0.05).sum())}", border=True)

ch = m[(m["차이(s)"].abs() > 0.05) | (m["통과 기준"] != m["통과 비교"])].sort_values("차이(%)")
if ch.empty:
    st.success("두 실험의 결과가 형상마다 같습니다.")
    st.stop()
fig = go.Figure(go.Bar(
    y=ch["형상"], x=ch["차이(%)"], orientation="h",
    marker_color=[P["better"] if v < 0 else P["worse"] for v in ch["차이(%)"]],
    text=[f"{v:+.2f}%" for v in ch["차이(%)"]], textposition="outside", textfont=dict(color=P["ink2"]),
    customdata=ch[["설명 기준", "makespan(s) 기준", "makespan(s) 비교"]].values,
    hovertemplate="형상 %{y}<br>%{customdata[0]}<br>%{customdata[1]:,.1f} → %{customdata[2]:,.1f} s (%{x:+.2f}%)<extra></extra>"))
lim = max(1.0, float(ch["차이(%)"].abs().max()) * 1.25)
style(fig, height=max(240, 28 * len(ch) + 90), legend=False,
      title="바뀐 형상의 makespan 차이 (%) · 파랑 = 비교 실험이 빠름, 빨강 = 느림")
fig.update_xaxes(range=[-lim, lim], zeroline=True, zerolinecolor=P["axis"], zerolinewidth=1)
fig.update_yaxes(type="category", showgrid=False)
chart(fig)

view = ch.assign(**{"통과 기준": ch["통과 기준"].map(badge), "통과 비교": ch["통과 비교"].map(badge)})
cols = ["형상", "설명 기준", "통과 기준", "통과 비교", "makespan(s) 기준", "makespan(s) 비교", "차이(s)", "차이(%)",
        "고른 설정 기준", "고른 설정 비교", "겹침 점수(%) 기준", "겹침 점수(%) 비교"]
st.dataframe(view[cols].rename(columns={"설명 기준": "설명"}), hide_index=True, width="stretch",
             column_config={k: st.column_config.NumberColumn(format="%.1f") for k in ("makespan(s) 기준", "makespan(s) 비교")} |
                           {"차이(s)": st.column_config.NumberColumn(format="%+.1f"), "차이(%)": st.column_config.NumberColumn(format="%+.2f"),
                            "겹침 점수(%) 기준": st.column_config.NumberColumn(format="%.2f"),
                            "겹침 점수(%) 비교": st.column_config.NumberColumn(format="%.2f")})
st.caption("풀이 시간은 여러 형상을 동시에 돌린 값이라 비교에 쓰지 않습니다. 계산 시간 비교는 '계산 시간 (코어 1개)'를 보세요.")
