import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import SET_NOTE, eval_index, run_table
from theme import badge, chart, header, pal, style

header("묶음 결과", "형상 묶음 하나에서 실험 하나의 결과입니다. 풀이 시간은 여러 형상을 동시에 돌린 값이라 참고만 합니다"
       "(코어 1개 시간은 '계산 시간' 쪽).")
idx = eval_index()
if idx.empty:
    st.info("runs/evals 에 모은 실험이 없습니다(tools/collect_evals.py).")
    st.stop()
P = pal()
sets = list(dict.fromkeys(idx["set"]))
labels = dict(zip(idx["set"], idx["set_label"]))
f1, f2 = st.columns([2, 3])
s = f1.radio("묶음", sets, format_func=labels.get, horizontal=True)
g = idx[idx["set"] == s].sort_values("name")
e = f2.selectbox("실험", g.to_dict("records"), index=len(g) - 1, format_func=lambda r: f"{r['exp']} — {r['desc']}")
st.caption(f"**{labels[s]}**: {SET_NOTE.get(s, '')}")
t = run_table(e["name"], e["set_dir"])

c = st.columns(4)
n_ok = int((t["통과"] == "PASS").sum())
c[0].metric("통과", f"{n_ok} / {len(t)}", border=True)
c[1].metric("makespan 합계", f"{t['makespan(s)'].sum():,.1f} s", border=True)
b = t["이론상 최소(s)"].sum()
c[2].metric("이론상 최소의 몇 배", f"{t['makespan(s)'].sum() / b:.2f}" if b else "-", border=True)
c[3].metric("충돌", f"{int(t['충돌'].fillna(0).sum())}", border=True)

ok = t["통과"] == "PASS"
fig = go.Figure(go.Bar(
    y=t["형상"], x=t["최소의 몇 배"], orientation="h",
    marker_color=[P["series"][0] if v else P["critical"] for v in ok],
    text=["" if v else "⛔ 실패" for v in ok], textposition="outside", textfont=dict(color=P["ink"]),
    customdata=t[["설명", "makespan(s)", "통과"]].assign(통과=t["통과"].map(badge)).values,
    hovertemplate="형상 %{y}  %{customdata[2]}<br>%{customdata[0]}<br>makespan %{customdata[1]:,.1f} s<br>"
                  "이론상 최소의 %{x:.2f}배<extra></extra>"))
fig.add_vline(x=1, line_color=P["axis"], line_width=1)
style(fig, height=max(260, 24 * len(t) + 90), legend=False,
      title="형상별: 이론상 최소 시간의 몇 배인가 (세로선 = 이론상 최소, 실패한 형상은 빨강)")
fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
fig.update_xaxes(range=[0, float(t["최소의 몇 배"].max()) * 1.15])
chart(fig)

show = t.drop(columns=["_cands"]).assign(통과=t["통과"].map(badge))
st.dataframe(show, hide_index=True, width="stretch", height=min(36 * (len(show) + 1) + 4, 800),
             column_order=["형상", "설명", "통과", "makespan(s)", "최소의 몇 배", "고른 설정", "겹침 점수(%)", "실패 층", "충돌",
                           "층수", "덮은 비율(%)", "넘친 비율(%)", "이론상 최소(s)", "풀이 시간(s)"],
             column_config={"설명": st.column_config.TextColumn(width="medium"),
                            "makespan(s)": st.column_config.NumberColumn(format="%.1f"),
                            "겹침 점수(%)": st.column_config.NumberColumn(format="%.2f"),
                            "덮은 비율(%)": st.column_config.NumberColumn(format="%.2f"),
                            "넘친 비율(%)": st.column_config.NumberColumn(format="%.2f"),
                            "이론상 최소(s)": st.column_config.NumberColumn(format="%.0f"),
                            "최소의 몇 배": st.column_config.NumberColumn(format="%.2f"),
                            "풀이 시간(s)": st.column_config.NumberColumn("풀이 시간(s, 동시 실행)", format="%.1f")})

with st.expander("형상별 후보 설정 비교 (짧은 시험과 끝까지 계획한 makespan)"):
    st.caption("짧은 시험: 고르게 뽑은 4개 층만 계획해 본 makespan. 가장 빠른 것보다 10% 넘게 느린 후보는 끝까지 계획하지 않습니다. "
               "'끝까지'가 비어 있으면 중간에 멈춘 후보입니다(후보를 전부 끝까지 계획한 실험에서만 다 찹니다).")
    sh = st.selectbox("형상", t["형상"], format_func=lambda k: f"{k}  {t.set_index('형상').loc[k, '설명']}")
    row = t.set_index("형상").loc[sh]
    cands = pd.DataFrame([{"후보 설정": k, "짧은 시험(s)": v[0], "끝까지(s)": v[1], "고름": "◀ 고름" if k == row["고른 설정"] else ""}
                          for k, v in row["_cands"].items()])
    st.dataframe(cands, hide_index=True, width="stretch",
                 column_config={"짧은 시험(s)": st.column_config.NumberColumn(format="%.1f"),
                                "끝까지(s)": st.column_config.NumberColumn(format="%.1f")})
