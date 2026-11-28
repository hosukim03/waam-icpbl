import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import SET_NOTE, SETTING_NOTE, eval_index, run_table, setting_name
from theme import badge, chart, header, pal, shape_note, shape_option, style

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
e = f2.selectbox("실험", g.to_dict("records"), index=len(g) - 1, format_func=lambda r: r['exp_label'],
                 key=f"exp_{s}")
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
    customdata=t[["모양", "makespan(s)", "통과"]].assign(통과=t["통과"].map(badge)).values,
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
             column_order=["형상", "모양", "통과", "makespan(s)", "최소의 몇 배", "계획", "겹침 점수(%)", "실패 층", "충돌",
                           "층수", "덮은 비율(%)", "넘친 비율(%)", "이론상 최소(s)", "풀이 시간(s)", "설명"],
             column_config={"설명": st.column_config.TextColumn("형상 설정", width="medium"),
                            "makespan(s)": st.column_config.NumberColumn(format="%.1f"),
                            "겹침 점수(%)": st.column_config.NumberColumn(format="%.2f"),
                            "덮은 비율(%)": st.column_config.NumberColumn(format="%.2f"),
                            "넘친 비율(%)": st.column_config.NumberColumn(format="%.2f"),
                            "이론상 최소(s)": st.column_config.NumberColumn(format="%.0f"),
                            "최소의 몇 배": st.column_config.NumberColumn(format="%.2f"),
                            "풀이 시간(s)": st.column_config.NumberColumn("풀이 시간(s, 동시 실행)", format="%.1f")})
st.caption(SETTING_NOTE)

with st.expander("형상별 후보 계획 비교 (짧은 시험과 끝까지 짠 makespan)"):
    st.caption("이 형상에서 후보 계획 중 하나를 고른 과정입니다.\n\n"
               "1. **짧은 시험**: 모든 층을 짜면 오래 걸려서, 높이별로 고르게 뽑은 4개 층만 계획마다 먼저 짜 보고 "
               "걸리는 시간을 잽니다(전체 시간이 아니라 순서를 정하는 값, 층이 8개 이하면 이 단계 없이 모두 끝까지). 가장 빠른 "
               "계획보다 10% 넘게 느린 계획은 여기서 뺍니다.\n"
               "2. **끝까지**: 짧은 시험에서 빠른 계획부터 모든 층을 짭니다. 도중에 앞서 끝낸 계획보다 느려질 게 "
               "확실해지거나 계산량 한도에 닿으면 그 자리에서 멈추므로 칸이 빕니다(분석용으로 모든 계획을 끝까지 짠 실험은 다 찹니다).\n"
               "3. 끝까지 간 계획 중 가장 빨리 끝나는 것을 고릅니다(◀ 고름).")
    ti = t.set_index("형상")
    sh = st.selectbox("형상", t["형상"], format_func=lambda k: shape_option(k, ti.loc[k]))
    row = ti.loc[sh]
    st.caption(shape_note(row))
    cands = pd.DataFrame([{"후보 계획": setting_name(k), "짧은 시험(s)": v[0], "끝까지(s)": v[1],
                           "고름": "◀ 고름" if k == row["계획 코드"] else ""}
                          for k, v in row["_cands"].items()])
    st.dataframe(cands, hide_index=True, width="stretch", placeholder="",   # 멈춘 후보의 '끝까지'는 빈칸(None 대신)
                 column_config={"짧은 시험(s)": st.column_config.NumberColumn(format="%.1f"),
                                "끝까지(s)": st.column_config.NumberColumn(format="%.1f")})
