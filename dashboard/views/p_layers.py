import plotly.graph_objects as go
import streamlit as st

from data import IOU_MIN, OVER_MAX, eval_index, layer_metrics, run_table
from theme import badge, chart, header, pal, style

header("층별 점검", "형상 하나를 층마다 채점한 값입니다. 채점기는 층마다 겹침 점수가 98% 이상이어야 통과시킵니다.")
idx = eval_index()
if idx.empty:
    st.info("runs/evals 에 모은 실험이 없습니다.")
    st.stop()
P = pal()
sets = list(dict.fromkeys(idx["set"]))
labels = dict(zip(idx["set"], idx["set_label"]))
f1, f2, f3 = st.columns([1, 2, 2])
s = f1.selectbox("묶음", sets, format_func=labels.get)
g = idx[idx["set"] == s].sort_values("name").to_dict("records")
e = f2.selectbox("실험", g, index=len(g) - 1, format_func=lambda x: f"{x['exp']} — {x['desc']}")
t = run_table(e["name"], e["set_dir"]).set_index("형상")
fails = [k for k in t.index if t.loc[k, "통과"] != "PASS"]
sh = f3.selectbox("형상 (떨어진 형상이 먼저)", fails + [k for k in t.index if k not in fails],
                  format_func=lambda k: f"{k}  {badge(t.loc[k, '통과'])}  {t.loc[k, '설명']}")
d = layer_metrics(e["name"], sh)
if d.empty:
    st.warning("층별 기록이 없습니다.")
    st.stop()
bad_mask = d["passed"].astype(str).str.lower() != "true"
c = st.columns(4)
c[0].metric("결과", badge(t.loc[sh, "통과"]), border=True)
c[1].metric("기준 미달 층", f"{int(bad_mask.sum())} / {len(d)}", border=True)
c[2].metric("가장 낮은 층 겹침 점수", f"{d['iou'].min() * 100:.2f}%", border=True)
c[3].metric("makespan", f"{t.loc[sh, 'makespan(s)']:,.1f} s", border=True)

x = d["layer_index"] + 1
fig = go.Figure()
fig.add_scatter(x=x, y=100 * d["iou"], name="겹침 점수", mode="lines", line=dict(color=P["series"][0]),
                hovertemplate="층 %{x}<br>겹침 점수 %{y:.2f}%<extra></extra>")
fig.add_scatter(x=x, y=100 * d["coverage"], name="덮은 비율", mode="lines", line=dict(color=P["series"][2]),
                hovertemplate="층 %{x}<br>덮은 비율 %{y:.2f}%<extra></extra>")
if bad_mask.any():
    fig.add_scatter(x=x[bad_mask], y=100 * d["iou"][bad_mask], name="⛔ 기준 미달 층", mode="markers",
                    marker=dict(size=9, color=P["critical"], line=dict(width=2, color=P["surface"])),
                    hovertemplate="⛔ 층 %{x}: 겹침 점수 %{y:.2f}%<extra></extra>")
fig.add_hline(y=100 * IOU_MIN, line_dash="dash", line_color=P["muted"], line_width=1,
              annotation_text="기준 98%", annotation_font_color=P["muted"], annotation_position="bottom left")
style(fig, height=360, title="층별 겹침 점수와 덮은 비율 (%)")
fig.update_xaxes(title="층 (아래부터)")
chart(fig)

fig2 = go.Figure()
fig2.add_bar(x=x, y=100 * d["underfill_ratio"], name="빈틈 (덜 쌓임)", marker_color=P["series"][1],
             hovertemplate="층 %{x}<br>빈틈 %{y:.2f}%<extra></extra>")
fig2.add_bar(x=x, y=100 * d["overfill_ratio"], name="넘침 (밖에 쌓임)", marker_color=P["series"][0],
             hovertemplate="층 %{x}<br>넘침 %{y:.2f}%<extra></extra>")
fig2.add_hline(y=100 * OVER_MAX, line_dash="dash", line_color=P["muted"], line_width=1,
               annotation_text="넘침 기준 3%", annotation_font_color=P["muted"], annotation_position="top left")
style(fig2, height=300, title="층별 빈틈과 넘침 (단면 넓이 대비 %)")
fig2.update_layout(barmode="group")
fig2.update_xaxes(title="층 (아래부터)", showgrid=False)
chart(fig2)

if bad_mask.any():
    st.subheader("기준 미달 층")
    bad = d[bad_mask]
    st.dataframe(bad.assign(층=bad["layer_index"] + 1, **{"겹침 점수(%)": 100 * bad["iou"], "빈틈(%)": 100 * bad["underfill_ratio"],
                                                       "넘침(%)": 100 * bad["overfill_ratio"]})
                 [["층", "겹침 점수(%)", "빈틈(%)", "넘침(%)", "target_area_mm2"]]
                 .rename(columns={"target_area_mm2": "단면 넓이(mm²)"}),
                 hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.2f") for k in ("겹침 점수(%)", "빈틈(%)", "넘침(%)")} |
                               {"단면 넓이(mm²)": st.column_config.NumberColumn(format="%.0f")})
