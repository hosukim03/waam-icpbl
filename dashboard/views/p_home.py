import plotly.graph_objects as go
import streamlit as st

from data import COVER_MIN, IOU_MIN, OVER_MAX, bounds, eval_index, official_reports, read_report, run_table
from theme import badge, chart, header, hero, pal, style

header("공개 형상 공식 결과",
       "교수님이 공개한 형상 6개를 코어 1개로 풀고 채점기로 채점한 값입니다. 풀이는 고성능 코어 2번 하나만 쓰게 고정하고 "
       "충전기를 꽂은 채로 `tools/report_public.py --cpus 2` 로 잽니다. 계산 시간은 채점을 뺀 풀이 시간입니다.")
reps = official_reports()
if not reps:
    st.info("results/ 에 report_public_*_1core.csv 가 없습니다.")
    st.stop()
label = lambda p: p.stem.replace("report_public_", "").replace("_1core", "")  # noqa: E731
P = pal()
left, right = st.columns([3, 1], vertical_alignment="bottom")
p = right.selectbox("측정 기록", reps, format_func=lambda p: f"{label(p)} (코어 1개)")
i = reps.index(p)
df = read_report(p)
b = bounds("tests")
df["이론상 최소(s)"] = df["모델"].map(lambda m: b.get(m, {}).get("bound_s"))
df["최소의 몇 배"] = df["makespan(s)"] / df["이론상 최소(s)"]
prev = read_report(reps[i + 1]) if i + 1 < len(reps) else None
ms, ct = df["makespan(s)"].sum(), df["계산 시간(s)"].sum()
with left:
    if prev is not None:
        dm = ms / prev["makespan(s)"].sum() - 1
        hero("공개 6개 makespan 합계", f"{ms:,.1f} s", f"{'▼' if dm < 0 else '▲'} {abs(dm) * 100:.2f}%  앞 기록({label(reps[i + 1])}) 대비",
             good=dm < 0 if abs(dm) > 1e-9 else None)
    else:
        hero("공개 6개 makespan 합계", f"{ms:,.1f} s")

c = st.columns(4)
dt = None if prev is None else f"{(ct / prev['계산 시간(s)'].sum() - 1) * 100:+.1f}%"
c[0].metric("계산 시간 합계 (코어 1개)", f"{ct:,.1f} s", dt, delta_color="inverse", border=True)
c[1].metric("통과", f"{(df['통과'] == 'PASS').sum()} / {len(df)}", border=True)
c[2].metric("충돌", f"{int(df['충돌'].sum())}", border=True)
c[3].metric("이론상 최소의 몇 배", f"{ms / df['이론상 최소(s)'].sum():.2f}", border=True,
            help="겹침·이동·대기 없이, 넓은 층은 세 대가, 좁은 층(가장 먼 두 점이 200 mm 이하)은 한 대가 쌓는다고 친 시간 대비. "
                 "실제로는 팔 거리 때문에 세 대가 다 못 붙는 일이 많아 낙관적인 값이다.")

st.subheader("모델별 결과")
l, r = st.columns([3, 2])
view = df.assign(통과=df["통과"].map(badge))[["모델", "makespan(s)", "계산 시간(s)", "통과", "충돌", "겹침 점수", "최소의 몇 배"]]
l.dataframe(view, hide_index=True, width="stretch",
            column_config={"makespan(s)": st.column_config.NumberColumn(format="%.1f"),
                           "계산 시간(s)": st.column_config.NumberColumn(format="%.1f"),
                           "겹침 점수": st.column_config.NumberColumn(format="%.4f"),
                           "최소의 몇 배": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=4)})
fig = go.Figure(go.Bar(y=df["모델"], x=df["최소의 몇 배"], orientation="h", marker_color=P["series"][0],
                       text=df["최소의 몇 배"].map(lambda v: f"{v:.2f}"), textposition="outside", textfont=dict(color=P["ink2"]),
                       hovertemplate="모델 %{y}<br>이론상 최소의 %{x:.2f}배<extra></extra>"))
fig.add_vline(x=1, line_color=P["axis"], line_width=1)
style(fig, height=300, legend=False, title="이론상 최소 시간의 몇 배인가")
fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
fig.update_xaxes(range=[0, float(df["최소의 몇 배"].max()) * 1.18])
with r:
    chart(fig)
    st.caption("세로선이 이론상 최소(1배)입니다. 1에 가까울수록 더 줄일 여지가 적습니다.")

st.subheader("공식 측정 기록")
hist = []
for q in reversed(reps):
    d = read_report(q)
    hist.append({"기록": label(q), "makespan 합계(s)": d["makespan(s)"].sum(), "계산 시간 합계(s)": d["계산 시간(s)"].sum()})
l, r = st.columns(2)
for col, key, ttl in ((l, "makespan 합계(s)", "makespan 합계 (s)"), (r, "계산 시간 합계(s)", "계산 시간 합계 (s, 코어 1개)")):
    f = go.Figure(go.Scatter(x=[h["기록"] for h in hist], y=[h[key] for h in hist], mode="lines+markers+text",
                             line=dict(color=P["series"][0]), marker=dict(size=9, line=dict(width=2, color=P["surface"])),
                             text=["" for _ in hist[:-1]] + [f"{hist[-1][key]:,.1f}"], textposition="top center",
                             textfont=dict(color=P["ink"]), hovertemplate="%{x}<br>%{y:,.1f} s<extra></extra>"))
    style(f, height=260, legend=False, title=ttl)
    with col:
        chart(f)

with st.expander("채점 기준 (채점기 v1.0.2)"):
    st.markdown(f"- 겹침 점수(IoU: 겹친 넓이 ÷ 목표와 쌓은 곳을 합친 넓이) **{IOU_MIN:.0%} 이상**, 층마다와 전체 모두\n"
                f"- 덮은 비율(목표 중 비드가 덮은 비율) **{COVER_MIN:.0%} 이상**, 넘친 비율(목표 밖에 쌓인 비율) **{OVER_MAX:.0%} 이하**\n"
                "- 로봇 팔끼리 거리 250 mm 초과, 용접봉 끝끼리 200 mm 초과(0.1초마다 검사), 충돌 0\n"
                "- 쌓는 속도 8 mm/s ± 10%, 이동 속도 165 mm/s 이하")

st.subheader("묶음별 최근 실험")
idx = eval_index()
if idx.empty:
    st.info("runs/evals 에 모은 실험이 없습니다(tools/collect_evals.py).")
else:
    rows = []
    for s, g in idx.groupby("set", sort=False):
        e = g.sort_values("name").iloc[-1]
        t = run_table(e["name"], e["set_dir"])
        bb = t["이론상 최소(s)"].sum()
        rows.append({"묶음": e["set_label"], "실험": e["exp"], "설명": e["desc"], "통과": f"{(t['통과'] == 'PASS').sum()} / {len(t)}",
                     "makespan 합계(s)": t["makespan(s)"].sum(), "이론상 최소의 몇 배": t["makespan(s)"].sum() / bb if bb else None})
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"makespan 합계(s)": st.column_config.NumberColumn(format="%.1f"),
                                "이론상 최소의 몇 배": st.column_config.NumberColumn(format="%.2f")})
