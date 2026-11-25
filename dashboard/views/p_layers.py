import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from data import (COVER_MIN, IOU_MIN, OVER_MAX, ROOT, RUNS, eval_index, layer_metrics, make_shape_view, run_table, shape_view,
                  shape_view_layer, shape_view_path, target_mesh)
from theme import badge, chart, header, pal, style

header("층별 점검", "형상 하나를 층마다 채점한 값입니다. 합격하려면 궤적 규칙·팔 닿는 거리·충돌·모양 검사를 모두 넘어야 하고, "
       "모양은 모든 층의 겹침 점수가 98% 이상이어야 합니다(아래 합격 기준 상자).")
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
e = f2.selectbox("실험", g, index=len(g) - 1, format_func=lambda x: f"{x['exp']} — {x['desc']}", key=f"exp_{s}")
t = run_table(e["name"], e["set_dir"]).set_index("형상")
fails = [k for k in t.index if t.loc[k, "통과"] != "PASS"]
sh = f3.selectbox("형상 (떨어진 형상이 먼저)", fails + [k for k in t.index if k not in fails],
                  format_func=lambda k: f"{k}  {badge(t.loc[k, '통과'])}  {t.loc[k, '설명']}", key=f"shape_{e['name']}")
d = layer_metrics(e["name"], sh)
if d.empty:
    st.warning("층별 기록이 없습니다.")
    st.stop()
bad_mask = d["passed"].astype(str).str.lower() != "true"
c = st.columns(4)
c[0].metric("결과", badge(t.loc[sh, "통과"]), border=True)
c[1].metric("기준 미달 층", f"{int(bad_mask.sum())} / {len(d)}", border=True)
low = d["iou"].min() * 100
c[2].metric("가장 낮은 층 겹침 점수", f"{low:.2f}%", delta=f"{low - IOU_MIN * 100:+.2f}%p (기준 {IOU_MIN * 100:.0f}% 이상)",
            border=True)
c[3].metric("makespan", f"{t.loc[sh, 'makespan(s)']:,.1f} s", border=True)

# 합격 기준: 채점기 검사 전부(궤적 규칙·팔 닿는 거리·충돌·모양 4가지)와 이 형상의 값. 하나라도 못 넘으면 실패
ok = lambda b: "✅ 넘음" if b else "⛔ 못 넘음"  # noqa: E731
r = t.loc[sh]
n_bad = int(bad_mask.sum())
ev = RUNS / "evals" / e["name"] / sh
sm = json.loads((ev / "summary.json").read_text(encoding="utf-8")) if (ev / "summary.json").exists() else {}
cp = ev / "config.yaml" if (ev / "config.yaml").exists() else ROOT / e["set_dir"] / sh / "config.yaml"
cf = yaml.safe_load(cp.read_text(encoding="utf-8")) if cp.exists() else {}
pr, vd = cf.get("process", {}), cf.get("validation", {})
tol = float(vd.get("speed_relative_tolerance", 0.1))
reach_mm = min((rb.get("xy_reach_radius_mm", 1500.0) for rb in cf.get("robots", [])), default=1500.0)
n_err = sum(1 for it in sm.get("issues", []) if str(it.get("severity", "")).lower() == "error")
n_col = sm.get("collision", {}).get("collision_event_count")
reach_ok = sm.get("reach", {}).get("passed")
with st.container(border=True):
    st.markdown("**합격 기준: 아래 검사를 모두 넘어야 합격** (하나라도 못 넘으면 그 형상은 실패). "
                f"모양 검사의 핵심은 **모든 층의 겹침 점수가 {IOU_MIN * 100:.0f}% 이상**인지입니다.")
    st.markdown("**겹침 점수**는 목표 모양과 실제로 쌓인 모양이 얼마나 똑같은지입니다. "
                "맞게 쌓인 넓이 ÷ (목표 넓이 + 밖으로 넘친 넓이)로 계산해서, 덜 쌓여도 넘쳐도 내려가고 딱 맞으면 100% 입니다. "
                "예) 목표 100 중 99 를 채우고 밖으로 1 이 넘치면 99 ÷ 101 = 98.0% 로 겨우 합격입니다.")
    rows = ["| 검사 | 합격 조건 | 기준 | 이 형상 | 판정 |", "|---|---|---|---|---|",
            f"| 궤적 규칙 | 쌓기 속도 {pr.get('deposition_speed_mm_s', 8):g} mm/s · 이동 {pr.get('travel_speed_mm_s', 150):g} mm/s "
            f"이하(±{tol * 100:.0f}%), 층 높이에서만 쌓기, 기다릴 때 제자리, 작업대 안에서 쌓기 | 위반 0건 | "
            f"{n_err}건 | {ok(n_err == 0)} |",
            f"| 팔 닿는 거리 | 로봇 받침에서 수평 {reach_mm:,.0f} mm 안 | 넘는 점 0개 | "
            f"{'모두 안' if reach_ok else ('넘는 점 있음' if reach_ok is False else '-')} | "
            f"{ok(bool(reach_ok)) if reach_ok is not None else '-'} |",
            f"| 충돌 | 로봇 팔끼리 부딪히지 않기(여유 {cf.get('collision', {}).get('arm_clearance_mm', 50):g} mm 포함) | 0건 | "
            f"{n_col if n_col is not None else '-'}건 | {ok(n_col == 0) if n_col is not None else '-'} |",
            f"| 모양 | 모든 층의 겹침 점수 | {IOU_MIN * 100:.0f}% 이상 | 가장 낮은 층 {low:.2f}%"
            f"{f' ({n_bad}개 층 미달)' if n_bad else ''} | {ok(n_bad == 0)} |",
            f"| 모양 | 형상 전체 겹침 점수 | {IOU_MIN * 100:.0f}% 이상 | {r['겹침 점수(%)']:.2f}% | "
            f"{ok(r['겹침 점수(%)'] >= IOU_MIN * 100)} |",
            f"| 모양 | 형상 전체 덮은 비율 (= 100% − 빈틈) | {COVER_MIN * 100:.0f}% 이상 | {r['덮은 비율(%)']:.2f}% | "
            f"{ok(r['덮은 비율(%)'] >= COVER_MIN * 100)} |",
            f"| 모양 | 형상 전체 넘침 | {OVER_MAX * 100:.0f}% 이하 | {r['넘친 비율(%)']:.2f}% | "
            f"{ok(r['넘친 비율(%)'] <= OVER_MAX * 100)} |"]
    st.markdown("\n".join(rows))
st.markdown("[↓ 형상에서 보기: 실제 형상(3D)과 층 단면에서 빈틈·넘침 자리 보기](#shape-view)")

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
               annotation_text="넘침 기준 3% (층마다가 아니라 형상 전체에 적용)", annotation_font_color=P["muted"],
               annotation_position="top left")
style(fig2, height=300, title="층별 빈틈과 넘침 (단면 넓이 대비 %)")
fig2.update_layout(barmode="group")
fig2.update_xaxes(title="층 (아래부터)", showgrid=False)
chart(fig2)

if bad_mask.any():
    st.subheader("기준 미달 층")
    st.caption(f"층마다 겹침 점수가 {IOU_MIN * 100:.0f}% 이상이어야 합니다. 빈틈·넘침은 층마다 따로 기준이 없고 형상 전체로만 "
               f"봅니다(위 통과 조건 표). 겹침 점수 = 맞게 쌓인 넓이 ÷ (목표 넓이 + 넘친 넓이).")
    bad = d[bad_mask]
    KS = f"겹침 점수(%, {IOU_MIN * 100:.0f} 이상)"
    GAP = f"{IOU_MIN * 100:.0f}%까지 모자람(%p)"
    st.dataframe(bad.assign(층=bad["layer_index"] + 1, **{KS: 100 * bad["iou"], GAP: 100 * (IOU_MIN - bad["iou"]),
                                                       "빈틈(%)": 100 * bad["underfill_ratio"],
                                                       "넘침(%)": 100 * bad["overfill_ratio"]})
                 [["층", KS, GAP, "빈틈(%)", "넘침(%)", "target_area_mm2"]]
                 .rename(columns={"target_area_mm2": "단면 넓이(mm²)"}),
                 hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.2f") for k in (KS, "빈틈(%)", "넘침(%)")} |
                               {GAP: st.column_config.NumberColumn(format="%.2f"),
                                "단면 넓이(mm²)": st.column_config.NumberColumn(format="%.0f")})

# ------------------------------------------------------------------ 형상에서 보기
st.subheader("형상에서 보기: 어디가 비고 어디가 넘쳤나", anchor="shape-view")
st.caption("채점기와 같은 방법으로 층마다 목표 단면과 쌓인 면(불 켜고 지나간 길을 비드 폭 8 mm 로 부풀린 것)을 다시 만들어 "
           "겹쳤습니다. 왼쪽은 실제 형상(목표 STL) 위에 빈틈·넘침 자리를, 오른쪽은 고른 층의 단면을 보여 줍니다. "
           "둘 다 끌어서 돌리거나 옮기고 휠로 확대합니다. 단면은 두 번 누르면 처음 크기로 돌아갑니다.")
zp = shape_view_path(e["name"], sh)
if not zp.exists():
    if not (RUNS / "evals" / e["name"] / sh / "trajectory.csv.gz").exists():
        st.info("이 실험은 궤적을 같이 모으지 않아 그릴 수 없습니다(tools/collect_evals.py … --traj).")
        st.stop()
    st.info("아직 계산하지 않은 형상입니다. 형상 크기에 따라 몇 초에서 몇 분 걸립니다(층이 350개인 공개 05 는 약 4분).")
    if st.button("형상 그림 계산하기", type="primary"):
        with st.spinner("채점기 방식으로 층마다 단면을 만드는 중…"):
            ok, msg = make_shape_view(e["name"], e["set_dir"], sh)
        if ok:
            st.rerun()
        st.error(msg)
    st.stop()

mt = zp.stat().st_mtime
V = shape_view(e["name"], sh, mt)
LY = {ly["L"]: ly for ly in V["layers"]}
order = [ly["L"] for ly in V["layers"]]
worst = min(V["layers"], key=lambda ly: ly["iou"])["L"]
UNDER_C, OVER_C = P["series"][1], P["series"][0]          # 위 막대그래프와 같은 색: 빈틈 주황, 넘침 파랑
KIND = {"under": "빈틈", "over": "넘침"}

c1, c2, c3 = st.columns([1.3, 2.2, 1])
L = c1.selectbox("층 (처음엔 점수가 가장 낮은 층)", order, index=order.index(worst),
                 format_func=lambda k: f"{k + 1}층  {LY[k]['iou'] * 100:.2f}%  {'✅' if LY[k]['passed'] else '⛔'}",
                 key=f"sv_layer_{e['name']}_{sh}")
spots = LY[L]["big"]
spot = c2.selectbox("자리로 확대 (그 층의 큰 조각부터)", [-1] + list(range(len(spots))),
                    format_func=lambda i: "층 전체" if i < 0 else
                    f"{KIND[spots[i]['kind']]} {spots[i]['area']:.1f} mm²  · 굵기 약 {spots[i]['w']:.1f} mm  "
                    f"· ({spots[i]['cx']:.0f}, {spots[i]['cy']:.0f})")
show_path = c3.toggle("불 켜고 지나간 길", value=True)
M = target_mesh(e["set_dir"], sh)
rng = [float(np.ptp(M[a])) if M is not None else 1.0 for a in "xyz"]
flat = rng[2] / max(rng) < 0.18                             # 넓적한 형상은 층이 안 보이니 높이를 늘려 볼 수 있게
tall = c3.toggle("높이 늘려 보기 (3D)", value=flat, disabled=not flat,
                 help="넓적한 형상은 실제 비율이면 층이 얇아 안 보여서, 높이를 넓이의 18% 까지 늘려 그립니다.")
G = shape_view_layer(e["name"], sh, L, mt)


def ring_xy(polys: list, z: float | None = None):
    """고리들을 None 으로 끊어 이은 좌표(닫힌 선)."""
    xs, ys = [], []
    for rings in polys:
        for r in rings:
            a = np.asarray(r).reshape(-1, 2)
            xs += a[:, 0].tolist() + [a[0, 0], None]
            ys += a[:, 1].tolist() + [a[0, 1], None]
    zs = None if z is None else [None if v is None else z for v in xs]
    return xs, ys, zs


def svg_path(polys: list) -> str:
    """구멍까지 그리는 SVG 경로(채우기는 even-odd 규칙)."""
    out = []
    for rings in polys:
        for r in rings:
            a = np.asarray(r).reshape(-1, 2)
            out.append("M" + "L".join(f"{x:.2f},{y:.2f}" for x, y in a) + "Z")
    return "".join(out)


left, right = st.columns([1, 1])
# ---- 3D: 실제 형상 + 빈틈·넘침 판(그 층 윗면 높이) + 기준 미달 층 테두리 + 고른 층
f3 = go.Figure()
if M is not None:
    f3.add_trace(go.Mesh3d(x=M["x"], y=M["y"], z=M["z"], i=M["i"], j=M["j"], k=M["k"], name="목표 형상",
                           showlegend=True, color=P["muted"], opacity=0.22, flatshading=True, hoverinfo="skip",
                           lighting=dict(ambient=0.65, diffuse=0.6, specular=0.05, roughness=0.9)))
for kind, col in (("under", UNDER_C), ("over", OVER_C)):
    xs, ys, zs, cd = [], [], [], []
    for ly in V["layers"]:
        t = np.asarray(ly[f"{kind}_tri"], float).reshape(-1, 6)
        if not len(t):
            continue
        xs.append(t[:, 0::2].ravel())
        ys.append(t[:, 1::2].ravel())
        zs.append(np.full(t.shape[0] * 3, ly["z1"]))
        cd.append(np.full(t.shape[0] * 3, ly["L"] + 1))
    if xs:
        x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
        n = len(x) // 3
        f3.add_trace(go.Mesh3d(x=x, y=y, z=z, i=np.arange(n) * 3, j=np.arange(n) * 3 + 1, k=np.arange(n) * 3 + 2,
                               name=f"{KIND[kind]} ({'덜 쌓임' if kind == 'under' else '밖에 쌓임'})", showlegend=True,
                               color=col, opacity=1.0, flatshading=True, customdata=np.concatenate(cd),
                               hovertemplate=f"%{{customdata}}층 {KIND[kind]}<extra></extra>",
                               lighting=dict(ambient=0.9, diffuse=0.3)))
bad = [ly for ly in V["layers"] if not ly["passed"]]
if bad and M is not None:
    # 기준 미달 층의 높이: 형상 옆 모서리에 세운 빨간 막대(이어진 층은 한 막대로)
    x0, y0 = float(M["x"].min()) - 0.03 * max(rng), float(M["y"].min()) - 0.03 * max(rng)
    runs, cur = [], None
    for ly in bad:
        if cur and abs(ly["z0"] - cur[1]) < 1e-6:
            cur[1] = ly["z1"]
        else:
            cur = [ly["z0"], ly["z1"]]
            runs.append(cur)
    rz = sum(([a, b, None] for a, b in runs), [])
    f3.add_trace(go.Scatter3d(x=[x0 if v is not None else None for v in rz], y=[y0 if v is not None else None for v in rz],
                              z=rz, mode="lines", name=f"⛔ 기준 미달 층 높이 ({len(bad)}개 층)",
                              line=dict(color=P["critical"], width=10), hoverinfo="skip"))
    if len(bad) <= 10:                                     # 몇 개 층만 떨어졌으면 그 층 테두리도 그린다
        bx, by, bz = [], [], []
        for ly in bad:
            x, y, z = ring_xy(ly["outline"], ly["z1"])
            bx += x
            by += y
            bz += z
        f3.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", name="⛔ 기준 미달 층 테두리",
                                  line=dict(color=P["critical"], width=3), hoverinfo="skip"))
if G is not None:
    x, y, z = ring_xy(G["target"], LY[L]["z1"])
    f3.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=f"고른 층 ({L + 1}층)",
                              line=dict(color=P["ink"], width=5), hoverinfo="skip"))
ax3 = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=P["grid"], zerolinecolor=P["grid"], color=P["muted"],
           showspikes=False)
m = max(rng)
zr = max(rng[2] / m, 0.18) if tall else rng[2] / m
k_up = zr / (rng[2] / m) if rng[2] > 0 else 1.0
style(f3, height=640, legend=True,
      title="실제 형상에서 본 빈틈·넘침 자리" + (f" (높이 ×{k_up:.1f} 로 늘려 그림)" if k_up > 1.05 else ""))
f3.update_layout(scene=dict(xaxis=dict(title="x (mm)", **ax3), yaxis=dict(title="y (mm)", **ax3),
                            zaxis=dict(title="z (mm)", **ax3), aspectmode="manual",
                            aspectratio=dict(x=rng[0] / m, y=rng[1] / m, z=zr),
                            camera=dict(eye=dict(x=0.95, y=-1.3, z=0.8), center=dict(x=0, y=0, z=-0.12))),
                 uirevision=f"{e['name']}/{sh}/{tall}", margin=dict(l=0, r=0, t=40, b=0),
                 legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0, font=dict(color=P["ink2"])))
with left:
    st.plotly_chart(f3, width="stretch", theme=None, key="sv3d", config={"displaylogo": False})

# ---- 단면: 쌓인 곳(회색) · 빈틈(주황) · 넘침(파랑) · 목표 테두리(검정) · 불 켠 길(점선)
ly = LY[L]
f2 = go.Figure()
style(f2, height=640, legend=True,
      title=f"{L + 1}층 단면 · 겹침 점수 {ly['iou'] * 100:.2f}% · 빈틈 {ly['under_area']:,.0f} mm² · 넘침 {ly['over_area']:,.0f} mm²")
if G is not None:
    shapes = [dict(type="path", path=svg_path(G["dep"]), fillcolor=P["deemph"], line=dict(width=0), fillrule="evenodd",
                   layer="below", name="쌓인 곳", showlegend=True),
              dict(type="path", path=svg_path(G["under"]), fillcolor=UNDER_C, line=dict(width=0.6, color=UNDER_C),
                   fillrule="evenodd", name="빈틈 (덜 쌓임)", showlegend=True),
              dict(type="path", path=svg_path(G["over"]), fillcolor=OVER_C, line=dict(width=0.6, color=OVER_C),
                   fillrule="evenodd", name="넘침 (밖에 쌓임)", showlegend=True)]
    f2.update_layout(shapes=[s for s in shapes if s["path"]])
    x, y, _ = ring_xy(G["target"])
    f2.add_trace(go.Scatter(x=x, y=y, mode="lines", name="목표 단면 테두리", line=dict(color=P["ink"], width=1.3),
                            hoverinfo="skip"))
    if show_path and G["path"]:
        px, py = [], []
        for pl in G["path"]:
            a = np.asarray(pl).reshape(-1, 2)
            px += a[:, 0].tolist() + [None]
            py += a[:, 1].tolist() + [None]
        f2.add_trace(go.Scatter(x=px, y=py, mode="lines", name="불 켜고 지나간 길 (비드 가운데)",
                                line=dict(color=P["ink2"], width=1, dash="dot"), hoverinfo="skip"))
    if spots:                                              # 큰 조각 가운데에 동그라미(작은 조각도 찾기 쉽게)와 설명
        f2.add_trace(go.Scatter(x=[b["cx"] for b in spots], y=[b["cy"] for b in spots], mode="markers",
                                name="큰 조각 자리", marker=dict(size=9, color="rgba(0,0,0,0)",
                                                              line=dict(width=1.5, color=P["ink"])),
                                text=[f"{KIND[b['kind']]} {b['area']:.1f} mm²<br>굵기 약 {b['w']:.1f} mm" for b in spots],
                                hovertemplate="%{text}<br>(%{x:.1f}, %{y:.1f})<extra></extra>"))
f2.update_yaxes(scaleanchor="x", scaleratio=1, title="y (mm)")
f2.update_xaxes(title="x (mm)")
if spot >= 0:
    b = spots[spot]
    r = max(12.0, 0.75 * max(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]))
    f2.update_xaxes(range=[b["cx"] - r, b["cx"] + r])
    f2.update_yaxes(range=[b["cy"] - r, b["cy"] + r])
f2.update_layout(uirevision=f"{e['name']}/{sh}/{L}/{spot}", margin=dict(l=8, r=16, t=40, b=100),
                 legend=dict(orientation="h", yanchor="top", y=-0.13, xanchor="left", x=0, font=dict(color=P["ink2"])))
with right:
    st.plotly_chart(f2, width="stretch", theme=None, key="sv2d",
                    config={"displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
st.caption(f"3D 의 판은 그 층 윗면 높이에 놓았고 1 mm² 넘는 조각만 그렸습니다. 단면은 0.3 mm² 넘는 조각만 칠했지만 넓이 합에는 "
           f"모든 조각이 들어 있습니다. 계산 {V['secs']} s (tools/shape_diff.py, 결과는 실험 폴더의 layer_shapes.zip).")
