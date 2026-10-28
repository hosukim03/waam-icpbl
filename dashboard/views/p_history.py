import pandas as pd
import streamlit as st

from data import eval_index, run_table
from theme import header

header("실험 기록", "runs/evals 에 모은 실험 전체입니다. 실험 이름 앞 네 자리는 날짜(월일)입니다. "
       "실험을 새로 모으려면 `tools/collect_evals.py eval <결과 폴더> <이름> <묶음 폴더> \"<설명>\"` 를 돌립니다.")
idx = eval_index()
if idx.empty:
    st.info("runs/evals 에 모은 실험이 없습니다.")
    st.stop()
order = list(dict.fromkeys(idx.sort_values("name")["exp"]))
sets = list(dict.fromkeys(idx["set"]))
labels = dict(zip(idx["set"], idx["set_label"]))
cell, detail = {}, []
for e in idx.to_dict("records"):
    t = run_table(e["name"], e["set_dir"])
    n_ok, ms = int((t["통과"] == "PASS").sum()), t["makespan(s)"].sum()
    cell[(e["exp"], e["set"])] = (n_ok, len(t), ms)
    b = t["이론상 최소(s)"].sum()
    detail.append({"실험": e["exp"], "묶음": e["set_label"], "통과": f"{n_ok} / {len(t)}", "makespan 합계(s)": ms,
                   "이론상 최소의 몇 배": ms / b if b else None, "충돌": int(t["충돌"].fillna(0).sum()), "날짜": e["date"]})

st.subheader("실험 목록")
st.dataframe(pd.DataFrame([{"실험": x, "날짜": idx[idx["exp"] == x]["date"].iloc[0], "무엇을 바꿨나": idx[idx["exp"] == x]["desc"].iloc[0]}
                           for x in order]), hide_index=True, width="stretch")

st.subheader("한눈에: 실험 × 묶음")
st.caption("칸마다 통과 수와 makespan 합계, 괄호 안은 바로 앞 실험 대비 변화입니다. ✅ 모두 통과 · ⚠️ 떨어진 형상 있음. "
           "묶음끼리는 형상이 달라 합계를 서로 비교하지 않습니다.")
rows = []
for i, x in enumerate(order):
    row = {"실험": x}
    for s in sets:
        if (x, s) not in cell:
            row[labels[s]] = "-"
            continue
        n_ok, n, ms = cell[(x, s)]
        prev = next((cell[(y, s)] for y in reversed(order[:i]) if (y, s) in cell), None)
        d = "" if prev is None else f" ({(ms / prev[2] - 1) * 100:+.2f}%)"
        row[labels[s]] = f"{'✅' if n_ok == n else '⚠️'} {n_ok}/{n} · {ms:,.0f} s{d}"
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with st.expander("묶음별 자세히"):
    st.dataframe(pd.DataFrame(detail), hide_index=True, width="stretch", height=36 * (len(detail) + 1) + 4,
                 column_config={"makespan 합계(s)": st.column_config.NumberColumn(format="%.1f"),
                                "이론상 최소의 몇 배": st.column_config.NumberColumn(format="%.2f")})
