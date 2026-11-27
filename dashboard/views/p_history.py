import pandas as pd
import streamlit as st

from data import eval_index, run_table
from theme import header

NB = chr(0xA0)  # 줄바꿈 없는 공백(no-break space)

header("실험 기록", "runs/evals 에 모은 실험 전체입니다. 실험은 한 순서대로 번호를 붙였고, 기록 폴더 이름은 실험 목록 표에 있습니다. "
       "실험을 새로 모으려면 `tools/collect_evals.py eval <결과 폴더> <이름> <묶음 폴더> \"<설명>\"` 를 돌립니다.")
idx = eval_index()
if idx.empty:
    st.info("runs/evals 에 모은 실험이 없습니다.")
    st.stop()
order = list(dict.fromkeys(idx.sort_values("name")["exp"]))
name_of = dict(zip(idx["exp"], idx["exp_label"]))
sets = list(dict.fromkeys(idx["set"]))
labels = dict(zip(idx["set"], idx["set_label"]))
cell, detail = {}, []
for e in idx.to_dict("records"):
    t = run_table(e["name"], e["set_dir"])
    n_ok, ms = int((t["통과"] == "PASS").sum()), t["makespan(s)"].sum()
    cell[(e["exp"], e["set"])] = (n_ok, len(t), ms)
    b = t["이론상 최소(s)"].sum()
    detail.append({"실험": e["exp_label"], "묶음": e["set_label"], "통과": f"{n_ok} / {len(t)}", "makespan 합계(s)": ms,
                   "이론상 최소의 몇 배": ms / b if b else None, "충돌": int(t["충돌"].fillna(0).sum()), "날짜": e["date"]})

st.subheader("실험 목록")
st.dataframe(pd.DataFrame([{"실험": name_of[x], "날짜": idx[idx["exp"] == x]["date"].iloc[0],
                            "무엇을 바꿨나": idx[idx["exp"] == x]["desc"].iloc[0], "기록 폴더": x}
                           for x in order]), hide_index=True, width="stretch")

st.subheader("한눈에: 실험 × 묶음")
st.caption("칸마다 통과 수와 makespan 합계, 괄호 안은 바로 앞 실험 대비 변화입니다. ✅ 모두 통과 · ⚠️ 떨어진 형상 있음. "
           "묶음끼리는 형상이 달라 합계를 서로 비교하지 않습니다.")
rows = []
for i, x in enumerate(order):
    row = {"실험": name_of[x]}
    for s in sets:
        if (x, s) not in cell:
            row[labels[s]] = ":gray[–]"  # 그냥 "-" 는 Markdown 목록 점(•)이 된다
            continue
        n_ok, n, ms = cell[(x, s)]
        prev = next((cell[(y, s)] for y in reversed(order[:i]) if (y, s) in cell), None)
        d = "" if prev is None else f" :gray[({(ms / prev[2] - 1) * 100:+.2f}%)]"
        # 칸이 좁으면 괄호 앞에서만 줄이 바뀌게, 나머지 띄어쓰기는 줄바꿈 없는 공백(NB)으로
        row[labels[s]] = f"{'✅' if n_ok == n else '⚠️'}{NB}{n_ok}/{n}{NB}·{NB}{ms:,.0f}{NB}s{d}"
    rows.append(row)
# st.dataframe 은 칸 글자를 줄바꿈하지 않아 묶음이 늘면 오른쪽 칸이 화면 밖으로 잘린다 → 줄바꿈되는 st.table
st.table(pd.DataFrame(rows), hide_index=True)

with st.expander("묶음별 자세히"):
    st.dataframe(pd.DataFrame(detail), hide_index=True, width="stretch", height=36 * (len(detail) + 1) + 4,
                 column_config={"makespan 합계(s)": st.column_config.NumberColumn(format="%.1f"),
                                "이론상 최소의 몇 배": st.column_config.NumberColumn(format="%.2f")})
