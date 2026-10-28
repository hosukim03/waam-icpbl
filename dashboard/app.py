"""WAAM 개발자 대시보드 (혼자 쓰는 개발용).

실행:  streamlit run dashboard/app.py --server.port 8503      (Anaconda 파이썬, 폴더 C:\\projects\\waam 에서)

알고리즘을 고칠 때마다 결과가 어떻게 바뀌었는지 확인하는 곳이다. 자료는 tools/collect_evals.py 가 runs/ 에 모은
평가·계산 시간 기록과 results/ 의 공식 측정 CSV 다. runs/ 는 git 이 추적하지 않는다(이 컴퓨터에만 있다).
"""
import datetime
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, RUNS  # noqa: E402
from theme import inject_css  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
st.set_page_config(page_title="WAAM 개발 대시보드", page_icon=str(ASSETS / "icon.svg"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))
inject_css()


def stamp(p: Path) -> str:
    return datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M") if p.exists() else "-"


page = st.navigation({
    "개요": [st.Page("views/p_home.py", title="공개 형상 공식 결과", icon="🏁", default=True),
           st.Page("views/p_history.py", title="실험 기록", icon="🗂")],
    "실험 결과": [st.Page("views/p_runs.py", title="묶음 결과", icon="📋"),
              st.Page("views/p_compare.py", title="실험 비교", icon="🔁"),
              st.Page("views/p_timing.py", title="계산 시간 (코어 1개)", icon="⏱"),
              st.Page("views/p_layers.py", title="층별 점검", icon="🧱")],
    "형상": [st.Page("views/p_shapes.py", title="형상 그림", icon="🖼")],
}, expanded=True)
with st.sidebar:
    reps = sorted((ROOT / "results").glob("report_public_*_1core.csv"))
    st.caption(f"INE4107 IC-PBL · 로봇 3대 경로 계획  \n평가 기록 갱신 {stamp(RUNS / 'evals' / 'index.json')}  \n"
               f"계산 시간 기록 갱신 {stamp(RUNS / 'timing' / 'index.json')}  \n"
               f"공식 측정 {reps[-1].stem.replace('report_public_', '').replace('_1core', '') if reps else '-'}")
    if st.button("자료 다시 읽기", width="stretch"):
        st.cache_data.clear()
        st.rerun()
page.run()
