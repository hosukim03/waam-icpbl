import streamlit as st

from data import IMAGE_TITLES, SET_NAMES, SET_NOTE, images, shape_desc
from theme import header

header("형상 그림", "tools/ 에 그려 둔 그림입니다(최근 것부터). 그림 위에 마우스를 올리면 오른쪽 위에 크게 보기 단추가 나옵니다.")
imgs = images()
if imgs:
    p = st.selectbox("그림", imgs, format_func=lambda p: f"{IMAGE_TITLES.get(p.name, p.stem)}  ·  {p.name}")
    with st.container(border=True):
        st.image(str(p), width="stretch")
else:
    st.info("tools/ 에 그림이 없습니다.")

st.subheader("묶음별 형상 설명")
s = st.radio("묶음", [k for k in SET_NAMES if k != "tests"], format_func=SET_NAMES.get, horizontal=True)
st.caption(f"**{SET_NAMES[s]}**: {SET_NOTE.get(s, '')}")
d = shape_desc(f"data/{s}")
st.dataframe([{"형상": k, "생성 설정": v} for k, v in d.items()], hide_index=True, width="stretch")
