"""층별 점검·움직임 그림·작업 일정이 같은 묶음·실험·형상을 보게, 고른 값을 화면 사이에 넘긴다.

Streamlit 은 화면을 옮기면 그 화면 칸(위젯)의 값을 잊는다. 그래서 고른 값을 따로 적어 두었다가 다른 화면에서 그 칸의
값으로 넣는다. 앞에서 고른 값이 이 화면에 없으면(예: 경로를 모아 두지 않은 실험) 이 화면의 기본값을 보이고, 적어 둔
값은 이 화면에서 직접 바꾸기 전까지 그대로 둔다(다시 앞 화면으로 가면 고른 값이 그대로 보이게).
"""
import streamlit as st

_KEPT = "_pick"     # session_state[_KEPT] = {"set": 묶음 폴더 이름, "exp": 실험 폴더 이름, "shape": 형상 번호}


def kept(kind: str):
    return st.session_state.get(_KEPT, {}).get(kind)


def _copy(kind: str, key: str) -> None:
    st.session_state.setdefault(_KEPT, {})[kind] = st.session_state[key]


def pick(col, label: str, options: list, kind: str, key: str, default: int = 0, **kw):
    """세 화면이 같이 쓰는 고르기 칸. 다른 화면에서 고른 값이 options 에 있으면 그 값으로 연다."""
    want = kept(kind)
    if want in options:
        st.session_state[key] = want
    elif st.session_state.get(key) not in options:
        st.session_state[key] = options[default]
    v = col.selectbox(label, options, key=key, on_change=_copy, args=(kind, key), **kw)
    st.session_state.setdefault(_KEPT, {}).setdefault(kind, v)   # 처음 연 화면의 값부터 적어 둔다
    return v
