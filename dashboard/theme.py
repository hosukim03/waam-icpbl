"""대시보드 공통 모양: 색(검증한 기본 팔레트), 글꼴, 숫자 카드, plotly 그래프 꾸미기.

색은 dataviz 기본 팔레트의 앞 세 칸(파랑·주황·청록)만 쓴다. 밝은·어두운 화면 모두 색약 시뮬레이션으로 검사했다
(validate_palette.py --pairs all: 밝은 화면 최악 ΔE 9.2, 어두운 화면 9.4). 밝은 화면의 청록은 배경 대비가 3:1 아래라
그래프마다 표를 같이 둔다. 통과·실패 같은 상태 색은 따로 두고, 색만이 아니라 아이콘과 글자를 함께 붙인다.
"""
import plotly.graph_objects as go
import streamlit as st

FONT = '"Pretendard", "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", system-ui, sans-serif'
PAL = {
    "light": {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"], "deemph": "#c3c2b7",
              "good": "#0ca30c", "warn": "#fab219", "critical": "#d03b3b", "better": "#2a78d6", "worse": "#e34948",
              "mid": "#f0efec", "good_text": "#006300"},
    "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"], "deemph": "#383835",
             "good": "#0ca30c", "warn": "#fab219", "critical": "#d03b3b", "better": "#3987e5", "worse": "#e66767",
             "mid": "#383835", "good_text": "#0ca30c"},
}


def mode() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def pal() -> dict:
    return PAL["dark" if mode() == "dark" else "light"]


def inject_css() -> None:
    st.markdown(f"""<style>
html, body, [class*="css"], .stMarkdown, .stMetric, button, input, textarea {{ font-family: {FONT}; }}
.block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1400px; }}
h1 {{ font-size: 1.7rem !important; font-weight: 700 !important; letter-spacing: -0.01em; margin-bottom: 0.1rem !important; }}
h2, h3 {{ font-weight: 650 !important; letter-spacing: -0.005em; }}
h3 {{ font-size: 1.12rem !important; margin-top: 1.4rem !important; }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 600; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.86rem; }}
[data-testid="stCaptionContainer"] {{ line-height: 1.5; }}
.waam-hero {{ font-size: 3.1rem; font-weight: 650; line-height: 1.1; margin: 0.2rem 0 0.1rem; }}
.waam-hero-label {{ font-size: 0.92rem; opacity: 0.75; }}
.waam-hero-delta {{ font-size: 0.95rem; font-weight: 600; }}
.waam-kicker {{ font-size: 0.8rem; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.6; margin-bottom: 0.2rem; }}
</style>""", unsafe_allow_html=True)


def header(title: str, subtitle: str, kicker: str = "WAAM 다중 로봇 경로 계획") -> None:
    st.markdown(f'<div class="waam-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def hero(label: str, value: str, delta: str | None = None, good: bool | None = None, note: str = "") -> None:
    p = pal()
    color = p["good_text"] if good else (p["critical"] if good is False else p["ink2"])
    d = f'<div class="waam-hero-delta" style="color:{color}">{delta}</div>' if delta else ""
    st.markdown(f'<div class="waam-hero-label">{label}</div><div class="waam-hero">{value}</div>{d}'
                f'<div class="waam-hero-label">{note}</div>', unsafe_allow_html=True)


def badge(status: str) -> str:
    return "✅ 통과" if status == "PASS" else ("⛔ 실패" if status == "FAIL" else f"· {status}")


def style(fig: go.Figure, height: int = 360, legend: bool = True, title: str | None = None) -> go.Figure:
    """얇은 선·막대, 실선 머리카락 격자, 글자는 글자색(계열 색 아님), 범례는 계열이 둘 이상일 때만."""
    p = pal()
    fig.update_layout(
        template="none", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=p["ink2"]),
        title=dict(text=title, font=dict(size=14, color=p["ink"]), x=0, xanchor="left", y=0.98) if title else None,
        margin=dict(l=8, r=16, t=48 if title else 16, b=8),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                                       font=dict(color=p["ink2"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=p["surface"], bordercolor=p["axis"], font=dict(family=FONT, color=p["ink"])),
        bargap=0.45, bargroupgap=0.12,
    )
    ax = dict(showgrid=True, gridcolor=p["grid"], gridwidth=1, zeroline=False, linecolor=p["axis"], linewidth=1,
              ticks="", tickfont=dict(color=p["muted"]), title_font=dict(color=p["ink2"]), automargin=True)
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))
    fig.update_traces(line=dict(width=2), selector=dict(type="scatter"))
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
