"""대시보드가 읽는 자료: runs/(tools/collect_evals.py 가 모은 평가·계산 시간 기록), results/(공식 측정 CSV), data/(형상)."""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SET_NAMES = {"tests": "공개 6", "check1010_v102": "연습 21", "fresh1027": "새 시드 21", "novel1027": "새 방식 21"}
SET_NOTE = {"tests": "교수님이 공개한 형상(채점 대상과 같은 종류)",
            "check1010_v102": "우리 생성기의 21종류, 설정을 고를 때 결과를 본 연습용",
            "fresh1027": "우리 생성기, 처음 보는 시드. 종류는 같고 크기·두께·층수만 다름",
            "novel1027": "21종류와 다른 방식으로 만든 형상(처음 보는 종류)"}
# 채점기 v1.0.2 기준
IOU_MIN, COVER_MIN, OVER_MAX = 0.98, 0.98, 0.03


def set_label(set_dir: str) -> str:
    name = Path(set_dir).name
    return SET_NAMES.get(name, name)


@st.cache_data(ttl=30)
def eval_index() -> pd.DataFrame:
    p = RUNS / "evals" / "index.json"
    rows = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["set"] = df["set_dir"].map(lambda s: Path(s).name)
    df["set_label"] = df["set_dir"].map(set_label)
    df["exp"] = df["name"].map(lambda s: s.split("/")[0])
    return df.sort_values(["exp", "set"])


def _solve(path: Path) -> dict:
    out = {"cands": {}, "best": None, "time": {}}
    for ln in path.read_text(encoding="utf-8").splitlines():
        f = ln.split("\t")
        if f[0] == "best":
            out["best"] = f[1]
        elif f[0] == "time":
            out["time"] = {k: float(v.rstrip("s")) for k, v in (x.split("=") for x in f[1].split())}
        elif f[0] not in ("features", "sample") and len(f) == 3:
            out["cands"][f[0]] = (float(f[1]) if f[1] else None, float(f[2]) if f[2] else None)
    return out


@st.cache_data(ttl=30)
def bounds(set_name: str) -> dict:
    p = RUNS / "evals" / "bounds" / f"{set_name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(ttl=30)
def shape_desc(set_dir: str) -> dict:
    d = ROOT / set_dir
    out = {}
    if d.exists():
        for job in sorted(p for p in d.iterdir() if p.is_dir()):
            f = job / "desc.txt"
            out[job.name] = f.read_text(encoding="utf-8").strip() if f.exists() else ""
    return out


@st.cache_data(ttl=30)
def run_table(name: str, set_dir: str) -> pd.DataFrame:
    """실험 하나의 형상별 결과."""
    base = RUNS / "evals" / name
    b = bounds(Path(set_dir).name)
    desc = shape_desc(set_dir)
    rows = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        s = _solve(d / "solve.txt")
        sm = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {}
        sh = sm.get("shape", {})
        ms = float(sm["schedule"]["makespan_s"]) if sm else (s["cands"].get(s["best"], (None, None))[1] or float("nan"))
        bb = b.get(d.name, {})
        rows.append({"형상": d.name, "설명": desc.get(d.name, ""), "층수": bb.get("layers"),
                     "고른 설정": s["best"], "makespan(s)": ms, "통과": sm.get("status", "?"),
                     "겹침 점수(%)": 100 * sh.get("iou", float("nan")), "덮은 비율(%)": 100 * sh.get("coverage", float("nan")),
                     "넘친 비율(%)": 100 * sh.get("overfill_ratio", float("nan")),
                     "실패 층": sh.get("failed_layer_count"), "충돌": sm.get("collision", {}).get("collision_event_count"),
                     "이론상 최소(s)": bb.get("bound_s"), "최소의 몇 배": ms / bb["bound_s"] if bb.get("bound_s") else None,
                     "풀이 시간(s)": s["time"].get("total"), "_cands": s["cands"]})
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def layer_metrics(name: str, shape: str) -> pd.DataFrame:
    p = RUNS / "evals" / name / shape / "layer_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=30)
def timing_index() -> pd.DataFrame:
    p = RUNS / "timing" / "index.json"
    rows = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    df = pd.DataFrame(rows)
    if not df.empty:
        df["set_label"] = df["set_dir"].map(set_label)
    return df


# 측정 로그에 적힌 방법 이름 → 화면에 보일 이름
WAY_NAMES = {"now": "지금", "A": "A (80 mm 더함)", "B": "B (160→80 mm)", "old": "고치기 전", "new": "고친 뒤",
             "before": "adapt 넣기 전", "adapt": "adapt 넣은 뒤",
             "끔": "국소 탐색 끔", "ls1": "층마다 1번", "ls2": "층마다 2번", "ls4": "층마다 4번"}
# tools/ 그림 파일 → 화면에 보일 제목
IMAGE_TITLES = {"novel_fail.png": "다른 방식 형상에서 떨어진 3개 — 어디가 비었나",
                "novel1027.png": "다른 방식으로 만든 확인용 형상 21개 (novel1027)",
                "seed_compare.png": "같은 종류끼리: 예전 묶음(check808) ↔ 새 시드 묶음(fresh1027)",
                "fresh1027.png": "새 시드 형상 21개 (fresh1027, 단면 선 그림)",
                "our_shapes.png": "우리 생성기 21종류 (check808)",
                "public_models.png": "공개 형상 6개"}


@st.cache_data(ttl=30)
def timing_table(name: str) -> pd.DataFrame:
    p = RUNS / "timing" / f"{name}.log"
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+) (\S+)\s+벽시계\s+([\d.]+) s\s+makespan ([\d.]+)\s+(\S+)", ln)
        if m:
            # 국소 탐색을 쓴 줄은 'after=… used=1' 이 실제 makespan 이다(makespan 칸은 국소 탐색 전 값)
            a = re.search(r"after=([\d.]+) used=1", ln)
            rows.append({"형상": m.group(1), "방법": WAY_NAMES.get(m.group(2), m.group(2)), "계산 시간(s)": float(m.group(3)),
                         "makespan(s)": float(a.group(1) if a else m.group(4)), "고른 설정": m.group(5)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def official_reports() -> list[Path]:
    return sorted((ROOT / "results").glob("report_public_*_1core.csv"), reverse=True)


def read_report(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, encoding="utf-8-sig", dtype={"model": str})
    return df.rename(columns={"model": "모델", "makespan_s": "makespan(s)", "gen_s": "계산 시간(s)", "status": "통과",
                              "collisions": "충돌", "iou": "겹침 점수"})


def images() -> list[Path]:
    return sorted((ROOT / "tools").glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
