"""대시보드가 읽는 자료: runs/(tools/collect_evals.py 가 모은 평가·계산 시간 기록), results/(공식 측정 CSV), data/(형상)."""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SET_NAMES = {"tests": "공개 6", "check1010_v102": "연습 1", "novel1027": "연습 2"}
SET_NOTE = {"tests": "교수님이 공개한 형상 6개(채점 대상과 같은 종류)",
            "check1010_v102": "우리 생성기의 21종류를 하나씩(21개), 설정을 고를 때 결과를 본 연습용",
            "novel1027": "21종류와 다른 방식으로 만든 형상 21개(처음 보는 종류)"}
# 대시보드에서 숨기는 묶음과 그림(2026-11-25): fresh1027 은 연습 1 과 같은 생성기에 시드만 바꾼 묶음이라 이름만 늘려
# 헷갈리게 해서 뺀다. 자료(runs/evals, data/)는 그대로 둔다.
HIDDEN_SETS = {"fresh1027"}
HIDDEN_IMAGES = {"fresh1027.png", "seed_compare.png"}
# 기록 이름·설명에 남은 옛 묶음 이름 → 화면 이름(자료 파일은 그대로 두고 보일 때만 바꾼다)
OLD_SET_WORDS = [("새 방식 21개", "연습 2(21개)"), ("새방식", "연습2"), ("새 방식", "연습 2"), ("연습 21개", "연습 1(21개)")]


def rename_sets(text: str) -> str:
    for a, b in OLD_SET_WORDS:
        text = text.replace(a, b)
    return text


# 실험 화면 이름(2026-11-25): 기록 폴더 이름(날짜 + 그날 순서 + 줄임말)은 그대로 두고, 한 순서대로 번호를 붙여
# '실험 N · 무엇을 바꿨나'로 보인다. 여기 없는 새 실험은 폴더 이름을 그대로 쓴다.
EXP_TITLES = {"1027a_80mm": "80 mm 조각 후보 더함", "1027b_fix": "점수 계산을 채점기와 맞춤",
              "1027c_adapt": "벽 두께 맞춤 경로 더함", "1028a_ls4": "국소 탐색 층마다 4번",
              "1028b_ls2": "국소 탐색 층마다 2번", "1110a_ls2_adapt뺌": "벽 두께 맞춤 경로 뺌"}
# 계산 시간 측정 기록 화면 이름(2026-11-25): 로그 파일 이름은 그대로 두고, 잰 순서(날짜, 같은 날은 모은 순서)대로
# 번호를 붙여 '측정 N · 무엇을 쟀나 (묶음)'으로 보인다. 여기 없는 새 기록은 파일 이름을 쓴다.
TIMING_TITLES = {"1027_공개_지금_A_B": "80 mm 조각 후보 비교", "1027_공개_B_고치기전후": "점수 계산 고치기 전후",
                 "1027_새방식_지금_B": "80 mm 조각 후보 비교", "1028_공개_국소탐색_횟수": "국소 탐색 횟수 비교",
                 "1124_공개_코랩": "노트북과 코랩 비교"}
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
    df = df[~df["set"].isin(HIDDEN_SETS)].copy()
    df["set_label"] = df["set_dir"].map(set_label)
    df["desc"] = df["desc"].fillna("").map(rename_sets)
    df["exp"] = df["name"].map(lambda s: s.split("/")[0])
    num = {x: i + 1 for i, x in enumerate(sorted(df["exp"].unique()))}   # 폴더 이름이 날짜로 시작해 이름순 = 한 순서
    df["exp_label"] = df["exp"].map(lambda x: f"실험 {num[x]} · {EXP_TITLES.get(x, x)}")
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
        df = df[~df["set_dir"].map(lambda s: Path(s).name).isin(HIDDEN_SETS)]
        df = df.sort_values("date", kind="stable").reset_index(drop=True)   # 같은 날은 index.json 에 모은 순서
        df["num"] = df.index + 1
        df["set_label"] = df["set_dir"].map(set_label)
        # 화면에 보일 기록 이름(name 은 로그 파일 이름이라 그대로)
        df["label"] = [f"측정 {n} · {TIMING_TITLES.get(x, rename_sets(x))} ({s})"
                       for n, x, s in zip(df["num"], df["name"], df["set_label"])]
        df["desc"] = df["desc"].fillna("").map(rename_sets)
    return df


# 측정 로그에 적힌 방법 이름 → 화면에 보일 이름
WAY_NAMES = {"now": "지금", "A": "A (80 mm 더함)", "B": "B (160→80 mm)", "old": "고치기 전", "new": "고친 뒤",
             "before": "adapt 넣기 전", "adapt": "adapt 넣은 뒤",
             "끔": "국소 탐색 끔", "ls1": "층마다 1번", "ls2": "층마다 2번", "ls4": "층마다 4번",
             "노트북": "노트북 코어 1개", "코랩": "코랩 무료 CPU"}
COLAB_LIMIT_S = 300.0   # 최종 테스트: 코랩에서 형상 하나당 5분
# tools/ 그림 파일 → 화면에 보일 제목
IMAGE_TITLES = {"novel_fail.png": "연습 2 형상에서 떨어진 3개 — 어디가 비었나",
                "novel1027.png": "연습 2 형상 21개 (novel1027, 21종류와 다른 방식)",
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
    return sorted((p for p in (ROOT / "tools").glob("*.png") if p.name not in HIDDEN_IMAGES),
                  key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------- 움직임 그림(궤적을 층마다 나누기)
PARK_R = 900.0      # 주차점: 베이스 방향으로 작업대 중심에서 이만큼(src/schedule_v1.PARK_R)
WAIT_MIN = 1.0      # 이보다 짧은 기다림은 그림에 표시하지 않는다(s)


def traj_shapes(name: str) -> list[str]:
    """궤적을 같이 모은(collect_evals --traj) 형상들."""
    base = RUNS / "evals" / name
    return sorted(p.name for p in base.iterdir() if (p / "trajectory.csv.gz").exists()) if base.exists() else []


@st.cache_data(ttl=600, max_entries=6, show_spinner="궤적을 층마다 나누는 중…")
def motion(name: str, shape: str, set_dir: str) -> dict | None:
    """궤적의 구간(행 i → i+1)을 로봇·층·종류로 나눈다. 종류: D 쌓기, T 이동하기, W 기다리기, E 층 끝 기다리기.
    층은 쌓는 높이(z)로 정하고, 쌓지 않는 구간은 그 로봇이 다음에 쌓는 층에 넣는다. 다만 한 층의 마지막 쌓기 바로 뒤에
    이어지는 기다림은 다른 로봇이 그 층을 끝내기를 기다리는 시간이라 앞 층의 '층 끝 기다림'(E)으로 둔다.
    로봇이 그 층에서 쌓지 않으면(2대 설정의 세 번째 로봇 등) 시간으로 층을 정한다."""
    import numpy as np
    import yaml
    d = RUNS / "evals" / name / shape
    p = d / "trajectory.csv.gz"
    if not p.exists():
        return None
    cfg_p = d / "config.yaml" if (d / "config.yaml").exists() else ROOT / set_dir / shape / "config.yaml"
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
    df = pd.read_csv(p)
    zs = np.round(df.loc[df["mode"] == "D", "z_mm"].to_numpy(), 3)
    levels = np.unique(zs)
    rows = []
    for rid, g in df.groupby("robot_id", sort=True):
        t, x, y, z = (g[c].to_numpy(float) for c in ("time_s", "x_mm", "y_mm", "z_mm"))
        m = g["mode"].to_numpy(str)
        n = len(t) - 1
        if n <= 0:
            continue
        sm = m[:-1].copy()
        isd = sm == "D"
        lay = np.full(n, -1)
        lay[isd] = np.searchsorted(levels, np.round(z[:-1][isd], 3))
        di = np.nonzero(isd)[0]
        if len(di):
            pos = np.searchsorted(di, np.arange(n))
            nxt, prv = di[np.minimum(pos, len(di) - 1)], di[np.maximum(pos - 1, 0)]
            lay[~isd] = lay[nxt][~isd]
            run = False                                   # 마지막 쌓기 바로 뒤의 기다림 줄
            for i in range(n):
                if isd[i]:
                    run = True
                    continue
                if run and sm[i] == "W" and pos[i] > 0 and lay[nxt[i]] != lay[prv[i]]:
                    sm[i], lay[i] = "E", lay[prv[i]]
                else:
                    run = False
        rows.append(pd.DataFrame({"robot": rid, "layer": lay, "kind": sm, "t0": t[:-1], "t1": t[1:],
                                  "x0": x[:-1], "y0": y[:-1], "x1": x[1:], "y1": y[1:]}))
    seg = pd.concat(rows, ignore_index=True)
    seg = seg[seg["t1"] > seg["t0"]]
    # 쌓지 않는 로봇의 구간: 층마다 쌓는 로봇들이 그 층을 시작한 시각으로 나눈다
    starts = seg[seg["layer"] >= 0].groupby("layer")["t0"].min().sort_index()
    miss = seg["layer"] < 0
    if miss.any() and len(starts):
        seg.loc[miss, "layer"] = np.maximum(np.searchsorted(starts.to_numpy(), seg.loc[miss, "t0"].to_numpy(),
                                                            side="right") - 1, 0)
    robots = []
    for r in sorted(cfg["robots"], key=lambda r: r["id"]):
        b = np.asarray(r["base_xyz_mm"][:2], float)
        robots.append({"id": r["id"], "base": b.tolist(), "park": (b / np.linalg.norm(b) * PARK_R).tolist(),
                       "reach": float(r["xy_reach_radius_mm"])})
    span = seg.groupby("layer").agg(t0=("t0", "min"), t1=("t1", "max"))
    return {"seg": seg, "robots": robots, "n": len(levels), "span": span,
            "work_r": float(cfg.get("workspace", {}).get("radius_mm", 700.0))}


@st.cache_data(ttl=600, max_entries=4, show_spinner="3D 재생 자료를 만드는 중…")
def timeline3d(name: str, shape: str, set_dir: str, tol: float = 5.0) -> dict | None:
    """3D 시간 재생용 자료. 로봇마다 1초 간격의 토치 위치와 그 순간 동작(0 쌓기, 1 이동, 2 기다리기, 3 끝남),
    쌓은 선의 꼭짓점(tol mm 간격으로 줄임, 끊긴 곳은 None, 시각은 앞 꼭짓점과 같게 두어 늘 커지게),
    층마다 그 로봇의 첫 꼭짓점 번호, 층이 시작하는 시각(그 높이에서 처음 쌓은 시각)."""
    import numpy as np
    import yaml
    d = RUNS / "evals" / name / shape
    p = d / "trajectory.csv.gz"
    if not p.exists():
        return None
    cfg_p = d / "config.yaml" if (d / "config.yaml").exists() else ROOT / set_dir / shape / "config.yaml"
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
    df = pd.read_csv(p)
    T = float(df["time_s"].max())
    ts = np.arange(0.0, np.ceil(T) + 1.0, 1.0)
    dep = df[df["mode"] == "D"]
    levels = np.unique(np.round(dep["z_mm"].to_numpy(), 3))
    zr = np.round(dep["z_mm"].to_numpy(), 3)
    starts = [float(dep["time_s"].to_numpy()[zr == z0].min()) for z0 in levels]
    code = {"D": 0, "T": 1, "W": 2}
    out, zmax = [], float(df["z_mm"].max())
    for r in sorted(cfg["robots"], key=lambda r: r["id"]):
        g = df[df["robot_id"] == r["id"]]
        t, x, y, z = (g[c].to_numpy(float) for c in ("time_s", "x_mm", "y_mm", "z_mm"))
        m = g["mode"].to_numpy(str)
        k = np.searchsorted(t, ts, side="right") - 1
        mc = np.array([code.get(v, 2) for v in m])
        mode = np.where(k >= len(t) - 1, 3, mc[np.clip(k, 0, len(t) - 1)])
        mode = np.where(k < 0, 2, mode)
        pos = [np.round(np.interp(ts, t, a), 1).tolist() for a in (x, y, z)]
        vt, vx, vy, vz, vzz = [], [], [], [], []
        di = np.nonzero(m[:-1] == "D")[0]
        if len(di):
            for run in np.split(di, np.nonzero(np.diff(di) != 1)[0] + 1):
                ii = np.r_[run, run[-1] + 1]
                keep, lx, ly = [], None, None
                for j, q in enumerate(ii):
                    if lx is None or j == len(ii) - 1 or np.hypot(x[q] - lx, y[q] - ly) >= tol:
                        keep.append(q)
                        lx, ly = x[q], y[q]
                vt += [round(float(t[q]), 2) for q in keep] + [round(float(t[keep[-1]]), 2)]
                vx += [round(float(x[q]), 1) for q in keep] + [None]
                vy += [round(float(y[q]), 1) for q in keep] + [None]
                vz += [round(float(z[q]), 2) for q in keep] + [None]
                vzz += [float(z[q]) for q in keep] + [float(z[keep[-1]])]
        vzz = np.round(np.asarray(vzz, float), 3)
        first = np.searchsorted(vzz, levels, side="left").tolist() if len(vzz) else [0] * len(levels)
        out.append({"id": r["id"], "base": [float(v) for v in r["base_xyz_mm"]], "x": pos[0], "y": pos[1], "z": pos[2],
                    "m": mode.tolist(), "vt": vt, "vx": vx, "vy": vy, "vz": vz, "first": first})
    return {"T": T, "robots": out, "levels": levels.tolist(), "starts": starts, "zmax": zmax,
            "work_r": float(cfg.get("workspace", {}).get("radius_mm", 700.0))}


def _thin(xs, ys, hs, tol=3.0):
    """이어진 선에서 앞 점과 tol(mm) 안인 점은 빼고 0.1 mm·0.1 s 로 반올림한다(끝점·끊긴 곳은 남김). 재생 그림을 가볍게."""
    ox, oy, oh = [], [], []
    last = None
    for i, (x, y, h) in enumerate(zip(xs, ys, hs)):
        end = i + 1 == len(xs) or xs[i + 1] is None
        if x is None or last is None or end or (x - last[0]) ** 2 + (y - last[1]) ** 2 >= tol * tol:
            ox.append(None if x is None else round(x, 1)); oy.append(None if y is None else round(y, 1))
            oh.append(None if h is None else round(h, 1))
            last = None if x is None else (x, y)
    return ox, oy, oh


def layer_view(seg: pd.DataFrame, robots: list, t0: float, thin: bool = False) -> list[dict]:
    """한 층의 구간(seg)을 로봇마다 그림용으로: 쌓은 선 D, 옮긴 길 T, 기다린 곳 W, 층 끝 기다림 E, 비켜선 곳 K, 시간 합."""
    import numpy as np
    out = []
    for r in robots:
        gr = seg[seg["robot"] == r["id"]]
        dt = (gr["t1"] - gr["t0"]).to_numpy()
        kind = gr["kind"].to_numpy()
        v = {"id": r["id"], "sum": {k: float(dt[kind == k].sum()) for k in "DTWE"}}
        for k in "DT":
            xs, ys, hs, prev = [], [], [], None
            sel = gr[gr["kind"] == k]
            for x0, y0, x1, y1, a, b in sel[["x0", "y0", "x1", "y1", "t0", "t1"]].itertuples(index=False):
                if k == "T" and (x1 - x0) ** 2 + (y1 - y0) ** 2 < 0.25:      # 제자리에서 오르내리기만
                    continue
                if prev is None or k == "T" or abs(prev[0] - x0) > 1e-6 or abs(prev[1] - y0) > 1e-6:
                    if xs:
                        xs.append(None); ys.append(None); hs.append(None)
                    xs.append(x0); ys.append(y0); hs.append(a - t0)
                xs.append(x1); ys.append(y1); hs.append(b - t0)
                prev = (x1, y1)
            v[k] = _thin(xs, ys, hs, tol=3.0 if k == "D" else 0.0) if thin else (xs, ys, hs)   # 재생용은 0.1 mm 로 반올림
        busy = v["sum"]["D"] > 0
        w = gr[(gr["kind"] == "W") & ((gr["t1"] - gr["t0"]) >= WAIT_MIN)] if busy else gr.iloc[:0]
        v["W"] = (w["x0"].tolist(), w["y0"].tolist(), (w["t1"] - w["t0"]).tolist(), (w["t0"] - t0).tolist())
        e = gr[gr["kind"] == "E"] if busy else gr.iloc[:0]
        v["E"] = (e["x0"].tolist(), e["y0"].tolist(), (e["t1"] - e["t0"]).tolist())
        pk = gr[(gr["kind"] == "T") & (np.hypot(gr["x1"] - r["park"][0], gr["y1"] - r["park"][1]) < 1.0)]
        v["K"] = (pk["x1"].tolist(), pk["y1"].tolist(), (pk["t1"] - t0).tolist())
        v["n_wait"], v["n_park"] = len(v["W"][0]), len(v["K"][0])
        out.append(v)
    return out


@st.cache_data(ttl=600, max_entries=4, show_spinner="층마다 그림을 만드는 중…")
def layer_frames(name: str, shape: str, set_dir: str) -> list[list[dict]] | None:
    """재생용: 모든 층의 layer_view(쌓은 선은 3 mm 간격으로 줄임)."""
    M = motion(name, shape, set_dir)
    if M is None:
        return None
    seg, span = M["seg"], M["span"]
    return [layer_view(seg[seg["layer"] == L], M["robots"], float(span.loc[L, "t0"]), thin=True)
            if L in span.index else [] for L in range(M["n"])]


# ---------------------------------------------------------------- 층별 점검: 형상에서 보기(tools/shape_diff.py 가 만든 layer_shapes.zip)
VENV_PY = ROOT / "WAAM_Validator" / ".venv" / "Scripts" / "python.exe"


def shape_view_path(name: str, shape: str) -> Path:
    return RUNS / "evals" / name / shape / "layer_shapes.zip"


@st.cache_data(ttl=600, max_entries=8, show_spinner=False)
def shape_view(name: str, shape: str, mtime: float) -> dict | None:
    """층마다 점수·넓이·큰 조각 목록·3D 용 빈틈/넘침 삼각형·기준 미달 층 테두리(mtime 은 다시 만들면 캐시를 새로 읽게)."""
    import zipfile
    p = shape_view_path(name, shape)
    if not p.exists():
        return None
    with zipfile.ZipFile(p) as z:
        return json.loads(z.read("summary.json"))


@st.cache_data(ttl=600, max_entries=24, show_spinner=False)
def shape_view_layer(name: str, shape: str, layer: int, mtime: float) -> dict | None:
    """층 하나의 단면 그림: 목표 단면·쌓인 면·빈틈·넘침(고리는 [x0, y0, x1, y1, …])·불 켜고 지나간 길."""
    import zipfile
    with zipfile.ZipFile(shape_view_path(name, shape)) as z:
        try:
            return json.loads(z.read(f"L{layer:04d}.json"))
        except KeyError:
            return None


@st.cache_data(ttl=3600, max_entries=8, show_spinner="형상(STL)을 읽는 중…")
def target_mesh(set_dir: str, shape: str) -> dict | None:
    """목표 STL → 꼭짓점 x, y, z 와 삼각형 꼭짓점 번호 i, j, k (이진·글자 STL 모두)."""
    import numpy as np
    p = ROOT / set_dir / shape / "target.stl"
    if not p.exists():
        return None
    raw = p.read_bytes()
    n = int.from_bytes(raw[80:84], "little") if len(raw) >= 84 else -1
    if len(raw) == 84 + 50 * n:
        rec = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
        v = rec["v"].reshape(-1, 3).astype(float)
    else:
        v = np.array(re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", raw.decode("utf-8", "ignore")), float)
    u, inv = np.unique(np.round(v, 4), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    return {"x": u[:, 0], "y": u[:, 1], "z": u[:, 2], "i": f[:, 0], "j": f[:, 1], "k": f[:, 2]}


def make_shape_view(name: str, set_dir: str, shape: str) -> tuple[bool, str]:
    """tools/shape_diff.py 를 WAAM venv 파이썬으로 돌려 layer_shapes.zip 을 만든다(채점기 함수가 그 venv 에 있다)."""
    import subprocess
    cmd = [str(VENV_PY), "-X", "utf8", str(ROOT / "tools" / "shape_diff.py"), name, set_dir, shape, "--force"]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    except (OSError, subprocess.TimeoutExpired) as ex:
        return False, str(ex)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-800:]
