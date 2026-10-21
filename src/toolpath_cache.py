# -*- coding: utf-8 -*-
"""층별 적층 경로(best_paths)를 여러 코어로 나눠 한 번만 계산해 파일로 저장한다.

포트폴리오의 세 설정(V7·V8·균형 V8)은 모두 같은 층별 경로를 쓰는데, 지금까지는 각자 따로
계산했다(04번 40초, 05번 112초). 여기서 한 번만, 층을 코어별로 나눠 계산하고
설정들은 --paths 파일로 받아 쓴다. 경로 계산 자체는 같으므로 결과 궤적도 같다.

usage: python toolpath_cache.py <job_dir> <cache.pkl> [--workers N]   (WAAM venv 파이썬)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


_LAYERS = None


def _slice(job: Path):
    from waam_validator.config.loader import load_config
    from waam_validator.shape.target import load_target_mesh, slice_target_layers
    cfg = load_config(job / "config.yaml")
    import toolpath
    toolpath.configure(cfg)          # 워커마다 config 의 비드 폭 등으로 경로 상수를 맞춘다
    mesh = load_target_mesh(job / "target.stl", cfg)
    pr = cfg.process
    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    return slice_target_layers(mesh, list(range(n)), cfg)


def _init(job, spiral=True, both=False):
    global _SPIRAL, _BOTH
    _SPIRAL, _BOTH = spiral, both
    # 단면 도형은 pickle 로 넘기면 정밀도 격자·좌표가 미세하게 바뀌어 경로가 달라진다.
    # 그래서 워커마다 STL 을 직접 잘라 스케줄러와 똑같은 도형을 쓴다.
    global _LAYERS
    _LAYERS = _slice(Path(job))


_SPIRAL = True


_BOTH = False


def _work(k):
    from toolpath import best_paths
    if _BOTH:                                   # 나선 포함·고리만 두 결과를 한 번에
        (p1, s1, _), (p2, s2, _) = best_paths(_LAYERS[k], spiral=True, both=True)
        return k, (p1, p2), (s1["iou"], s2["iou"])
    paths, sc, _ = best_paths(_LAYERS[k], spiral=_SPIRAL)
    return k, paths, sc["iou"]


REUSE_BLOCK = 25     # 재사용 사슬 길이(코어 수와 무관하게 고정 → 어떤 컴퓨터에서도 같은 결과)
REUSE_DIFF = 0.005   # 아래층과 단면 차이(대칭차 면적 ÷ 면적)가 이보다 작을 때만 재사용
_REUSE = False


def _ok(A, paths):
    from toolpath import score, meets
    sc = score(A, paths)
    return meets(sc), sc["iou"]


def _work_block(ks):
    """연속한 층 묶음을 차례로 처리한다. 아래층 경로를 그대로 써도 이 층의 형상 기준(IoU·넘침)을
    통과하면 재사용하고, 아니면 새로 고른다. 점수 재기(0.01~0.04s)가 경로 고르기(0.9~2s)보다 훨씬 싸다.
    (05 는 350층 전부, 04·505-02 는 97~98% 재사용 가능, 꽃병처럼 층마다 모양이 바뀌면 25%)"""
    # 처음엔 '형상 기준만 통과하면 재사용'했더니 통과는 하지만 느린 경로를 써서 404 가 +2.0%(벌집 +6.5%)
    # 나빠졌다. 그래서 아래층과 단면이 거의 똑같을 때(차이 0.5% 미만)만 재사용한다. 이때는 새로 골라도
    # 사실상 같은 경로가 나오므로 makespan 손해 없이 시간만 준다(05 같은 관·기둥형).
    out, prev, prevA = [], None, None
    for k in ks:
        A = _LAYERS[k]
        if prev is not None and prevA is not None:
            diff = A.symmetric_difference(prevA).area / max(A.area, 1e-9)
            if diff >= REUSE_DIFF:
                prev = None
        prevA = A
        if prev is not None:
            if _BOTH:
                (p1, p2) = prev
                ok1, i1 = _ok(A, p1)
                ok2, i2 = _ok(A, p2)
                if ok1 and ok2:
                    out.append((k, (p1, p2), (i1, i2)))
                    continue
                if ok1 or ok2:
                    # 한쪽만 통과하면 떨어진 쪽만 새로 고른다(고리형은 기준을 겨우 넘는 경로라 자주 떨어진다)
                    from toolpath import best_paths
                    if ok1:
                        q2, s2, _ = best_paths(A, spiral=False)
                        r = (k, (p1, q2), (i1, s2["iou"]))
                    else:
                        q1, s1, _ = best_paths(A, spiral=True)
                        r = (k, (q1, p2), (s1["iou"], i2))
                    out.append(r)
                    prev = r[1]
                    continue
            else:
                ok, i = _ok(A, prev)
                if ok:
                    out.append((k, prev, i))
                    continue
        r = _work(k)
        out.append(r)
        prev = r[1]
    return out


def build(job: Path, workers: int = 0, spiral: bool = True, both: bool = False, reuse: bool = False):
    global _LAYERS, _SPIRAL, _BOTH
    _SPIRAL, _BOTH = spiral, both
    _LAYERS = _slice(job)
    tasks = [k for k in range(len(_LAYERS)) if not _LAYERS[k].is_empty]
    nw = workers or mp.cpu_count()
    out = {}
    if reuse:
        blocks = [tasks[i:i + REUSE_BLOCK] for i in range(0, len(tasks), REUSE_BLOCK)]
        if nw <= 1 or len(blocks) < 2:
            for b in blocks:
                for k, p, iou in _work_block(b):
                    out[k] = (p, iou)
        else:
            with mp.Pool(min(nw, len(blocks)), initializer=_init, initargs=(str(job), spiral, both)) as pool:
                for res in pool.imap_unordered(_work_block, blocks):
                    for k, p, iou in res:
                        out[k] = (p, iou)
        return out
    if nw <= 1 or len(tasks) < 4:
        for t in tasks:
            k, p, iou = _work(t)
            out[k] = (p, iou)
    else:
        with mp.Pool(min(nw, len(tasks)), initializer=_init, initargs=(str(job), spiral, both)) as pool:
            for k, p, iou in pool.imap_unordered(_work, tasks, chunksize=max(1, len(tasks) // (nw * 4))):
                out[k] = (p, iou)
    return out


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("cache")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--both", default=None, metavar="RING_PKL",
                    help="나선 포함 결과는 cache 에, 고리만 결과는 이 파일에 함께 저장(한 번 계산)")
    ap.add_argument("--reuse", action="store_true",
                    help="아래층 경로가 이 층 형상 기준을 통과하면 재사용(경로 계산 시간 단축)")
    ap.add_argument("--no-spiral", dest="spiral", action="store_false",
                    help="고리를 이어 붙인 나선 경로를 후보에서 뺀다")
    a = ap.parse_args()
    t0 = time.time()
    data = build(Path(a.job), a.workers, a.spiral, both=bool(a.both), reuse=a.reuse)
    Path(a.cache).parent.mkdir(parents=True, exist_ok=True)
    if a.both:                                  # (나선, 고리) 쌍을 두 파일로 나눠 저장
        sp = {k: (p[0], iou[0]) for k, (p, iou) in data.items()}
        ring = {k: (p[1], iou[1]) for k, (p, iou) in data.items()}
        with open(a.both, "wb") as f:
            pickle.dump(ring, f, protocol=pickle.HIGHEST_PROTOCOL)
        data = sp
    with open(a.cache, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"경로 캐시 {len(data)}층 {time.time() - t0:.1f}s → {a.cache}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
