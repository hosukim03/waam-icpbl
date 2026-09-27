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
    mesh = load_target_mesh(job / "target.stl", cfg)
    pr = cfg.process
    n = int(np.ceil((mesh.bounds[1, 2] - pr.build_plane_z_mm) / pr.layer_height_mm - 1e-9))
    return slice_target_layers(mesh, list(range(n)), cfg)


def _init(job, spiral=True):
    global _SPIRAL
    _SPIRAL = spiral
    # 단면 도형은 pickle 로 넘기면 정밀도 격자·좌표가 미세하게 바뀌어 경로가 달라진다.
    # 그래서 워커마다 STL 을 직접 잘라 스케줄러와 똑같은 도형을 쓴다.
    global _LAYERS
    _LAYERS = _slice(Path(job))


_SPIRAL = True


def _work(k):
    from toolpath import best_paths
    paths, sc, _ = best_paths(_LAYERS[k], spiral=_SPIRAL)
    return k, paths, sc["iou"]


def build(job: Path, workers: int = 0, spiral: bool = True):
    global _LAYERS, _SPIRAL
    _SPIRAL = spiral
    _LAYERS = _slice(job)
    tasks = [k for k in range(len(_LAYERS)) if not _LAYERS[k].is_empty]
    nw = workers or mp.cpu_count()
    out = {}
    if nw <= 1 or len(tasks) < 4:
        for t in tasks:
            k, p, iou = _work(t)
            out[k] = (p, iou)
    else:
        with mp.Pool(min(nw, len(tasks)), initializer=_init, initargs=(str(job), spiral)) as pool:
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
    ap.add_argument("--no-spiral", dest="spiral", action="store_false",
                    help="고리를 이어 붙인 나선 경로를 후보에서 뺀다")
    a = ap.parse_args()
    t0 = time.time()
    data = build(Path(a.job), a.workers, a.spiral)
    Path(a.cache).parent.mkdir(parents=True, exist_ok=True)
    with open(a.cache, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"경로 캐시 {len(data)}층 {time.time() - t0:.1f}s → {a.cache}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
