# -*- coding: utf-8 -*-
"""CEM 병렬 학습용 일꾼 프로세스.
프로세스마다 형상별 층 경로를 캐시해 두고, 가중치만 바꿔 가며 스케줄링을 반복한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from schedule_rl import NFEAT, Policy, parse, run

CACHE, JOBS, OPTS, HID = {}, {}, None, 8


def init(train_dir, layers, hidden, max_steps=0):
    global JOBS, OPTS, HID
    OPTS = parse(["x", "y", "--max-layers", str(layers), "--max-steps", str(max_steps)])
    JOBS = {p.name: p for p in sorted(Path(train_dir).iterdir()) if (p / "target.stl").exists()}
    HID = hidden


def unpack(vec, hidden):
    p = Policy(hidden=hidden)
    n1 = hidden * NFEAT
    p.W1 = vec[:n1].reshape(hidden, NFEAT)
    p.b1 = vec[n1:n1 + hidden]
    p.w2 = vec[n1 + hidden:n1 + 2 * hidden]
    p.b2 = float(vec[-1])
    return p


def work(args):
    """(후보번호, 형상이름, 가중치) -> (후보번호, 형상이름, makespan). 가중치 None 이면 그리디."""
    idx, name, vec = args
    pol = None if vec is None else unpack(np.asarray(vec, float), HID)
    _, mk, _, _ = run(JOBS[name], OPTS, policy=pol, cache=CACHE.setdefault(name, {}))
    return idx, name, mk
