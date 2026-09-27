#!/usr/bin/env bash
# usage: EXTRA="--weights ../v0/rl_policy_v2.npz" ./run_all_rl.sh public|bench 01 02 ...
V=~/waam_icpbl/WAAM_Validator
cd "$V"
SET=$1; shift
for m in "$@"; do
  if [ "$SET" = "bench" ]; then SRC=../bench/tests/$m; JOB=../bench/jobs_rl/$m; RUN=../bench/runs_rl/$m; J=../bench/runs_rl_$m.json
  else SRC=tests/$m; JOB=../jobs_rl/$m; RUN=../runs_rl/$m; J=../runs_rl_$m.json; fi
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python ../v0/schedule_rl.py $SRC $JOB $EXTRA || { echo "GEN FAIL $m"; continue; }
  rm -rf $RUN
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python -m waam_validator.cli run $JOB --output $RUN --json > $J 2>/dev/null
  python - "$J" "$m" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
print(f"  {sys.argv[2]} {d['status']} IoU {d['shape']['iou']:.3f} coll {d['collision']['collision_event_count']} {d['failure_reasons'][:2]}")
PY
done
