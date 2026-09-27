#!/usr/bin/env bash
V=~/waam_icpbl/WAAM_Validator
cd "$V"
for m in "$@"; do
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python ../v0/schedule_v7.py ../bench/tests/$m ../bench/jobs_v7/$m || { echo "GEN FAIL $m"; continue; }
  rm -rf ../bench/runs_v7/$m
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python -m waam_validator.cli run ../bench/jobs_v7/$m --output ../bench/runs_v7/$m --json > ../bench/runs_v7_$m.json 2> ../bench/runs_v7_$m.err
  python - "$m" <<'PY'
import json,sys
m=sys.argv[1]
d=json.load(open(f"../bench/runs_v7_{m}.json",encoding="utf-8"))
print(f"  b{m} {d['status']} IoU {d['shape']['iou']:.3f} coll {d['collision']['collision_event_count']} {d['failure_reasons'][:2]}")
PY
done
