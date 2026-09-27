#!/usr/bin/env bash
# usage: ./run_all_v7.sh 01 02 ...   (schedule_v8.py 생성 + 검증)
V=~/waam_icpbl/WAAM_Validator
cd "$V"
for m in "$@"; do
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python ../v0/schedule_v8.py tests/$m ../jobs_v8/$m || { echo "GEN FAIL $m"; continue; }
  rm -rf ../runs_v8/$m
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python -m waam_validator.cli run ../jobs_v8/$m --output ../runs_v8/$m --json > ../runs_v8_$m.json 2> ../runs_v8_$m.err
  python - "$m" <<'PY'
import json,sys
m=sys.argv[1]
d=json.load(open(f"../runs_v8_{m}.json",encoding="utf-8")); c=d["collision"]
print(f"  {m} {d['status']} IoU {d['shape']['iou']:.3f} coll {d['collision']['collision_event_count']} {d['failure_reasons'][:2]}")
PY
done
