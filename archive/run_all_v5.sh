#!/usr/bin/env bash
# usage: ./run_all_v5.sh 01 02 ...   (schedule_v5.py 로 생성 후 검증)
V=~/waam_icpbl/WAAM_Validator
cd "$V"
for m in "$@"; do
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python ../v0/schedule_v5.py tests/$m ../jobs_v5/$m "${EXTRA[@]}" || { echo "GEN FAIL $m"; continue; }
  rm -rf ../runs_v5/$m
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python -m waam_validator.cli run ../jobs_v5/$m --output ../runs_v5/$m --json > ../runs_v5_$m.json 2> ../runs_v5_$m.err
  echo "$m exit=$?"
done
