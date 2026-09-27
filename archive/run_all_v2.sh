#!/usr/bin/env bash
# usage: ./run_all_v2.sh 01 02 ...   (generate with schedule_v2.py, then validate)
V=~/waam_icpbl/WAAM_Validator
cd "$V"
for m in "$@"; do
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python ../v0/schedule_v2.py tests/$m ../jobs_v2/$m || { echo "GEN FAIL $m"; continue; }
  rm -rf ../runs_v2/$m
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/waam-validator.exe run ../jobs_v2/$m --output ../runs_v2/$m --json > ../runs_v2_$m.json 2> ../runs_v2_$m.err
  echo "$m exit=$?"
done
