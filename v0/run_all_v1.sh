#!/usr/bin/env bash
# usage: ./run_all_v1.sh 01 02 ...   (generate with schedule_v1.py, then validate)
V=~/waam_icpbl/WAAM_Validator
cd "$V"
for m in "$@"; do
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python ../v0/schedule_v1.py tests/$m ../jobs_v1/$m || { echo "GEN FAIL $m"; continue; }
  rm -rf ../runs_v1/$m
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/waam-validator.exe run ../jobs_v1/$m --output ../runs_v1/$m --json > ../runs_v1_$m.json 2> ../runs_v1_$m.err
  echo "$m exit=$?"
done
