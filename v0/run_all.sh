#!/usr/bin/env bash
V=~/waam_icpbl/WAAM_Validator
cd "$V"
for m in "$@"; do
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python ../v0/schedule_v0.py tests/$m ../jobs/$m || { echo "GEN FAIL $m"; continue; }
  rm -rf ../runs/$m
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/waam-validator.exe run ../jobs/$m --output ../runs/$m --json > ../runs_$m.json 2> ../runs_$m.err
  echo "$m exit=$?"
done
