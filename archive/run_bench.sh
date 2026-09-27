#!/usr/bin/env bash
# usage: ./run_bench.sh [ids...]  -- 참고 답안(ref)과 V2(ours)를 벤치마크 형상에 대해 채점
V=~/waam_icpbl/WAAM_Validator
B=~/waam_icpbl/bench
cd "$V"
ids=${@:-01 02 03 04 05 06 07 08 09 10}
for m in $ids; do
  rm -rf $B/runs_ref/$m $B/runs_ours/$m
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/waam-validator.exe run $B/tests/$m --output $B/runs_ref/$m --json > $B/ref_$m.json 2> $B/ref_$m.err
  r1=$?
  s=$(date +%s.%N)
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python ../v0/schedule_v2.py $B/tests/$m $B/jobs_ours/$m > $B/gen_$m.log 2>&1 || { echo "$m GEN FAIL"; tail -3 $B/gen_$m.log; continue; }
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/waam-validator.exe run $B/jobs_ours/$m --output $B/runs_ours/$m --json > $B/ours_$m.json 2> $B/ours_$m.err
  r2=$?
  echo "$m ref_exit=$r1 ours_exit=$r2 | $(cat $B/gen_$m.log | tail -1)"
done
