#!/bin/bash
set -uo pipefail
S=/tmp/claude-2000/-home-shinaoka-work-t4a-reduce-2406/7036dc7e-cb48-4619-846e-725e49469709/scratchpad
source ~/work/gpu-env.sh; export PATH=/usr/local/cuda-12.6/bin:$PATH
BASE=$HOME/tensor4all/t4a-cuda-kernels-pass1-base/target/release/examples/bench_kernels
CAND=$HOME/tensor4all/t4a-cuda-kernels-pass1/target/release/examples/bench_kernels
cd ~/tensor4all/t4a-cuda-kernels-pass1
python3 $S/idle.py "native before"
OUT=$S/native3 REPS=3 CPUS=0-7 BASE_BIN=$BASE CAND_BIN=$CAND CASES="reduce-sum-f32-full-1048576 reduce-max-f32-full-1048576 reduce-min-f32-full-1048576 reduce-sum-f64-full-1048576 reduce-max-f64-full-1048576 reduce-min-f64-full-1048576 reduce-sum-f32-full-16777216 reduce-max-f32-full-16777216 reduce-min-f32-full-16777216 reduce-sum-f64-full-16777216 reduce-max-f64-full-16777216 reduce-min-f64-full-16777216 reduce-prod-f32-full-1048576 reduce-prod-f64-full-1048576 reduce-sumsquares-f32-full-1048576 reduce-sum-f32-4096sq-axis1 reduce-max-f32-4x131072-axis1 reduce-sum-f32-axis-600 reduce-sum-f32-4096sq-axis0 reduce-sum-f64-4096sq-axis0 reduce-max-i32-4096sq-axis0 reduce-sum-c64-4096sq-axis0 reduce-sum-f32-full-1000000 reduce-sum-f32-full-10000000" bash bench/run-native-pair.sh > $S/native3.log 2>&1
python3 $S/idle.py "native after"
C="reduce-sum-f32-full-1048576 reduce-max-f32-full-1048576 reduce-min-f32-full-1048576 reduce-sum-f64-full-1048576 reduce-max-f64-full-1048576 reduce-min-f64-full-1048576 reduce-sum-f32-full-16777216 reduce-max-f32-full-16777216 reduce-min-f32-full-16777216 reduce-sum-f64-full-16777216 reduce-max-f64-full-16777216 reduce-min-f64-full-16777216"
$S/knsys.sh $BASE $S/nsys-b $C > /dev/null 2>&1
$S/knsys.sh $CAND $S/nsys-c $C > /dev/null 2>&1
$S/campaign-pass1.sh
echo ALLDONE
