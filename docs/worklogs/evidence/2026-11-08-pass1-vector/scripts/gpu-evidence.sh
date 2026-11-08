#!/bin/bash
set -uo pipefail
S=/tmp/claude-2000/-home-shinaoka-work-t4a-reduce-2406/7036dc7e-cb48-4619-846e-725e49469709/scratchpad
source ~/work/gpu-env.sh; export PATH=/usr/local/cuda-12.6/bin:$PATH
K=$HOME/tensor4all/t4a-cuda-kernels-pass1; P=$HOME/tensor4all/t4a-cuda-kernels-shfl-proto
E=$K/docs/worklogs/evidence/2026-11-08-pass1-vector
cd $K; SHA=$(git rev-parse HEAD)
# (a) the GPU test binaries, with the feature that builds them
{ echo "# commit $SHA"; echo "# GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); nvcc $(nvcc --version | tail -1)"; echo "# \$ cargo test -j 4 --release --features gpu-tests --no-fail-fast"; cargo test -j 4 --release --features gpu-tests --no-fail-fast 2>&1; echo "# exit $?"; } > $E/a-cargo-test-gpu-tests.log
{ echo "# commit $SHA"; echo "# \$ cargo clippy -j 4 --keep-going --all-targets --features gpu-tests -- -D warnings"; cargo clippy -j 4 --keep-going --all-targets --features gpu-tests -- -D warnings 2>&1; echo "# exit $?"; } > $E/a-cargo-clippy-gpu-tests.log
# (c) mutants
mkdir -p $E/c-mutants; cp cuda/src/reduce.cu $S/reduce.cu.head; HASH=$(sha256sum cuda/src/reduce.cu | cut -d' ' -f1)
for m in drop-tail batch-overlap first-batch-short last-batch-short aligned-test-too-weak scalar-group-load-wrong prod-regrouped; do
  cp $S/reduce.cu.head cuda/src/reduce.cu; python3 $S/mutate2.py cuda/src/reduce.cu $m
  git diff -- cuda/src/reduce.cu > $E/c-mutants/$m.patch
  { echo "# commit $SHA + c-mutants/$m.patch"; echo "# \$ cargo test -j 4 --release --features gpu-tests --no-fail-fast --test gpu_reduce --test gpu_dynamic_rank -- vector_walk rank_nine_contiguous product_partials pass1_"; cargo test -j 4 --release --features gpu-tests --no-fail-fast --test gpu_reduce --test gpu_dynamic_rank -- vector_walk rank_nine_contiguous product_partials pass1_ 2>&1; echo "# exit $?"; } > $E/c-mutants/$m.log
done
cp $S/reduce.cu.head cuda/src/reduce.cu; [ "$(sha256sum cuda/src/reduce.cu | cut -d' ' -f1)" = "$HASH" ] && echo "reduce.cu restored" || echo "RESTORE FAILED"
git status --short cuda/
# (c) SASS and registers
{ echo "# commit $SHA"; echo "# \$ scripts/sass-diff.sh 31b734e"; bash scripts/sass-diff.sh 31b734e 2>&1; } > $E/c-sass-diff.log
{ echo "# commit $SHA"; echo "# \$ nvcc -std=c++17 -cubin -arch=sm_80 -I cuda/include -Xptxas -v cuda/src/reduce.cu"; nvcc -std=c++17 -cubin -arch=sm_80 -I cuda/include -Xptxas -v -o $S/head.cubin cuda/src/reduce.cu 2>&1; git show 31b734e:cuda/src/reduce.cu > $S/base.cu; echo "# --- baseline 31b734e"; nvcc -std=c++17 -cubin -arch=sm_80 -I cuda/include -Xptxas -v -o $S/base.cubin $S/base.cu 2>&1; } > $E/c-ptxas.log
# shuffle prototype: bit identity with the serial walk, then timing
cd $P
T4A_DUMP_PARTIALS=$S/dump-proto.txt cargo test -j 4 --release --features gpu-tests --test gpu_reduce -- zz_dump > $S/dump-proto.log 2>&1
cp cuda/src/reduce.cu $S/reduce.cu.proto; git -C $K show 31b734e:cuda/src/reduce.cu > cuda/src/reduce.cu
T4A_DUMP_PARTIALS=$S/dump-serial.txt cargo test -j 4 --release --features gpu-tests --test gpu_reduce -- zz_dump > $S/dump-serial.log 2>&1
cp $S/reduce.cu.proto cuda/src/reduce.cu
cmp $S/dump-proto.txt $S/dump-serial.txt && echo "PROTO BIT-IDENTICAL to serial walk" || echo "PROTO DIFFERS"
C="reduce-sum-f32-full-1048576 reduce-max-f32-full-1048576 reduce-sum-f64-full-1048576 reduce-max-f64-full-1048576 reduce-sum-f32-full-16777216 reduce-max-f32-full-16777216 reduce-min-f32-full-16777216 reduce-sum-f64-full-16777216 reduce-max-f64-full-16777216 reduce-min-f64-full-16777216"
rm -rf $S/nsys-p $S/nsys-v
$S/knsys.sh $P/target/release/examples/bench_kernels $S/nsys-p $C > /dev/null 2>&1
$S/knsys.sh $K/target/release/examples/bench_kernels $S/nsys-v $C > /dev/null 2>&1
grep -h check= $S/nsys-p/*.log | grep -o 'case=[^ ]*\|check=[a-z]*' | paste - - | sort | uniq -c | head -12
# native campaign repeat (the 65a74d5 one had a 7% IQR baseline cell)
cd $K; python3 $S/idle.py "native4 before"
OUT=$S/native4 REPS=3 CPUS=0-7 BASE_BIN=$HOME/tensor4all/t4a-cuda-kernels-pass1-base/target/release/examples/bench_kernels CAND_BIN=$K/target/release/examples/bench_kernels CASES="reduce-sum-f32-full-1048576 reduce-max-f32-full-1048576 reduce-min-f32-full-1048576 reduce-sum-f64-full-1048576 reduce-max-f64-full-1048576 reduce-min-f64-full-1048576 reduce-sum-f32-full-16777216 reduce-max-f32-full-16777216 reduce-min-f32-full-16777216 reduce-sum-f64-full-16777216 reduce-max-f64-full-16777216 reduce-min-f64-full-16777216 reduce-prod-f32-full-1048576 reduce-prod-f64-full-1048576 reduce-sumsquares-f32-full-1048576 reduce-sum-f32-4096sq-axis1 reduce-max-f32-4x131072-axis1 reduce-sum-f32-axis-600 reduce-sum-f32-4096sq-axis0 reduce-sum-f64-4096sq-axis0 reduce-max-i32-4096sq-axis0 reduce-sum-c64-4096sq-axis0 reduce-sum-f32-full-1000000 reduce-sum-f32-full-10000000" bash bench/run-native-pair.sh > $S/native4.log 2>&1
python3 $S/idle.py "native4 after"
echo ALLDONE
