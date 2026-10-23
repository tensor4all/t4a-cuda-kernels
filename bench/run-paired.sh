#!/usr/bin/env bash
# Paired A100 benchmark driver for the frozen performance experiment.
#
# Runs both sides under `nsys profile --capture-range=cudaProfilerApi` (the
# harnesses delimit the measured region with cuProfilerStart/cuProfilerStop and
# keep the warm-ups outside it), one process per case per side per repetition,
# and summarises the paired table with bench/summarize.py.
#
# Evidence rules (bench/summarize.py enforces the same ones on the artifacts):
#   * A campaign starts in a **fresh** directory. `--out` is required and must not
#     exist or must be empty; nothing is ever reused from an earlier run.
#   * The case list is the declared one (`--cases`, else the frozen list below),
#     never the set of stdout files that happen to exist.
#   * Every declared case x repetition x side is captured. A failed capture is
#     recorded, not swallowed, and the campaign then reports itself INCOMPLETE and
#     exits non-zero *without* a paired table: fewer repetitions cannot produce a
#     valid verdict.
#
# Sides:
#   candidate  examples/bench_kernels.rs of this repository (release build with
#              --features gpu-tests)
#   baseline   the throwaway tenferro-rs worktree at 14bc1ab with
#              bench/cubecl-baseline/baseline.patch applied (see its README)
#
# Raw nsys exports stay outside the repository under --out.
#
# Usage:
#   bench/run-paired.sh --out DIR [--reps N] [--cases "a,b,..."] \
#                       [--candidate-bin PATH] [--baseline-gpu-bin PATH] \
#                       [--baseline-linalg-bin PATH] [--allow-busy]
#
# The GPU is shared: every process starts only after
# `nvidia-smi --query-compute-apps=pid --format=csv,noheader` reports no other
# compute process (unless --allow-busy is given).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out=""
reps=3
cases_csv=""
gpu_bin=""
linalg_bin=""
candidate_bin=""
allow_busy=0
baseline_target="/home/shinaoka/tensor4all/shinaoka-advisor/work/runs/t4a-cuda-kernels-complete/baseline-target"
candidate_target="/home/shinaoka/tensor4all/shinaoka-advisor/work/runs/t4a-cuda-kernels-perf/candidate-target"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --out) out="$2"; shift 2 ;;
    --reps) reps="$2"; shift 2 ;;
    --cases) cases_csv="$2"; shift 2 ;;
    --candidate-bin) candidate_bin="$2"; shift 2 ;;
    --baseline-gpu-bin) gpu_bin="$2"; shift 2 ;;
    --baseline-linalg-bin) linalg_bin="$2"; shift 2 ;;
    --baseline-target) baseline_target="$2"; shift 2 ;;
    --candidate-target) candidate_target="$2"; shift 2 ;;
    --allow-busy) allow_busy=1; shift ;;
    -h | --help)
        sed -n '2,38p' "${BASH_SOURCE[0]}"
        exit 0
        ;;
    *)
        echo "unknown argument: $1" >&2
        exit 2
        ;;
    esac
done

if [[ -z "$out" ]]; then
    echo "--out DIR is required: a campaign must name its own fresh directory" >&2
    exit 2
fi
shopt -s nullglob dotglob
existing=("$out"/*)
shopt -u nullglob dotglob
if ((${#existing[@]})); then
    echo "refusing to reuse $out: it already holds ${#existing[@]} entr(ies)" >&2
    echo "a campaign starts in a fresh directory; nothing is reused across runs" >&2
    exit 1
fi
mkdir -p "$out"

if [[ -z "$cases_csv" ]]; then
    # The frozen case list of `docs/worklogs/2026-10-21-complete-port-perf.md`
    # §4.1. Both harnesses implement exactly these names.
    cases_csv="elementwise-add-f32-1000,elementwise-add-f32-1000000,\
elementwise-add-f32-100000000,elementwise-exp-f32-10000000,\
elementwise-mul-c64-10000000,elementwise-div-c64-10000000,\
fusion-chain-f32-1000,fusion-chain-f32-10000000,\
fusion-add-mul-c64-10000000,fusion-two-live-outs-f32-10000000,\
prepare-fused-cold-compile,\
layout-copy-f32-4096sq,layout-transpose-f32-4096sq,\
layout-transpose-f32-4093x4099,layout-transpose-f32-4099x4093,\
layout-transpose-offset-f32-4096sq,layout-transpose-batched-f32-64x256x256,\
layout-transpose-32x8-p1-v1-f32-4096sq,layout-transpose-16x8-p1-v2-f32-4096sq,\
layout-materialize-negative-stride-f32-4096sq,\
layout-slice-f32-4096sq-set1,layout-slice-f32-4096sq-set2,\
layout-slice-f32-4096sq-set3,layout-slice-f32-4096sq-set4,\
layout-pad-f32-4096sq-set1,layout-pad-f32-4096sq-set2,\
layout-pad-f32-4096sq-set3,layout-pad-f32-4096sq-set4,\
reduce-sum-f32-axis-511,reduce-sum-f32-axis-512,reduce-sum-f32-axis-513,\
reduce-sum-f32-axis-600,reduce-sum-f32-4096sq-axis0,reduce-sum-f32-4096sq-axis1,\
reduce-sum-f64-4096sq-axis0,reduce-sum-c64-4096sq-axis0,\
reduce-max-i32-4096sq-axis0,reduce-sum-f32-full-1000000,\
reduce-sum-f32-full-10000000,\
indexing-gather-f32-1024x256-table1000,indexing-gather-f32-1024x256-table100000,\
indexing-gather-f32-1024x256-table1000000,\
indexing-scatter-add-f32-1e6-into-16,indexing-scatter-add-c64-1e6-into-16,\
linalg-rrqr-256x256-f64,linalg-rrqr-256x256-c64,linalg-rrqr-1024x256-f64,\
linalg-rrqr-1024x256-c64,linalg-lu-apply-pivots-1024-f64,\
linalg-qr-apply-phase-1024x256-f64,linalg-qr-apply-phase-1024x256-c64,\
linalg-householder-explicit-v-1024x256-f64,\
linalg-householder-explicit-v-1024x256-c64"
fi

if [[ -z "$gpu_bin" ]]; then
    gpu_bin="$(ls -t "$baseline_target"/release/deps/tenferro_gpu-* 2>/dev/null | grep -v '\.d$' | head -1 || true)"
fi
if [[ -z "$linalg_bin" ]]; then
    linalg_bin="$(ls -t "$baseline_target"/release/deps/tenferro_linalg-* 2>/dev/null | grep -v '\.d$' | head -1 || true)"
fi
if [[ -z "$candidate_bin" ]]; then
    candidate_bin="$candidate_target/release/examples/bench_kernels"
fi

for binary in "$gpu_bin" "$linalg_bin" "$candidate_bin"; do
    if [[ ! -x "$binary" ]]; then
        echo "missing benchmark binary: $binary" >&2
        echo "build it first (see bench/cubecl-baseline/README.md and the worklog)" >&2
        exit 1
    fi
done

IFS=',' read -r -a cases <<<"$cases_csv"
if ((${#cases[@]} == 0)); then
    echo "the declared case list is empty" >&2
    exit 2
fi
for case in "${cases[@]}"; do
    if [[ -z "$case" ]]; then
        echo "the declared case list contains an empty entry" >&2
        exit 2
    fi
done

gpu_model="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
{
    echo "out=$out"
    echo "reps=$reps"
    echo "gpu=$gpu_model"
    echo "driver=$driver"
    echo "candidate_bin=$candidate_bin"
    echo "baseline_gpu_bin=$gpu_bin"
    echo "baseline_linalg_bin=$linalg_bin"
    echo "cases=$cases_csv"
    echo "date=$(date -Is)"
} >"$out/environment.txt"

wait_for_idle() {
    if [[ "$allow_busy" == "1" ]]; then
        return
    fi
    local waited=0
    while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
        if ((waited % 60 == 0)); then
            echo "waiting for an idle GPU (${waited}s): $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')" >&2
        fi
        sleep 5
        waited=$((waited + 5))
        if ((waited > 7200)); then
            echo "GPU still busy after ${waited}s" >&2
            return 1
        fi
    done
}

# One capture, no reuse and no swallowed failure: a missing artifact is recorded
# in `failed` and makes the campaign incomplete.
failed=()
run_case() {
    local side="$1" case="$2" rep="$3" binary="$4"
    shift 4
    local dir="$out/$side"
    mkdir -p "$dir"
    local report="$dir/$case.rep$rep"
    local stdout="$dir/$case.rep$rep.stdout"
    # The baseline harness selects its case through this variable; the candidate
    # takes it as a positional argument.
    export T4A_BENCH_CASES="$case"
    if ! wait_for_idle; then
        failed+=("$side/$case rep$rep: no idle GPU")
        return
    fi
    # `--capture-range=cudaProfilerApi` delimits the measured region; the harness
    # starts the range after its warm-ups and ends it after 100 iterations.
    if ! timeout 1800 nsys profile \
        --force-overwrite true \
        --trace=cuda \
        --sample=none \
        --cpuctxsw=none \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --output "$report" \
        "$binary" "$@" >"$stdout" 2>"$dir/$case.rep$rep.nsys.log"; then
        echo "FAILED: $side/$case rep$rep (see $dir/$case.rep$rep.nsys.log)" >&2
        failed+=("$side/$case rep$rep: nsys profile")
        return
    fi
    # One file per report so the two nine-column tables cannot be confused.
    if ! timeout 900 nsys stats --report cuda_gpu_kern_sum --format csv \
        "$report.nsys-rep" >"$dir/$case.rep$rep.kern.csv" \
        2>"$dir/$case.rep$rep.kern.log"; then
        echo "FAILED: nsys stats (kernel sum) for $side/$case rep$rep" >&2
        failed+=("$side/$case rep$rep: cuda_gpu_kern_sum")
        return
    fi
    # The second report reads the SQLite export the first call created; passing
    # the .nsys-rep again would trip over the existing (older) export.
    if ! timeout 900 nsys stats --report cuda_api_sum --format csv \
        "$report.sqlite" >"$dir/$case.rep$rep.api.csv" \
        2>"$dir/$case.rep$rep.api.log"; then
        echo "FAILED: nsys stats (api sum) for $side/$case rep$rep" >&2
        failed+=("$side/$case rep$rep: cuda_api_sum")
        return
    fi
    echo "captured $side/$case rep$rep"
}

for rep in $(seq 1 "$reps"); do
    for case in "${cases[@]}"; do
        if [[ "$case" == linalg-* ]]; then
            run_case baseline "$case" "$rep" "$linalg_bin" \
                --exact gpu::linalg::rank_revealing_qr::bench_harness::bench_tenferro_linalg \
                --ignored --nocapture
        else
            run_case baseline "$case" "$rep" "$gpu_bin" \
                --exact cubecl::bench_harness::bench_tenferro_gpu --ignored --nocapture
        fi
        run_case candidate "$case" "$rep" "$candidate_bin" "$case"
    done
done

# Belt and braces: every declared artifact must exist before the summariser is
# allowed to derive a verdict.
missing=()
for rep in $(seq 1 "$reps"); do
    for case in "${cases[@]}"; do
        for side in baseline candidate; do
            base="$out/$side/$case.rep$rep"
            for suffix in .stdout .kern.csv .api.csv; do
                [[ -s "$base$suffix" ]] || missing+=("$side/$case.rep$rep$suffix")
            done
        done
    done
done

if ((${#failed[@]} || ${#missing[@]})); then
    echo "" >&2
    echo "CAMPAIGN INCOMPLETE: ${#failed[@]} failed capture(s), ${#missing[@]} missing artifact(s)" >&2
    for item in "${failed[@]}"; do
        echo "  failed: $item" >&2
    done
    for item in "${missing[@]}"; do
        echo "  missing: $item" >&2
    done
    echo "No paired table is produced: a verdict needs every declared case x repetition x side." >&2
    exit 1
fi

python3 "$repo_root/bench/summarize.py" --root "$out" --reps "$reps" --cases "$cases_csv" \
    | tee "$out/paired.md"
echo "paired table: $out/paired.md"
