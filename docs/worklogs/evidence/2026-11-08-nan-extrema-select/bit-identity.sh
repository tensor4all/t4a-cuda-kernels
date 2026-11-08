#!/usr/bin/env bash
# Builds bit-identity.cu and runs it in every mode that has to be distinguishable, recording the
# exit code of each. The point: "a failed run exits nonzero" is only evidence when the log shows
# the code, not when the harness is trusted to return it.
#
#   ./bit-identity.sh > bit-identity.log 2>&1
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin="$here/bit-identity"
nvcc="${NVCC:-nvcc}"

if gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1) && [ -n "$gpu" ]; then
    echo "# $gpu"
else
    echo "# no NVIDIA device: nvidia-smi gives no answer"
fi
echo "# $nvcc --version | tail -1: $("$nvcc" --version | tail -1)"
"$nvcc" -std=c++17 -arch=sm_80 -o "$bin" "$here/bit-identity.cu"
build_code=$?
echo "# exit $build_code  ($nvcc -std=c++17 -arch=sm_80 -o bit-identity bit-identity.cu)"
[ "$build_code" -eq 0 ] || exit "$build_code"

run() { # run <label> [args...]
    local label=$1 out code
    shift
    out="$("$bin" "$@" 2>&1)"
    code=$?
    printf '\n## %s\n$ ./bit-identity %s\n%s\n# exit %d\n' "$label" "$*" "$out" "$code"
}

run "same values, checked"
run "copy from a null host pointer, checked" --copy-fails
run "allocation the device cannot satisfy (160 GB on an 80 GB card), checked" --n 40000000000
run "copy from a null host pointer, every return value ignored (the pre-fix harness)" --legacy --copy-fails
