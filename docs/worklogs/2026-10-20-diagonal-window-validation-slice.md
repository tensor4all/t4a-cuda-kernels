# Diagonal, windowing and validation slice — local evidence

Date: 2026-10-20. Plan:
[slice-diagonal-window-validation.md](../design/slice-diagonal-window-validation.md).

## What landed

ABI revision 2 adds `t4a_cuda_axis_values` and twelve entries in three new
kernel sources:

| Source | Entries |
| --- | --- |
| `cuda/src/diagonal.cu` | `extract_diagonal`, `embed_diagonal_copy`, `tril`, `triu` |
| `cuda/src/window.cu` | `slice`, `dynamic_slice`, `pad` |
| `cuda/src/validation.cu` | `init_validation`, `validate_float_indices`, `extract_invalid_float_index`, `validate_real_cast`, `extract_invalid_real_cast` |

Two baseline defects are handled explicitly rather than reproduced:

- `extract_diagonal_kernel` never skips the diagonal output axis while walking
  the remaining output coordinates, so rank >= 3 reads the wrong input element.
  The port implements the intended map; an independent CPU oracle checks it.
- `tril_kernel`/`triu_kernel` compare `col - k`, which is signed-overflow UB for
  extreme `k`. The port compares `col - row` against `k` and tests
  `k = i64::MIN/MAX`.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe (`nvidia-smi`) |
| Driver | 580.126.09 |
| CUDA toolkit | 12.6 installed; the tests invoke `nvcc` 12.0.140 from `PATH` |
| Test compile flags | `nvcc -std=c++17 -cubin -arch=sm_80 -I cuda/include` |
| Rust | 1.97.1 |

## Commands and results

```bash
cargo fmt --check                                                      # pass
cargo clippy -j 16 --all-targets --features gpu-tests -- -D warnings   # pass
cargo test -j 16                                                       # 30 planning tests pass, no CUDA
cargo test -j 16 --features gpu-tests                                  # 30 planning + 13 + 11 GPU tests pass
```

GPU coverage: extract-diagonal against an independent oracle built by
enumerating input coordinates (not by walking output cursors) for rank 2/3/4
inputs, both axis orders and every diagonal position, plus a strided input whose
addresses hold distinct values and NaN payload bits; embed composed with
`fill_zero`, and a second case with a gapped strided destination zeroed through
`fill_zero_view` so holes and the allocation tail are compared as well;
tril/triu for rank 2 and batched rank 3 with `k` of both signs and the `i64`
extremes; slice with steps > 1; dynamic-slice clamping for I32/I64/F32/F64 device
starts on axes that actually clamp; pad with positive, negative and interior
padding; validation flag selection (earliest position wins, no-write when valid,
F32 exact-integer cutoff, stride-2 sampling with nonfinite values only in skipped
slots, inclusive/exclusive max boundary, a huge-stride plan); and 2^16-element
runs for every entry that maps a tensor domain (tril, triu, extract, embed, pad,
slice, dynamic-slice), each asserting one physical launch.

## Review round

An independent review of the diff found and this branch fixes: a signed-overflow
`position * stride` in `validate_real_cast` (now bounds-checked by division); the
static-slice length rule (the output extent may be shorter than the reachable
count, which is how the baseline's explicit `limit` is expressed); a missing
destination-injectivity check on embed; dropped explicit infinity rejection in
both validation predicates (bit-pattern finiteness test); the validation domain
cap applied to the buffer length instead of the sample count; a diagonal oracle
that mirrored the kernel's own cursor walk; constant-valued strided-diagonal
fixtures; missing float index dtypes and a non-clamping axis in the dynamic-slice
test; and missing multi-block coverage and launch-count assertions. The ABI
document now also states the status-buffer sizes, alignment, disjointness and the
initialize -> validate -> extract ordering.

## Notes

- The real-cast validation flag holds the *sampling position* (thread index),
  not the element index; the extracted value is the source element. This matches
  the baseline and is now stated in the ABI document.
- `t4a_cuda_pad` takes `edge_high` only so the facade can derive and check the
  output extents; the kernel never reads it.
- gather/scatter, conversions and reductions remain deferred (see the plan).

## Not done in this slice

No hosted CI was used (the repository has no workflows), and no performance
comparison against the CubeCL baseline is claimed.
