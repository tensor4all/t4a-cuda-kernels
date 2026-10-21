# Reductions slice — local evidence

Date: 2026-10-20. Plan: [slice-reductions.md](../design/slice-reductions.md).

## What landed

ABI revision 3 adds two entries that carry five operations and three strategies:

- `t4a_cuda_reduce` (unit / plane), `t4a_cuda_reduce_partials` (split pass 1),
  for `SUM`, `PROD`, `SUM_SQUARES`, `MAX`, `MIN`.
- `src/lib.rs` gains `ReduceOp`, `ReduceStrategy`, `Split`, `ReductionPlan`,
  `split_policy`, `plan_reduce`, `plan_reduce_partials` and `plan_reduction`
  (the baseline's Auto decision as a pure host function).
- `THIRD_PARTY_NOTICES.md` carries the cubek-reduce notice inherited through
  tenferro-rs; `cuda/src/reduce.cu` and the reduction-planning region of
  `src/lib.rs` carry the matching derivation notices.

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
cargo test -j 16                                                       # 36 planning tests pass, no CUDA
cargo test -j 16 --features gpu-tests                                  # 36 planning + 13 + 16 + 11 GPU tests pass
```

GPU coverage: every op and dtype route in both strategies (F32/F64 for all five,
I32/I64 for sum/prod/max/min, C32/C64 for sum/prod), over a rank-3 middle axis; a
strided input with an offset; the strategy boundary at 31/32/33 and subwarp widths
8/16; split reductions at 512/513/1000/8448 with every partial inspected from a
sentinel-filled allocation and both passes enqueued before any observation;
split versus direct agreement for prod and a complex split; a middle-axis split
with a gapped, offset partials allocation and a non-compact output, checking
logical values, untouched holes and the strided output; `SUM_SQUARES` combining
with a plain `SUM` and rounding before the addition (bitwise F32 case); a NaN
confined to one split surviving `MAX`/`MIN`; single-element axes keeping `-0.0`
and a complex `INF` component; NaN/signed-zero extrema; integer wrap at the
`i32/i64` boundaries; exactly one physical launch for a direct reduction and two
for a split; and 2^16-element domains in both shapes.

## Findings

- Two kernel-side bugs were caught by the tests before any review: the
  accumulator seed must apply the operation's own transform (a sum-of-squares
  reduction was seeding with the unsquared first element), and the plane
  collective must combine lane **accumulators** with the fold's combine step
  (sum-of-squares was squaring accumulators again, overflowing to `inf` at 40
  elements).
- Float extrema need `fmax`/`fmin` semantics rather than a ternary comparison so
  that `max(-0.0, +0.0)` and `min(-0.0, +0.0)` match the baseline's ordered form.

## Review round

An independent review of the diff found and this branch fixes: a division-by-zero
panic in `split_policy` for an unsupported plane width (now a typed error); the
missing CubeK derivation notice for the translated host policy; coverage claims
that the matrix did not support (int/complex routes now run in both strategies,
plus split prod/complex and a gapped-partials middle-axis case); a missing
bitwise FMA-sensitive sum-of-squares case; the `ReductionPlan::Split` field that
duplicated an always-plane pass 1; phantom `plane_width` arguments in the design
prototypes (the kernels read it from `blockDim.x`); the SUM_SQUARES exception to
"length-1 returns the element"; the split interval wording excluding the
remainder; the partials-aliasing rule contradicting pass 2 reading partials; and
an unrelated, mis-attributed Julia notice block, replaced by an explicit deferral.

## Not done in this slice

Multi-axis reduction, non-keepdims outputs, conversions, gather/scatter,
linalg/RRQR, the fused-expression emitter and the NVRTC prepare/cache path. No
hosted CI was used and no performance comparison against the CubeCL baseline is
claimed.
