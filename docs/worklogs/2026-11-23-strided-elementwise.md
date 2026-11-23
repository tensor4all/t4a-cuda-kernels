# 2026-11-23 — layout-taking elementwise entries (ABI 15)

Consumer: tenferro-rs#2701 (the CUDA engine staged every non-compact elementwise operand into a compact copy, which
puts an allocation and a launch inside a timed call).

## What was done

- `t4a_cuda_strided_{unary,binary,compare,select}` and their `_any` twins in `cuda/src/elementwise.cu`; planners and
  `StridedBinaryOp` in `src/lib.rs`; ABI revision 15; crate 0.4.0.
- The flat binary/compare bodies were refactored into value helpers (`binary_float_value`, `binary_int_value`,
  `compare_value`) shared with the strided bodies.
- Scalar-side `BinaryOp::{Add, Sub}` (tags 14, 15) on `t4a_cuda_binary`, added at the consumer's request for
  composite constants.

## Findings

- **Complex multiply contraction.** The first GPU run had C64 `MUL` differ between `t4a_cuda_mul` and the strided body.
  Spelling the contraction out (`t4a_cuda_mul_elementwise`) in the form the flat SASS already used
  (`DMUL t = l.im*r.im; DFMA re = l.re*r.re - t`, `DMUL u = l.im*r.re; DFMA im = l.re*r.im + u`; same for FFMA)
  pins both entries to one form; the flat entry's SASS has the same instruction structure after the change (register
  allocation differs).
- **NaN sign/payload.** The remaining C64 `MUL` mismatch was 5 of 4355 elements, all NaN components: strided
  `0x7ff8000000000000`, flat `0xfff8000000000000`. `DFMA`/`DMUL` propagate an input NaN by operand position, and the
  negated product operand carries the flipped sign. The ABI now states that a NaN result's sign and payload are
  unspecified (the revision 14 golden has the same class); the tests compare NaN components as equal and everything
  else bit for bit. Scalar-side `ADD`/`SUB` have no such freedom and are compared fully bitwise, NaN included.

## Evidence

Host primerose: NVIDIA A100 80GB PCIe, driver 580.178.04, CUDA 12.0 (V12.0.140) `nvcc`, `sm_80`.

| Command | Result |
| --- | --- |
| `cargo fmt --check` | clean |
| `cargo clippy -j 16 --all-targets -- -D warnings` | clean |
| `cargo clippy -j 16 --features gpu-tests --test gpu_strided_elementwise -- -D warnings` | clean |
| `cargo test -j 16` | all pass (planning: 92) |
| `srun --gres=gpu:1 -c 8 cargo test -j 8 --features gpu-tests` | all pass; log in `evidence/2026-11-23-strided-elementwise/cargo-test-gpu-tests.log` |
| `nvcc --ptxas-options=-v` on `elementwise.cu` | new entries: 0 spill stores/loads; `strided_unary(_any)` has the same 72-byte stack frame as the flat `t4a_cuda_unary` (math-library range reduction), the others 0 |

`cargo clippy --all-targets --features gpu-tests -- -D warnings` reports 33 `needless_borrow` errors in
`examples/bench_kernels.rs` and `tests/gpu_dynamic_rank.rs` on `main` as well (a newer clippy); not touched here.

## Not done

- The plan was not reviewed independently before implementation; the finished diff was (see the PR).
- Strided *reductions* need nothing new: `t4a_cuda_reduce` already takes layouts.
