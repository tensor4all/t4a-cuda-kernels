# Evidence: vectorized pass-1 walk (code at `65a74d5`; `e-`, `f-`, `g-` at `3b65e03`)

These are raw outputs, as produced. The worklog
`../../2026-11-08-reduction-pass1-vector.md` explains each one.

| path | what |
| --- | --- |
| `a-cargo-{test,clippy}-default*.log` | `cargo test` / `cargo clippy` without features: no GPU test binary is built |
| `a-cargo-{test,clippy}-gpu-tests.log` | the same with `--features gpu-tests` (local A100) |
| `b-nsys-65a74d5/*.csv` | pass-1 and pass-2 device time per case, `nsys stats --report cuda_gpu_kern_sum` |
| `b-native-65a74d5-run2/records.jsonl` | native paired samples: one line per case, side and repetition |
| `b-native-65a74d5-run1-invalid/` | the repeated campaign's first run (one baseline cell at 7 % IQR) |
| `b-tenferro-65a74d5/` | `gpu/isolated_ops` records (every sample's `batch_ms`), base/cand/base/cand, and the probe |
| `c-mutants/*.{patch,log}` | the seven mutants and their full test logs |
| `c-sass-diff.log`, `c-ptxas.log` | `scripts/sass-diff.sh 31b734e` and `ptxas -v` output |
| `d-shuffle-prototype/` | the revision-12-topology shuffle prototype: patch, bit-identity dumps, nsys timings |
| `e-cargo-test-gpu-tests-3b65e03.log` | `cargo test -j 4 --release --features gpu-tests --no-fail-fast` re-run on `3b65e03` (CUDA 12.6, local A100) |
| `f-sass-ldg128.log` | `cuobjdump --dump-sass` of the pass-1 build: `LDG.E.128` counts per entry on both trees and an instruction excerpt |
| `g-pointer-guard-mutants/*.{patch,log}` | the pointer-guard source contract against three mutants, and the old text-order test against the hoisting mutant |
| `scripts/` | the drivers that produced the above |

The scripts carry the absolute paths of the session that ran them: a scratch directory, and
the worktrees `~/tensor4all/t4a-cuda-kernels-pass1{,-base}` and
`~/work/t4a-pass1-{tenferro,bench}`. Adjust those paths to re-run them.
