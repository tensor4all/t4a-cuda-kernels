# Slice — closing the kernel performance regressions (U8a/U8b remainder)

Status: plan. Follows the paired experiment in
[`2026-10-21-complete-port-perf.md`](../worklogs/2026-10-21-complete-port-perf.md) and the
maintainer decision of 2026-10-21 to fix the regression class before the tenferro CUDA
adoption (tenferro-rs #2208) leaves the CubeCL path.

## 1. Question

The frozen paired experiment reports 39 of 53 cases above the 1.10x line of
[cubecl-performance-parity.md](cubecl-performance-parity.md) §5. This slice removes the
regression classes that are *constant-factor* work on the candidate side — metadata
local memory, index width, redundant wide arithmetic — re-applies them to the re-sliced
sources on `main`, and reruns the same frozen experiment once.

## 2. Diagnostics versus gate

The gate is the frozen timing criterion (valid case, candidate median > 1.10x baseline =
regression). Static per-entry resource usage (stack frame, registers, `LDL`/`STL`
count) is **diagnostic evidence only**: the previous round proved that deleting local
traffic can leave a case slower (small-table ``gather`` 19 % slower at zero local
traffic, `lu_apply_pivots` 1.65x, `qr_apply_phase` 1.8-2.2x). `elementwise.unary`'s 72 B
frame is libdevice's sin/cos slow path, not metadata, and stays.

## 3. Case ownership

Every flagged case has one owner and one mechanism, or an explicit disposition.

| Cases | Owner | Mechanism |
| --- | --- | --- |
| `layout-pad-*` (3.6-3.8x), `layout-slice-*` (1.7-10.7x) | A | metadata local memory in `window.cu`; fold the compact destination/source addressing |
| `indexing-gather-*` (1.7-2.5x), `indexing-scatter-add-*` | A | metadata local memory in `gather_scatter.cu` |
| `linalg-householder-explicit-v-*`, `linalg-qr-apply-phase-*`, `linalg-lu-apply-pivots` | A | metadata local memory in `linalg_aux.cu` + RRQR Q init (`identity_shifted`) |
| `layout-transpose-*` (1.36-1.96x) | B | tile configuration and index width in `movement.cu` |
| `layout-materialize-negative-stride` (1.22x), `layout-copy` component of `indexing-scatter-add-c64` | B | 64-bit index arithmetic in the generic signed-stride path (`movement.cu`) |
| `elementwise-add-f32-1000000` (1.14x) | B | index width in `arithmetic.cu` |
| `reduce-*` (1.14-1.33x) | B | wide division/address arithmetic and outer dispatch in `reduce.cu`; the split policy and reduction order stay |
| `linalg-rrqr-*` (1.22-1.75x) | B | address-taken batch metadata and per-element row walks in `rrqr.cu` |
| Fusion, `elementwise-{div-c64,exp,mul-c64}`, `add-f32-*large*`, already-`ok` cases | — | no action; they are at parity |

Dispositions allowed by this slice, each needing the measured number and the reason in
the worklog:

- **Residual constant factor**: work both arms perform identically (for example
  `lu_apply_pivots`' O(k) dependent pivot walk) or a few-instruction kernel whose gap is
  branch-guard overhead. Reported as residual, not as fixed.
- **Trade recorded in the previous round**: small-table `gather` spends a longer
  dependency chain per element to keep the metadata out of local memory. If that case is
  still slower than baseline after the port, it is reported with both numbers.

## 4. Correctness fixes this slice must carry

1. `t4a_cuda_axis_value`/`t4a_cuda_axis_add` read every slot of a coordinate array, so
   arrays must be fully initialized before use: a rank-2 diagonal op today reads
   uninitialized slots past `rank` (UB, masked by multiplying with zero). Initialize the
   whole `T4A_CUDA_MAX_RANK` array at every producer.
2. `t4a_cuda_decode` keeps arbitrary signed strides and offsets. Folding a destination to
   flat addressing is allowed only where the planner guarantees a compact operand; the
   generic decoder must not become linear-only.
3. ABI stays revision 8: same symbols, signatures, POD layouts, semantics. No extent
   specialization; the transpose tile/index-width variants are runtime configuration, and
   the specialization record required by the parity contract is updated with them.

## 5. Evidence tooling corrections

The phase-2 harness under `bench/` is brought onto `main` and fixed before the rerun:

1. **Semantic kernel pairing.** `summarize.py` currently pairs the two arms' kernels by
   decreasing device time, which already mis-paired RRQR entries. Replace it with an
   explicit per-case correspondence (baseline symbol → `t4a_cuda_*` entry, revision-8
   consolidations such as `initialize_q` → `t4a_cuda_identity_shifted`) and assert the
   instance counts agree.
2. **Evidence integrity.** A fresh campaign directory per campaign with no automatic
   reuse, no swallowed capture failures, every declared case × repetition required, and
   an incomplete capture making the campaign incomplete instead of shrinking the sample.
3. **Predicated local-memory instructions** count in `scripts/resource-usage.sh`
   (`@P0 LDL` and friends), so the diagnostic is not an undercount.
4. **RRQR operand contract.** The old harness passes one buffer as both reflector storage
   and trailing-update target; revision 8 requires disjoint operands. The harness must
   use disjoint buffers, and the correct modules per sequence (`identity_shifted` lives in
   `linalg_aux.cu`).
5. **Factorization state.** Both arms must start warm-up from the same input state; the
   worklog states what is actually measured instead of claiming every iteration starts
   from a restored matrix.

## 6. Acceptance

- `cargo fmt --check`, `cargo clippy -j 4 --all-targets --features gpu-tests -- -D
  warnings`, `cargo test -j 4`, `cargo test -j 4 --features gpu-tests` (local A100) green;
  ABI/header/manifest agreement tests green.
- One complete valid paired campaign: every case `ok`, or `REGRESSION` carrying the
  measured ratio and its disposition from §3, or `inconclusive` per the declared rule
  (whole-campaign rerun, not a selective retry).
- Worklogs: the local-memory/resource table (before/after), the per-case ownership
  verdicts, and the paired campaign table with the environment.
