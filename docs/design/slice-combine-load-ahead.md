# Slice: combine-pass load-ahead (tenferro-rs #2406, "what would still move it" 1 and 2)

## Problem

A split full reduction ends with one combine launch of `t4a_cuda_reduce` (strategy `PLANE`)
over the 4096 partials. On an A100 (nsys, tenferro-rs #2406, last comment) that launch takes
10.9 µs for `PROD` (one 32-lane block, 128 folds per lane) and 4.7-6.4 µs for
`SUM`/`MAX`/`MIN` (one 512-lane block, 8 folds per lane). `SUM_SQUARES` combines with `SUM`
(`ReduceOp::combine`), so it is the `SUM` cell. The partials were just written by pass 1 and
sit in L2. The time is therefore load latency on a dependent path, not bandwidth: 16 KiB
(f32) or 32 KiB (f64) in 10.9 µs is 1.5-3 GB/s.

## Cause

`plane_reduce_body` (and its general-rank twin `plane_reduce_body_any`) walks lane `t`'s
elements `t, t + W, t + 2W, ...` with one loop-carried accumulator:

```cpp
T acc = fold.seed(in[in_base + reduce_index * axis_stride]);
for (reduce_index += plane_width; reduce_index < axis_len; reduce_index += plane_width) {
  acc = fold(acc, in[in_base + reduce_index * axis_stride]);
}
```

The issue describes this as 128 dependent serial loads. The SASS at `ab7395c` (CUDA 12.6,
production entry) is shallower than that. ptxas runs a one-load remainder loop for
`(n - 1) % 4` trips, then a ×4-unrolled loop that issues four `LDG`s before the four dependent
`FMUL`/`DMUL`/`FADD`/`FMNMX`. In load batches per lane:

| combine | folds after the seed | load batches (seed + remainder + ×4) |
|---|---:|---:|
| `PROD`, 32 lanes over 4096 | 127 | 1 + 3 + 31 = 35 |
| `SUM`/`MAX`/`MIN`, 512 lanes over 4096 | 7 | 1 + 3 + 1 = 5 |

The loads do not depend on the accumulator. Only the folds have to stay in order.

## Change

One shared device helper, `t4a_cuda_plane_walk`, replaces the loop in both plane bodies (the
twins cannot drift).

- **Batching.** It issues `B` loads before folding any of them, then folds the batch strictly
  in index order with the same `fold(acc, x)` call.
- **The seed.** It is the first load of the first batch.
- **Predication.** Every other load is predicated on its own `index < axis_len`.
- **Batch depth.** `B` is a fixed 64 bytes of elements: 16 F32/I32, 8 F64/I64/C32, 4 C64.

`t4a_cuda_reduce` is one symbol for the unit and plane strategies of every dtype, so the batch
sets the register count of every launch of that entry (40 registers at `ab7395c`, `_any` 43).
Its many-block launches (unit strategy, direct plane, the plane combine of a few partials per
output) use 32-thread blocks, and the SM's 32-resident-block limit already caps those at 1024
threads. At or below 64 registers they therefore keep their occupancy. The values with CUDA
12.6:

| batch | `t4a_cuda_reduce` / `_any` registers |
|---|---|
| 64 bytes (chosen) | 61 / 61 (62 / 62 before the overflow fallback below) |
| 96 bytes | 109 / 109 |
| 128 bytes | 142 / 142 |
| 16 elements of every 8-byte type | 69 / 65 |
| 32 elements of every type | 175 / 174 (above the 128 a 512-thread block allows) |

A form with precomputed per-load offsets (fewer address instructions) reached 94 registers. In
quick three-run checks it was faster on `PROD` f32 (17.1 against 18.5 µs host/op) but slower
on the unit-path control `reduce-sum-f32-axis-513` (15.8 against 15.4 µs at `ab7395c`), so it
was not taken.

What does not change:

- **Fold order.** Lane `t` seeds from element `t` and folds `t + kW` for `k = 1, 2, ...` in
  increasing `k`, exactly as before. The same `t4a_cuda_plane_fold` follows (xor tree, plus the
  shared-memory warp step for a wide block). No regrouping, no reassociation, no change of
  which lane folds which element. Each fold is the same single `fold(acc, x)` expression.
- **Addresses.** Only elements `< axis_len` are loaded, the same set as before. A load is
  issued earlier, never at a new address, and the address of an element past the axis is never
  formed.
- **Counter range.** The batched walk forms `index + b * W` up to `axis_len + B * W`. An axis
  within `B * W` of the `long long` limit (only a broadcast view can be that long) takes the
  original one-load loop, verbatim, on a uniform branch. That loop never steps more than `W`
  past the end, so no input that was representable before overflows now. The loop is an
  out-of-line function (`t4a_cuda_plane_walk_serial`). Inlined, it changed the scheduling of
  the batched walk enough to fail the control gate (`reduce-min-f32-full-16777216` at 1.041 in
  a full paired campaign). Two counter forms
  that avoid the branch were tried and rejected: a remaining-distance bound (`PROD` f64 1M at
  42.5 µs) and a per-lane element count (80 registers, `axis-511` 11.9 µs).
- **ABI.** No symbol, signature, struct, launch geometry, planner, manifest or ABI revision
  changes. `scripts/sass-diff.sh ab7395c` lists exactly `t4a_cuda_reduce` and
  `t4a_cuda_reduce_any`; the other 92 entries, pass 1 included, are identical.
- **Out of scope:** the unit strategy (`unit_reduce_body*`).

### What is and is not bit-identical

Every result bit inside the ABI contract is unchanged: every non-NaN value, signed zero and
infinity, whether a result is NaN, and the NaN payload the extrema select. The one exception
is outside the contract and was already unstable before this change: which NaN payload an
**f64/c64** `SUM`/`PROD`/`SUM_SQUARES` keeps when two NaNs with *different* payloads meet in
one add or multiply.

- **The hardware rule.** An f64 `DADD`/`DMUL` of two NaNs returns the payload of its second
  operand.
- **The compiler's choice.** ptxas picks the operand order per instruction, independently of
  the C++ order. At `ab7395c` the remainder loop emits `DADD R2, R8(x), R2(acc)`; the ×4 body
  emits `x, acc` for its first fold and `acc, x` for the other three.
- **The consequence in the old kernel.** The same lane already keeps a different payload at
  axis length 3001 than at 4096 (`combine_keeps_the_serial_lane_walk_bit_for_bit`, pattern 1).
- **The consequence for this change.** Any restructured walk moves those operand orders.

f32 arithmetic returns the canonical NaN (`0x7fffffff`), and integers have no NaN, so neither
is affected. The extrema select their NaN in source order, so they are not affected either.
`docs/abi/c-kernel-abi.md` ("NaN") already states that the payload is not part of the
contract, only "is NaN" is.

Rejected:

- Staging the partials in shared memory: static shared memory belongs to the whole entry and
  would cut occupancy on every path.
- A wider `PROD` block: it changes the topology.
- `prefetch.global.L1` before the unchanged loop (`CCTL.E.PF1`): measured slower (`PROD` f32
  1M: 21.9 → 27.2 µs host/op).

## Acceptance

1. **Bit identity.** `combine_keeps_the_serial_lane_walk_bit_for_bit` (`tests/gpu_reduce.rs`)
   drives `t4a_cuda_reduce` and `t4a_cuda_reduce_any` directly with strategy `PLANE` at the
   planner's combine widths. It covers the combine geometries (32 lanes for `PROD`, 512 for
   the others, over 4096), ragged lengths, `axis_len == plane_width`, short direct-plane axes,
   and contiguous, strided, reversed (stride -1) and broadcast (stride 0) axes.
   - Its eleven input patterns hold: one NaN; NaNs meeting in one lane's walk (lane 7's seed,
     element 1 and element 3); ±inf with a zero; -0.0; mixed signed zeros; near-overflow sums;
     products that overflow or underflow mid-walk; subnormals.
   - Two patterns each put two NaNs after a finite seed in one lane, alone in their output so
     the xor tree cannot mask them: the first batch, and a steady-state batch. They pin the fold
     order *inside* a batch through the extrema.
   - All of it runs for F32 and F64 under all five ops.
   - Each result must equal, bit for bit, a CPU emulation of the serial walk and the
     xor/warp tree (the pattern of `product_partials_keep_the_serial_lane_walk_bit_for_bit`).
   - The one exception is the f64 arithmetic payload case above, where it must be NaN. It is
     applied only where two distinct NaN payloads actually meet in one lane (both NaNs inside
     the axis), never to a single NaN or to the extrema.
   - `combine_inputs_exercise_every_special_class` checks that the expected results really
     contain each class.
   - The test passes on the unmodified `ab7395c` kernel, so the emulation is the previous
     implementation on these inputs.
   - It fails on a kernel that folds the steady-state batch in reverse, and on one that does so
     only for `MAX`/`MIN`.
2. **Old vs new, out of tree.** A driver-API program compares the two cubins byte for byte on
   every dtype/op route, 15 geometries, three layouts and both entries. It classifies each
   differing output as value, NaN-ness or payload-only (`docs/worklogs/evidence/2026-11-09-combine-load-ahead/`).
   - Value and NaN-ness differences must be zero.
   - A payload-only difference is tolerated only on f64/c64 `SUM`/`PROD`/`SUM_SQUARES`; on any
     other route it is a failure.
3. **Full GPU suite**, `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`.
4. **SASS/resources.** Production-entry SASS of the new loop, `ptxas -v` for the four reduce
   entries in both trees, and `scripts/sass-diff.sh ab7395c`.

## Measurement

A100 80GB PCIe, CUDA 12.6, one GPU job at a time.

- **GPU idle.** No compute process before each process, checked with `nvidia-smi`.
- **CPU isolation.** The measured processes run on CPUs 0-7, one L3 domain (checked from
  sysfs), and must be below 5 % busy before each process. The campaign driver and its probes
  run on CPUs 8-15, another L3 domain. Other work stays on 16-63.

- **Host µs/op.** The same `bench_kernels` source (with `reduce-prod-{f32,f64}-full-16777216`
  added) built on `ab7395c` and on this branch, run by the evidence directory's `campaign.sh`.
  - 3 warm-up repetitions (recorded, excluded), then 15 sample repetitions.
  - Per repetition, one process per side runs every case, the side order alternating AB/BA.
    Each case runs 2000 settling and 10 warm-up launches, then 100 timed iterations closed by
    one synchronize. A sample is therefore one synchronized batch.
  - Median and IQR over the 15 samples.
- **Cases.** `reduce-{sum,prod,max,min}-{f32,f64}-full-{1048576,16777216}` and
  `reduce-sumsquares-f32-full-1048576`. Controls: axis 511/512/513/600 (direct plane and the
  many-output unit combine), 4096² both axes, f64, c64, i32, 1e6/1e7, 4×131072.
- **Device time.** nsys per-kernel time of the combine launch, both trees, full-1M cells.
- **Gate.** A control slower than 1.03× on host time per op (the rule of #23/#26) blocks the
  change.
- **Not measured.** The general-rank entry has no `bench_kernels` case. Its loop is the same
  helper, and its registers are reported above.

## Results (2026-11-09)

Evidence directory: `docs/worklogs/evidence/2026-11-09-combine-load-ahead/`.

- **Combine device time** (`nsys/table.md`, median of three alternating rounds):
  - `PROD` f32 11.60 → 8.31 µs; `PROD` f64 11.90 → 10.33 µs.
  - `SUM`/`MAX`/`MIN`/`SUM_SQUARES` 5.4-6.5 → 5.0-5.7 µs.
  - Pass 1 is unchanged.
- **Host µs/op** (`campaign/table.md`):
  - `PROD` f32 1M 21.82 → 18.34 (0.841); `PROD` f64 1M 22.12 → 20.59 (0.931).
  - The other required 1M/16M cells are 0.939-1.007.
- **Control gate: one control fails.** `reduce-sum-f32-axis-513` (many-output unit combine) is
  at 1.037 against the 1.03 rule. That cell was 0.999 in a full paired campaign of the same
  walk without the near-`LLONG_MAX` fallback (every control ≤ 1.010 there), and 1.024 with the
  fallback inlined. The fallback is the cost. Whether to keep it (no new overflow at any
  representable length) or drop it (all controls pass; an axis within `B * W` of `2^63` would
  overflow `index`) is left to review.
- **Not reached.** "Most of the 10.9 µs" for `PROD`: about 28 % of the f32 combine goes.
  - On one warp, the address arithmetic of a predicated batch (≈ 18 instructions per load in
    the f64 loop) is now comparable to the L2 latency.
  - More loads in flight cost registers above the 64-register occupancy threshold of the
    shared entry.
