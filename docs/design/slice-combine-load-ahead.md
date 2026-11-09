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
| 64 bytes (chosen) | 62 / 62 |
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
   and contiguous, strided, reversed (stride -1) and broadcast (stride 0) axes. Its nine input
   patterns hold NaNs (one, or several meeting in one lane), ±inf with a zero, -0.0, mixed
   signed zeros, near-overflow sums, products that overflow or underflow mid-walk, and
   subnormals, for F32 and F64 under all five ops.
   - Each result must equal, bit for bit, a CPU emulation of the serial walk and the
     xor/warp tree (the pattern of `product_partials_keep_the_serial_lane_walk_bit_for_bit`).
   - The one exception is the f64 arithmetic payload case above, where it must be NaN.
   - `combine_inputs_exercise_every_special_class` checks that the expected results really
     contain each class.
   - The test passes on the unmodified `ab7395c` kernel, so the emulation is the previous
     implementation on these inputs. It fails on a kernel that folds one batch in reverse.
2. **Old vs new, out of tree.** A driver-API program compares the two cubins byte for byte on
   every dtype/op route, 15 geometries, three layouts and both entries, and classifies each
   difference as value, NaN-ness or payload-only. Value and NaN-ness differences must be zero
   (`docs/worklogs/evidence/2026-11-09-combine-load-ahead/`).
3. **Full GPU suite**, `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`.
4. **SASS/resources.** Production-entry SASS of the new loop, `ptxas -v` for the four reduce
   entries in both trees, and `scripts/sass-diff.sh ab7395c`.

## Measurement

A100 80GB PCIe, CUDA 12.6, GPU idle (no compute process before each process, checked with
`nvidia-smi`), CPUs 0-7 below 5 % busy before each process, one GPU job at a time.

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
