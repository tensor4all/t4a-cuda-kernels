# Slice — vectorized pass-1 lane mapping (tenferro-rs #2406, part 2)

Status: implemented and measured, 2026-11-08. Follows the residual of
[slice-reduction-throughput.md](slice-reduction-throughput.md) §8 and §9.5: pass-1 streaming is the
largest remaining term of a full reduction. Outcome: §8.

## 1. Question

At 16M elements, the split reduction's pass 1 (`t4a_cuda_reduce_partials`) reads far below the
A100's DRAM bandwidth, and the extrema fall well below `sum` on identical traffic. Can a
16-byte-vector lane mapping with batched loads remove that gap, without touching the plan, the
launch contract, `PROD`, or any strided path?

## 2. The current pass-1 walk (`t4a-cuda-kernels@e1f56e4`)

- **Geometry.** One block per `(output, split)`, `block = plane_width` (32 for tenferro), so a
  block is one warp. `cuda/src/reduce.cu:276-279` decodes `cube = blockIdx.x` into
  `output_index = cube / splits` and `split = cube % splits`.
- **Chunk.** `cuda/src/reduce.cu:295-296`: `[start, end)` with `start = split * chunk_len`, and
  `end = axis_len` for the last split (the floor remainder), else `start + chunk_len`.
- **Lane to element.** `cuda/src/reduce.cu:297-301`: lane `t` seeds from element `start + t`
  and folds `start + t + k*W` for `k = 1, 2, ...` while `< end`. One scalar load per lane per
  step, and one loop-carried accumulator.
- **Coalescing.** With `axis_stride == 1`, a warp's step reads `W` consecutive elements: a
  128-byte transaction for `f32`, 256 for `f64`. With `axis_stride != 1` each lane reads its
  own sector.
- **Bounds and tail.** There is no per-element bounds check beyond `reduce_index < end`. The
  planner guarantees `chunk_len >= plane_width` (`src/lib.rs:3136-3150`, the
  `MIN_CHUNK_PLANES = 8` floor), so every lane seeds inside its chunk. The tail is the last
  partial stripe: lanes whose next index reaches `end` stop, so a ragged stripe is handled by
  the loop condition, not by a branch.
- **The general-rank twin.** `partials_body_any` (`cuda/src/reduce.cu:391-440`) has the same
  walk over the dynamic-rank layout.

Measured device time of pass 1 (nsys, 100 launches, the `examples/bench_kernels.rs` cells) at
`e1f56e4` plus the 16M harness cases:

| case | pass 1 (µs) | GB/s |
| --- | ---: | ---: |
| `sum` f32 16M | 60.9 | 1102 |
| `max` f32 16M | 149.2 | 450 |
| `min` f32 16M | 148.7 | 451 |
| `sum` f64 16M | 94.2 | 1425 |
| `max` f64 16M | 163.3 | 822 |
| `min` f64 16M | 163.6 | 820 |
| `sum` f32 1M | 8.9 | 472 |
| `max` f32 1M | 10.4 | 403 |
| `sum` f64 1M | 9.0 | 934 |
| `max` f64 1M | 10.7 | 781 |

The A100 80GB PCIe's theoretical DRAM bandwidth is 1935 GB/s. Each lane walks 128 steps at
16M, and the same 16M `f32` input costs 149 µs under `max` and 61 µs under `sum`: about 1.2 µs
per step of each block wave, which is one DRAM latency per step. The extrema fold's
NaN-selecting branches stop the compiler from issuing the next loads before the current fold
completes. `sum` gets that overlap from unrolling.

## 3. Change

One device helper, shared by `partials_body` and `partials_body_any` so the twins cannot
drift, plus a uniform per-block choice of walk:

- **Eligibility.** The vector walk is used only when all of these hold:
  - `axis_stride == 1`;
  - the operation may be regrouped (`SUM`, `SUM_SQUARES`, `MAX`, `MIN`; a `kRegroup` trait on
    the fold);
  - the element type is `float`, `double`, `int` or `long long`, giving 4, 2, 4 and 2 lanes
    per 16-byte vector;
  - the chunk holds at least one group of `V` elements per lane (`len / V >= W`).

  Everything else takes the existing scalar walk, unchanged: strided axes, `PROD`, `C32` and
  `C64` (`t4a_cuda_c32` is 4-byte aligned, not a `float2`; `c-kernel-abi.md:55-56`), and short
  chunks. The chunk test is a guard, not a reachable case under the current planner:
  `chunk_len >= 8 * W` gives at least `2W` groups. The stride is tested before the chunk
  pointer is formed (§9).
- **Mapping.** The chunk divides into `groups = len / V` groups of `V` consecutive elements,
  counted from the chunk's first element, so the grouping never depends on the address. Lane
  `t` takes groups `t, t + W, t + 2W, ...`, so on a 16-byte-aligned chunk one warp
  instruction reads 512 contiguous bytes. A chunk that starts off a 16-byte boundary loads each
  group with `V` element loads instead, with the same grouping. The `len mod V` trailing
  elements are read one each by lanes `0..len mod V`.
- **Batched loads.** The vector loop issues four independent 16-byte loads per lane before it
  folds any of them, so memory-level parallelism no longer depends on how the compiler
  schedules the fold. The first and the last batch are predicated per group.
- **Fold order within a lane.** Each vector combines as a pairwise tree, `(seed(x0) ⊕
  seed(x1)) ⊕ (seed(x2) ⊕ seed(x3))`, which squares each element once for `SUM_SQUARES`. The
  accumulator is seeded from the lane's first vector, so no identity value is introduced and a
  `-0.0` sum stays `-0.0`. The tail elements fold in last.
- **Code generation of the scalar path.** The vector walk is a `__noinline__` device
  function, so the strided and `PROD` instantiations keep their own straight-line walk. That
  is the regression mechanism of the withdrawn grouped walk (throughput slice §9.2 and §9.4):
  it costs 1.13x on `reduce-sum-f32-4096sq-axis1` even when not executed. It is checked by SASS
  diff and the paired sentinel cells below.

**What does not change.**
- The plan: `split_policy`, `MAX_SPLITS`, `MIN_CHUNK_PLANES`, the 64-partials wide-combine
  threshold, and `combine_geometry`.
- The launch geometry, entry symbols, signatures and structs.
- The set of elements each block reduces.
- Pass 2.
- The serial lane-strided walk of every case outside the eligibility above.
- `PROD`'s walk, which stays bit-for-bit.

**ABI: revision 13.** The ABI's "Pass-1 lane assignment (revision 13)" section in
`docs/abi/c-kernel-abi.md` states the topology.
- Revision 12 kept every element on the lane of the elements `W` apart from it, and allowed
  only regrouping inside that lane's walk. Revision 13 puts an element on the lane of its
  group neighbours. That is a different numerical topology, and the ABI's bump rule now names
  it as a bump trigger. No symbol, signature, struct or launch geometry changes.
- Which inputs change, and how (eligible cases only; everything else is bit-identical):
  - `SUM`/`SUM_SQUARES` over `F32`/`F64` round differently.
  - Integer `SUM` is unchanged, modulo wrap.
  - `MAX`/`MIN` select the same value; only a NaN payload can differ.
  - Near overflow the classification can flip either way. For an `F32` `SUM` of 1024
    elements at `W = 32`, with `A = 2^127` and all other elements zero:
    - `x0 = x1 = A, x32 = x33 = -A` gave `0` under revision 12 and gives `NaN` now;
    - `x0 = x32 = A, x1 = x33 = -A` gave `NaN` under revision 12 and gives `0` now.

    `tests/gpu_reduce.rs::pass1_overflow_follows_the_revision_13_grouping` pins both. It fails
    on the revision-12 kernel.
- The NaN of the first witness is not a defect of empty or padding lanes:
  - With `A = 1`, the same placement sums to exactly `0`.
  - Two tests pin that nothing outside a lane's own groups and tail contributes: guard NaNs
    around the view, and a single `1.0` at every chunk position.
  - An empty axis never reaches the kernels: the planner refuses it, and the caller supplies
    the identity (0 for a sum).
- The topology depends only on `len`, `W`, the element type and the operation. Results are
  reproducible for a given input, layout, plan and build, wherever the input sits.
- Consumers' compiled-module caches key on the composed source, which contains
  `T4A_CUDA_ABI_REVISION`, so they invalidate. A consumer that records the revision (the
  tenferro-rs pin note) moves from 12 to 13 when it adopts this.

## 4. Correctness matrix (frozen)

New GPU tests in `tests/gpu_reduce.rs` (and the general-rank twin in `tests/gpu_dynamic_rank.rs`)
against a CPU `f64` reference, every op of §3 × F32/F64/I32/I64 where the op supports the dtype:

- `n % V != 0` and `n % (V*W) != 0` (a tail inside the last stripe), at lengths on both sides
  of the split threshold, plus `n < V` (direct plan; guards the planner boundary).
- A misaligned base: `in_layout.offset = 1, 2, 3` elements.
- Aligned base with misaligned chunk starts: axis 1028 with 4 splits gives `chunk_len = 257`,
  so splits 1 to 3 start off a 16-byte boundary while split 0 is aligned, all in one launch.
- Per-partial checks against the CPU sum of each chunk (pass 1 alone, sentinel-filled
  partials), so a lane that reads a vector twice, or skips one, fails at the partial rather than
  being averaged away.
- `MAX`/`MIN` with a NaN in the first group, in the body and in the tail; `SUM_SQUARES` seeds.
- `PROD` partials bit-identical to a CPU emulation of the scalar lane-strided walk plus the xor
  tree (the topology must not change).
- A strided axis stays on the scalar walk, which the existing strided tests cover.

## 5. Measurement protocol (frozen before the first candidate timing)

**Native/native paired** (`bench/run-native-pair.sh`):
- Arms: baseline = this branch's harness commit (library `e1f56e4`), candidate = this branch.
- 3 repetitions, AB/BA order, CPUs 0-7, under the bench slot.
- Cases:
  - `reduce-{sum,max,min}-{f32,f64}-full-{1048576,16777216}`;
  - `reduce-prod-{f32,f64}-full-1048576` (an unchanged-path guard);
  - `reduce-sumsquares-f32-full-1048576`;
  - `reduce-sum-f32-4096sq-axis1` and `reduce-max-f32-4x131072-axis1` (strided sentinels);
  - `reduce-sum-f32-axis-600` (2 partials);
  - `reduce-sum-f32-4096sq-axis0`, `reduce-sum-f64-4096sq-axis0`,
    `reduce-max-i32-4096sq-axis0` (contiguous multi-output);
  - `reduce-sum-c64-4096sq-axis0` (excluded dtype);
  - `reduce-sum-f32-full-{1000000,10000000}`.
- Pass-1 device time and GB/s come from nsys over the same binaries.

**Through tenferro (`gpu/isolated_ops`)**:
- One tenferro-rs commit, `origin/main`, built twice. The candidate overrides the kernels with
  `--config 'patch."https://github.com/tensor4all/t4a-cuda-kernels".t4a-cuda-kernels.path=...'`.
- Arms: `tenferro-cuda-direct`, `tenferro-cuda-eager` and `pytorch-cuda`, 15 samples and 3
  warmups each.
- Order base, cand, base, cand in one session, CPUs 0-7, under the bench slot.
- Probe cells: `reduce_{sum,max,min,prod}_full_{f32,f64}_n1048576` and
  `reduce_{sum,max,min}_full_{f32,f64}_n16777216`.
- `verification` must be `passed` against `cpu_fp64` on every cell.

**Gate (item 5 of the task).**
- **Primary metric:** the median, over the six `n16777216` cells, of the direct-arm ratio
  candidate ÷ baseline, PyTorch-drift-normalised.
- **Adopted** only if that median is ≤ 0.95, and no native paired case is worse than 1.03x the
  baseline median.
- **Otherwise** no PR is opened, and the measured result is reported.
- The `n1048576` cells are reported alongside. Pass 1 is L2-resident there, so they are not the
  gate.

## 6. Residuals declared up front

- The pass-2 combine costs 5-9 µs per call at 4096 partials (wide block, 8 folds per lane). It is
  untouched here.
- The direct (unsplit) plane path keeps scalar loads. It only serves axes shorter than 512
  elements.
- `C32`/`C64` and `PROD` stay scalar, by contract.
- The tenferro-rs pin moves only after a `t4a-cuda-kernels` release, which is not part of this
  slice.

## 7. Review

The repository asks for an independent `openai/gpt-6.1-sol` review through the
`independent-review` skill. That skill is not installed on this host. A fresh-context reviewer
agent stands in, and the substitution is stated in the PR.

## 8. Outcome (measured)

**Adopted.**
- The primary metric (§5) is **0.609**, against the ≤ 0.95 rule. It is the median over the six
  `n16777216` cells of the direct-arm ratio through tenferro, drift-normalised.
- The native paired campaign's worst case is **1.009x** (`reduce-prod-f64-full-1048576`), against
  the 1.03x limit.
- Pass 1 at 16M reaches 70-79 % of the A100's 1935 GB/s, up from 23-74 %. For example,
  `max` f32 goes from 149.3 µs to 49.2 µs, and from 450 GB/s to 1363 GB/s.
- `PROD`, strided and complex cells are unchanged within noise (0.994-1.009).
- All of these are measured on code `65a74d5` (revision 13).

The tables, commands and environment are in
[../worklogs/2026-11-08-reduction-pass1-vector.md](../worklogs/2026-11-08-reduction-pass1-vector.md).

**One adjustment after the first campaign.** In the first candidate, a lane loaded only its
first vector before the batch loop, and took the vectors left after the loop one at a time.
Chunks of 2-4 vectors per lane (1M elements, and the 600-long axis) therefore never ran a
batch, and paid one DRAM latency per vector:
- `reduce-sum-f32-axis-600` measured 1.053x;
- `reduce-sum-f64-full-1048576` measured 1.029x.

The fix predicates the first and the last batch per vector, so a short chunk also issues its
loads together. The whole campaign was then repeated, and the tables are from the repeat.

**Instruction evidence.**
- `scripts/sass-diff.sh 31b734e`: 92 entries identical. Only `t4a_cuda_reduce_partials` and
  `t4a_cuda_reduce_partials_any` differ, and both carry the new walk.
- The pass-1 entries go from 39-40 to 52-54 registers, with no spills. With one-warp blocks,
  the 32-blocks-per-SM limit still binds before registers do.
- The vector bodies emit `LDG.E.128`.

## 9. Review fixes (GPT-6.1-sol review of #23, points 1 and 2)

- **Stride before pointer.** The pass-1 dispatch used to form `in + in_base + start` before it
  tested `axis_stride == 1`.
  - On a broadcast or reversed split axis, that pointer can lie outside the allocation:
    `extents=[512], strides=[0]` over one element forms `in + 256`, and `strides=[-1],
    offset=511` forms `in + 767`. Forming such a pointer is undefined behaviour.
  - The stride is now tested first, and the chunk pointer is formed only on a contiguous
    axis.
  - The source-contract test `tests/planning.rs::pass1_checks_the_axis_stride_before_forming_a_chunk_pointer`
    fails on the old code and passes now.
  - `gpu_reduce.rs::pass1_reads_broadcast_and_reversed_axes_inside_their_allocation` runs both
    layouts on exact-size allocations. The UB is not observable, so this test passed before
    the fix as well.
- **Topology independent of the address.** The vector groups were counted from the chunk's
  first 16-byte boundary, so the same values, layout and plan grouped differently when the
  input pointer moved by 4 bytes.
  - With `[A, A, -A, -A, 0, ...]`, `A = 2^127`, the result was `NaN` at one alignment and `0` at
    another.
  - Groups are now `[g*V, g*V + V)` counted from the chunk start. Alignment only chooses
    between one 16-byte load and `V` element loads, and the remaining `len % V` elements are
    the tail.
  - `gpu_reduce.rs::pass1_topology_does_not_depend_on_the_input_address` asserts identical
    bits at leads 0-3. It failed on the old code with `0 vs NaN`.
- **Effect.**
  - A chunk that starts on a 16-byte boundary keeps exactly the previous grouping and
    instructions. This covers tenferro's 1M and 16M full reductions.
  - A chunk that starts off the boundary now uses element loads. `reduce-sum-f32-full-10000000`
    measures 1.034x the previous head, and is still 0.77x the original baseline.
- **Point 3 (the lane assignment widens revision 12's contract).** This is resolved by
  revision 13, with the ABI text rewritten to the implementation (§3, "ABI: revision 13").
- **Restoring revision-12 lanes with warp shuffles was not done.**
  - Lane `t` would need elements `t + 32j`, which sit in four different lanes' vectors. That
    is about four shuffles and selects per element, and roughly 10-15 µs at 16M on a 50 µs
    pass.
  - It would give up a quarter of the 16M gain, only to keep the old near-overflow
    classification, which can flip in either direction.

## 10. Restoring the revision-12 lanes with warp shuffles: measured, not adopted

A prototype was built to answer this; it is not on this branch.
- **Design.** 16-byte loads, with every element shuffled back to its revision-12 lane.
- **Bit identity.** It is bit-identical to the revision-12 serial walk at 1M and 16M.
- **Cost.** **+5.6 to +7.8 µs per call** of 16M pass-1 device time over the vector walk for
  F32/F64 `SUM`/`MAX`/`MIN` (`SUM` f64 +2.9 µs), which is 12-16 %. It is close to nothing at
  1M.
- **Why not adopted.**
  - It buys only bit compatibility with a topology that is not more accurate: the two
    documented witnesses flip in opposite directions.
  - It would still need code for chunks that start off a 16-byte boundary and for narrow
    planes.

The patch, the bit-identity dumps and the timings are in the worklog's §7 and its evidence
folder.
