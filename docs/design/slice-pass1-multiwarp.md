# Split pass 1 in multi-warp blocks (ABI revision 14)

Issue: tensor4all/t4a-cuda-kernels#30, item 3. Items 1–2 (combine load-ahead) are #29
(`docs/design/slice-combine-load-ahead.md`); this slice is stacked on it.

## Problem

Split pass 1 ran one 32-thread block per `(output, split)` cube. A full-array reduction of
`2^20` elements has 4096 cubes. That is about 38 blocks per SM on the A100's 108 SMs, against
the limit of 32 resident blocks per SM, so the last cubes wait for a second wave of blocks.

## Change

- **Geometry.** The planner returns `block = (32 * PARTIALS_PLANES_PER_BLOCK, 1, 1)` and
  `grid = ceil(cubes / PARTIALS_PLANES_PER_BLOCK)` only when all three hold (see "Planner
  rule" below):
  - `plane_width == 32`;
  - `cubes = output_len * splits <= PARTIALS_MULTIWARP_MAX_CUBES` (`2 * MAX_SPLITS` = 8192);
  - `chunk_len <= PARTIALS_MULTIWARP_MAX_CHUNK` (1024).

  Otherwise it returns the revision-13 geometry `block = (plane_width, 1, 1)`,
  `grid = cubes`. 8- and 16-lane planes always keep one plane per block.
- **Kernel.** `t4a_cuda_partials_plane_of_thread()` maps thread `i` of block `b` to lane
  `i % 32` of cube `b * (blockDim.x / 32) + i / 32`, with 32-bit shifts (the planner bounds
  the cube count by the grid limit). When `blockDim.x <= 32` it returns
  `cube = blockIdx.x` and `lane = threadIdx.x`, which is revision 13. Both walks (scalar and
  vector) take that `lane` instead of reading `threadIdx.x`.
- **One-dimensional on purpose.** tenferro launches `LaunchConfig::new(pass1.grid[0],
  pass1.block[0])` (`crates/tenferro-gpu/src/cuda/structural.rs` on `main`). A
  `(32, W, 1)` block would be launched as `(32, 1, 1)` there, and 3/4 of the partials would
  silently never be written.

## Why the partials are bit-identical

- Each cube is still walked by exactly one warp.
- Lane `t` of that warp folds the same elements in the same order as lane `t` of the old
  one-warp block, and the warp fold is the same xor tree with the full mask.
- **No `__syncthreads`.** Warps of a block never exchange data: the pass-1 plane fold is
  shuffles only, and pass 1 uses no shared memory.
- **Tail.** A warp whose cube is `>= output_len * splits` fails the existing
  `output_index >= output_len` test. That test is uniform across the warp, so the whole warp
  returns before any shuffle.
- **Old launches.** A revision-13 launch (`block = 32`, `grid = cubes`) still produces the
  same partials with the new kernel.

## ABI

The pass-1 *launch contract* changes, so this is revision 14
(`docs/abi/c-kernel-abi.md`, "Split pass-1 block geometry"). No symbol, signature, struct or
numerical topology changes. The cube-count bound is kept, so the planner refuses exactly the
inputs revision 13 refused.

## Verification

- **Golden.** `tests/gpu_reduce_golden.rs` checks against the `ab7395c` capture
  (`tests/data/reduce-golden.tsv`):
  - 1332 planned two-pass reductions;
  - fast and `_any` entries;
  - f32, f64, c32, c64;
  - sum, prod, max, min, sum_squares;
  - widths 8, 16 and 32;
  - ragged and multi-output shapes, including a block that straddles two outputs (3 × 390
    cubes);
  - NaN-payload, ±inf, signed-zero, -0 and overflow inputs.
- **What is compared.** The pass-1 partials hash, every final byte, and guard bytes after
  both destinations.
- **SASS.** `scripts/sass-diff.sh 4ea91e6` reports only `t4a_cuda_reduce_partials` and
  `t4a_cuda_reduce_partials_any` as differing; the other 92 entries are identical.

## Planner rule: few, short cubes

The issue asked to keep multi-warp blocks to few cubes and short chunks. The planner does that:
at most `2 * MAX_SPLITS` cubes, chunks of at most 1024 elements (`src/lib.rs`,
`plan_reduce_partials`; `tests/planning.rs` pins both boundaries), and
`PARTIALS_PLANES_PER_BLOCK = 4`.

What the measurements say about the two limits (nsys device µs per launch, A100, CUDA 12.6,
median of per-round medians; tables in `docs/worklogs/evidence/2026-11-10-pass1-multiwarp/`):

- **Long chunks.** On the 10M and 16M full-array reductions (chunks of 2441 to 4096
  elements), unconditional 4-warp blocks were 1 to 3 % slower. The clock64 probe below
  shows those warps run longer (median 51,194 clk against 41,929 clk), not that they hold
  slots after finishing.
- **Many cubes.** The earlier "+4 to +8 % on the 65,536-cube `4096sq-axis0` cases" was
  measured with a kernel whose plane mapping used a 64-bit divide. That cost was paid by
  one-warp launches too: f32 pass 1 took 64.70 µs at a029e80 against 57.79 µs at #29, with
  the same grid of 65536 × 32. With the 32-bit mapping (f21234e), an experiment without the
  cube limit makes pass 1 *faster* there:

  | case | base | f21234e | no cube limit |
  |---|---:|---:|---:|
  | f32 axis0 | 60.10 | 59.76 | 57.42 |
  | i32 axis0 | 60.86 | 60.02 | 57.15 |
  | f64 axis0 | 77.41 | 78.75 | 79.54 |
  | c64 axis0 | 153.68 | 153.97 | 152.66 |

  So the cube limit is a conservative choice, not a measured need. Lifting it is a possible
  follow-up and would need its own control gate.

## Measurement

Device µs (pass 1 + combine) for `ab7395c` (base), #29 (`4ea91e6`) and this branch
(`f21234e`); 3 rounds with rotated arm order (`nsys-final2.md`):

| case | base | #29 | this | this/base |
|---|---:|---:|---:|---:|
| sum f32 full 1048576 | 13.22 | 12.90 | 11.52 | 0.871 |
| prod f32 full 1048576 | 20.41 | 17.12 | 15.42 | 0.756 |
| max f32 full 1048576 | 14.24 | 13.47 | 11.97 | 0.841 |
| min f32 full 1048576 | 13.79 | 13.44 | 12.38 | 0.898 |
| sum f64 full 1048576 | 13.63 | 13.09 | 12.22 | 0.897 |
| prod f64 full 1048576 | 20.54 | 18.91 | 17.92 | 0.872 |
| max f64 full 1048576 | 14.69 | 13.86 | 13.41 | 0.913 |
| min f64 full 1048576 | 14.40 | 13.92 | 13.38 | 0.929 |
| sum_squares f32 full 1048576 | 12.45 | 12.13 | 11.36 | 0.912 |
| sum f32 full 1000000 | 12.42 | 12.67 | 11.23 | 0.904 |
| sum f32 axis 512 | 13.92 | 14.21 | 13.22 | 0.950 |
| sum f32 axis 513 | 14.24 | 14.79 | 13.70 | 0.962 |
| sum f32 axis 600 | 13.92 | 14.30 | 13.31 | 0.956 |
| max f32 4x131072 axis1 | 12.13 | 10.75 | 10.78 | 0.889 |
| sum f32 4096sq axis0 | 65.06 | 62.94 | 65.18 | 1.002 |
| sum f32 4096sq axis1 | 194.88 | 192.48 | 192.93 | 0.990 |
| sum f64 4096sq axis0 | 82.69 | 85.32 | 84.61 | 1.023 |
| sum c64 4096sq axis0 | 160.51 | 159.97 | 160.48 | 1.000 |
| max i32 4096sq axis0 | 66.72 | 63.84 | 64.74 | 0.970 |
| sum f32 full 10000000 | 35.23 | 35.62 | 35.42 | 1.005 |
| sum f32 full 16777216 | 50.46 | 50.11 | 49.98 | 0.990 |
| sum f64 full 16777216 | 88.45 | 88.32 | 88.74 | 1.003 |
| prod f32 full 16777216 | 72.70 | 69.25 | 69.34 | 0.954 |
| prod f64 full 16777216 | 102.31 | 100.16 | 101.06 | 0.988 |
| max f32 full 16777216 | 52.03 | 51.04 | 51.23 | 0.985 |
| min f64 full 16777216 | 89.54 | 91.46 | 89.95 | 1.005 |

Every case is at or below the 1.03 control gate. `f64 4096sq axis0` (1.023) does not use
multi-warp blocks; #29 alone is 1.032 there.

End to end (host µs per call, paired campaign `ab7395c` against `f21234e`, 3 warm-up + 15
repetitions alternating AB/BA, measured processes on CPUs 24-31, which is one L3 domain,
because CPUs 0-7 were busy with another user's work; `campaign/table.md`). Selected rows:

| case | base µs | this µs | ratio |
|---|---:|---:|---:|
| sum f32 full 1048576 | 14.734 | 13.044 | 0.885 |
| prod f32 full 1048576 | 21.789 | 17.077 | 0.784 |
| sum f64 full 1048576 | 15.321 | 13.154 | 0.859 |
| sum f32 axis 513 | 15.453 | 14.789 | 0.957 |
| sum f32 4096sq axis0 | 65.138 | 66.162 | 1.016 |
| sum f32 4096sq axis1 | 193.568 | 195.469 | 1.010 |
| max i32 4096sq axis0 | 65.202 | 64.488 | 0.989 |
| sum f64 4096sq axis0 | 93.387 | 94.476 | 1.012 |
| sum f32 full 10000000 | 39.919 | 40.633 | 1.018 |

All 29 cases are at or below 1.03; the highest is 1.018.

## Occupancy and the probe

Pass 1 uses 54 registers per thread, so an SM holds at most 36 of its warps. Without a
block limit, 4-warp blocks raise the resident warps per SM from 32 (the 32-block limit) to
36. `ncu` could not be run: the driver has `RmProfilingAdminOnly: 1`, so the counters
(`sm__warps_active` and others) need admin rights we do not have.

As a substitute, a copy of the kernel records `clock64()` at start and end of every warp,
and `%smid` (`probe/`; the clocks are SM-local, so only comparisons within one SM are made).
It ran 20 launches per case:

| case | planes | SM span (clk, median) | mean active warps/SM | slot time held by finished warps | span with < half-peak warps active |
|---|---:|---:|---:|---:|---:|
| full 1M (4096 cubes, chunk 256) | 1 | 6658 | 15.7 | 0.0 % | 48.6 % |
| full 1M | 4 | 5014 | 24.0 | 2.4 % | 35.4 % |
| 4096sq axis0 (65536 cubes, chunk 256) | 1 | 87276 | 22.6 | 0.0 % | 5.7 % |
| 4096sq axis0 | 4 | 82272 | 25.4 | 3.8 % | 4.2 % |
| full 16M (4096 cubes, chunk 4096) | 1 | 51358 | 30.2 | 0.0 % | 3.7 % |
| full 16M | 4 | 55720 | 34.9 | 0.2 % | 2.3 % |

On the 1M case, one-warp blocks leave the SM below half its peak warps for about half the
span (the second-wave tail), and 4-warp blocks shorten the span by 25 %. The hypothesis that
4-warp blocks lose on many cubes by holding slots after their warps finish is **not
confirmed**: those slots are 3.8 % of the slot time, and the probe's 4-warp run is shorter.
The probe kernel is the `a029e80` copy (64-bit divide), in both arms.
