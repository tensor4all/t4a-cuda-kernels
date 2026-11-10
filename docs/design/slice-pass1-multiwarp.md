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

Unconditional 4-warp blocks were faster on the full-array 1M-class problems (4096 cubes of
256 elements), but slower where one-warp blocks are already refilled warp by warp:

- many cubes: the 65,536-cube `reduce-*-4096sq-axis0` cases (+4 to +8 % pass-1 device time);
- long chunks: the 10M and 16M full-array reductions (chunks longer than 1024 elements,
  +1 to +3 %), which are bandwidth-bound walks.

So the planner keeps multi-warp blocks for at most `2 * MAX_SPLITS` cubes with chunks of at
most 1024 elements (`src/lib.rs`, `plan_reduce_partials`; `tests/planning.rs` pins the
boundaries). `PARTIALS_PLANES_PER_BLOCK` is 4.

Cases outside the rule still run the new kernel with one-warp blocks, so they still pay for
the plane mapping. A 64-bit divide there cost about 12 % of pass 1 on the 4096sq axis-0
cases; `t4a_cuda_partials_plane_of_thread()` therefore uses 32-bit shifts only.
