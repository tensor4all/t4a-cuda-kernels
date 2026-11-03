# Slice — reduction throughput (tenferro-rs #2406)

Status: implemented, mechanism 1 only (2026-11-03; four independent plan reviews, then
measurement). What shipped, what was measured and rejected, and what remains are in
§9. Follows the measured
diagnosis in tenferro-rs #2406 and the `reduce-*` ownership row of
[slice-perf-remainder.md](slice-perf-remainder.md) §3.

The claim this slice supports is **native-before → native-after improvement of these
kernels**. It does not re-establish CubeCL parity and does not by itself close the end-to-end
PyTorch gap; both stay as reported elsewhere.

## 1. Question

tenferro-rs #2406 splits a full-array `f32` reduction at `n = 1 048 576` (81 µs against
PyTorch's 10.2 µs) into ~28 µs per-call floor, 9 µs split machinery, **21 µs of combine on
one warp**, and 13 µs of pass-1 streaming; at `n >= 4 194 304` pass-1 bandwidth dominates and
the extrema fold costs up to 2x the `sum` fold on identical traffic. What do the kernels own
here, and what is the smallest change that removes it?

## 2. Measured mechanisms (tenferro-rs #2406, A100 80GB PCIe)

| # | Mechanism | Evidence in the issue |
| --- | --- | --- |
| 1 | The combine pass is one 32-lane block: at `splits = 4096` a single warp folds 4096 partials, 128 serial folds per lane, ~0.30 µs per fold | E3: +19.6…21.1 µs at `n = 1 M`, identical pass-1 traffic and bytes |
| 2 | `MAX_SPLITS = 4096` caps the split count, so the combine cost stops falling as `n` grows | planner table: 4096 splits for every `n >= 1 M` |
| 3 | Pass 1 folds one element per lane per step with a loop-carried dependency and no ILP | E4/E5: 427 GB/s for `max` vs 1030 GB/s for `sum` at 16 M |
| 4 | The extrema fold is written as two data-dependent `if`s around `fmaxf`/`fminf` | E5: 2.0x `sum`'s time at 16 M |

Mechanism 4 is a **candidate**: source-level `if`s do not prove executed branches, and the
max/sum gap does not isolate the fold. §4.4 keeps it only if compiled code and timing show a
win.

## 3. Scope

| # | Mechanism | Files |
| --- | --- | --- |
| 1 | wide combine pass | `cuda/src/reduce.cu` (fold helper, `plane_reduce_body` **and** `plane_reduce_body_any`), `src/lib.rs` (`plan_reduction`) |
| 2 | split-policy cap for the wide path | `src/lib.rs` (`MAX_SPLITS` application site) |
| 3 | pass-1 ILP | `cuda/src/reduce.cu` (chunk-fold helper shared by `partials_body` **and** `partials_body_any`) |
| 4 | extrema fold (candidate) | `cuda/src/reduce.cu` (a reduction-only helper; the shared elementwise helper is untouched) |
| — | evidence | `examples/bench_kernels.rs` (host elapsed time), a native/native paired runner under `bench/` |

**Operations in scope for 1–4: `Sum`, `SumSquares`, `Max`, `Min`.** `Prod` keeps its current
split count, combine width and serial chain for every dtype: changing which partials a lane
multiplies (or how many partials exist) can turn an overflow-then-`0` product into
`Inf * 0 = NaN`, which is a classification change, not a rounding difference. Integer `Sum` regrouping stays exact modulo 2^32/2^64 through the existing wrapping helpers;
`SumSquares` has no integer route (`supported_dtypes` is F32/F64). Complex supports only `Sum`/`Prod`, so a complex cell exercises the `Sum` combine.

## 4. Design

### 4.1 Wide combine block (mechanism 1)

- Only the **combine pass**, only for the operations in §3, and only when the partials count
  is at least one full wide block. `splits < 64` keeps today's combine strategy and geometry
  exactly (the frozen 512/513/600 cases and every 33–63-partials case are untouched).
- Wide widths are **powers of two** `{64, 128, 256, 512}` (`MAX_BLOCK_WIDTH = 512`, `MAX_WARPS = 16`).
  Higher widths are unreachable: every cap is at most 4096 partials, so
  `next_power_of_two(splits / 8) <= 512`, and the static warp-result array is sized for 16
  entries rather than 32.
  A 96-thread block would have three warp results and no xor tree for them, so arbitrary
  multiples of 32 are not offered.
- The caller-visible `plane_width` stays restricted to `PLANE_WIDTHS` (8/16/32) and is still
  what `Unit` uses; the wide width is chosen and validated inside `plan_reduction`, so no
  caller — tenferro included — can select a wide `Unit` walk or an unvalidated width.
- Fold structure for a wide block (all barriers are device-local `__syncthreads()`; the
  dependency boundary forbids host-wide and stream-wide synchronization, not block barriers):
  1. warp-local xor reduction with `t4a_cuda_plane_mask(32)` and
     `for (offset = 1; offset < 32; offset <<= 1)` — the loop stops at the warp, because a
     full-mask `__shfl_xor_sync` does not communicate across warps;
  2. lane 0 of each of the `warps = width / 32` warps writes its warp result to a static
     `__shared__` array of `MAX_WARPS = 16` accumulators (≤ 256 B for `c64`; the runtime
     forwards no dynamic shared memory, and static shared memory is part of the compiled
     entry, so `Launch::shared_bytes` stays 0);
  3. `__syncthreads()`, then **lanes `0..warps`** of warp 0 xor-reduce exactly those entries
     with `for (offset = 1; offset < warps; offset <<= 1)` and mask
     `t4a_cuda_plane_mask(warps)` — not `warps - 1`, which names the wrong lanes — and lane 0
     writes the output. That is the only barrier the fold needs: the shared array is written
     once and never reused.
- The wide step lives only in the fold the combine-pass entries use. `partials_body` and
  `partials_body_any` (pass 1) keep a narrow-only fold whose generated code is what it was:
  the planner never widens pass 1, and having the wide step in the pass-1 kernel measured
  12.5% slower on a strided pass-1 case (`reduce-sum-f32-4096sq-axis1`, see §9.5). Both plane
  bodies (fast and general-rank) still take the same wide-capable fold, so the twins cannot
  drift.
- Width selection, for the operations in §3 only:
  `splits >= 64` → `width = clamp(next_power_of_two(splits / TARGET_FOLDS_PER_LANE), 64,
  512)` with `TARGET_FOLDS_PER_LANE = 8` (so 64 and 65–127 partials use 64, 4096 partials use 512).
  `splits < 64` → today's geometry.
  Exactly 64 partials is the narrowest wide case: it must select `Plane` at width 64 and not
  fall into `strategy_for(64, 64)`, which would return `Unit`.
  `check_strategy` keeps `reduce_len >= width`; a width the partials cannot fill is a
  planning error, never a silent narrowing.
- No caller change: `tenferro-gpu`'s `reduce_auto` plans with `plan_reduction` and launches
  the returned `Launch` values verbatim (grid, block, variant), and forwards no dynamic
  shared memory.

### 4.2 Split cap (mechanism 2)

`MAX_SPLITS` is re-tuned only by the campaign of §6, from the declared candidates
1024 / 2048 / 4096, applied only to the §3 operations, with mechanisms 1, 3 and 4 held fixed
and 4096 as the control state. Declared rule: choose the candidate with the lowest median
host time over the eight selection cells, subject to no case in the whole matrix being worse
than at the 4096 control by more than the §6 non-regression limit; otherwise keep 4096 and
record that the cap is not the binding cost. `MIN_CHUNK_PLANES` is unchanged.

### 4.3 Pass-1 ILP (mechanism 3)

- The inner chunk fold becomes one device helper used by **both** `partials_body` (rank ≤
  `MAX_RANK`) and `partials_body_any` (general rank). They are separate implementations
  today; the arithmetic walk moves into the helper (input pointer, base, axis stride,
  start/end, block width, chunk length) while each body keeps its own layout decode.
  `unit_reduce_body` is out of scope.
- The lane-to-address mapping is unchanged: lane `t` of a block of width `W` folds
  `start + t + k*W`. That **preserves the existing access pattern**; it does not make every
  load coalesced (a strided axis stays strided).
- When a lane's chunk holds at least `GROUP * W` elements, the helper reads `GROUP = 4`
  lane-strided elements per iteration (`start + (k+j)*W`, `j = 0..3`), combines that group as
  a small tree `(x0 ⊕ x1) ⊕ (x2 ⊕ x3)`, and folds the group result into **one** loop-carried
  accumulator. Per iteration that is four loads, two pair combines that are independent of
  each other, one combine that depends on both pairs, and one accumulator combine — the ILP
  the serial loop lacked, and no second loop-carried accumulator. The group result is itself
  a temporary partial value. Shorter chunks keep the plain serial loop.
- The first group seeds the accumulator with its own tree (`acc = group`), so the chunk's
  first four lane-strided elements really do form one group; the serial path keeps the
  existing first-element seed.
- Numerical policy, stated rather than proved. `Sum` and `SumSquares` become
  **reassociated**: the fastest, freely-reassociated reduction is the default this project
  chose for throughput, matching how the reference implementations sum. What follows is a
  policy with recorded consequences, not an accuracy theorem:
  - The topology is fixed for a fixed input, layout, plan and build, so results are
    deterministic and reproducible run to run (not across widths, revisions or targets).
  - Ordinary inputs keep the existing result-scaled tolerances against the `f64` CPU
    reference (`1e-5` F32/C32, `1e-12` otherwise); `SumSquares` keeps its own `1e-3` split
    tolerance and the existing `sum_of_squares_combines_as_a_plain_sum` fixture (1 000 F32
    values `(i − 480)/8`, F64 reference `1 308 023.4375`). No tolerance is loosened.
  - **F32 `Sum` fixture A — group-internal cancellation.** One lane's first group is
    `2^26, 1, −2^26, 1` (i.e. at `k = 0..3` of that lane's walk). The serial chain returns
    `1.0f32`; the grouped tree returns `+0.0f32`; the exact sum is `2`. The fixture asserts
    serial `1.0`, grouped `+0.0`, reference `2`, and is an explicit numerical-policy example,
    not a tolerance check. With F64 the same constants give `2` in both orders, so this is an
    F32 `Sum` fixture.
  - **F32 `Sum` fixture B — near overflow.** One lane's first group is `A, A, −A, −A` with
    `A = 2^127`. Both orders overflow; only the grouped order creates opposing infinite
    partials: serial `+Inf`, grouped `NaN` (classification, not payload — the reduction ABI
    promises only "is NaN"). Exact sum `0`. With F64 these constants give `0` in both orders,
    so this fixture is F32 as well. It is the recorded, accepted consequence of the
    reassociation policy and is asserted so it cannot change silently.
  - Arithmetic extremes beyond these two fixtures are not otherwise guaranteed to match the
    serial order; the plan claims nothing further about them.
  - `SumSquares` squares each input exactly once with `t4a_cuda_square` before any grouping,
    and its split pass 2 still uses plain `Sum`.
  - `Max`/`Min` group comparisons, which are associative for non-NaN inputs and keep "any NaN
    operand yields NaN"; NaN payloads are not promised by the reduction ABI.
  - `Prod` is not regrouped, and its partials count and combine geometry are unchanged.
- These fixtures live in `tests/gpu_reduce.rs` next to the existing split tests and state, in
  their comments, which reference value each one compares against.

### 4.4 Extrema fold (mechanism 4, candidate)

- The shared `t4a_cuda_nan_max`/`_min` helpers are **not** touched: `tests/gpu_elementwise.rs`
  pins that two NaN operands keep the **lhs payload** in both operand orders, and any
  architecture intrinsic that may return a different NaN would break that contract.
- Candidate instead: a reduction-only extrema helper in `cuda/src/reduce.cu`, used by the
  `Max`/`Min` folds, that preserves the current operand selection exactly — a NaN operand
  wins lhs-first (so the surviving bits are the same operand's), and the non-NaN path is the
  same `fmaxf`/`fminf` call, so the signed-zero choice is unchanged — with the two
  data-dependent `if`s replaced by a branchless select.
- Retention is decided by an **isolated** A/B, not by the aggregate: with mechanisms 1–3
  fixed, the extrema cells are measured helper-on versus helper-off (two builds of this
  branch differing only in that helper). It stays only if the extrema cells improve; a win
  that only appears in the aggregate is not evidence for it, because mechanisms 1–3 could
  produce it alone. Otherwise it is removed and the negative result recorded.
- `fmaximum`/`fminimum` are not an option: the CUDA 12.6 headers this repository compiles
  against declare no such function.

## 5. What must not change

- **ABI revision 11 → 12.** No symbol, signature or POD layout changes, and no new entry
  point or kernel argument, but `docs/abi/c-kernel-abi.md` requires a revision bump for a
  changed *launch contract*, and it currently specifies plane blocks of exactly 8/16/32
  threads with no shared memory. This slice changes both for the combine pass, so the
  revision, the revision list and the launch-geometry section are updated together, and the
  extrema/arithmetic sections record the reassociation policy of §4.3. `ABI_REVISION` is
  compile-time only and tenferro-rs never reads it; what invalidates a consumer's cached
  compiled module is the composed source, because `Module::compose` prepends the ABI header
  (which contains `T4A_CUDA_ABI_REVISION`) and the runtime keys its cache on the actual source
  text it compiles. Changing this crate therefore invalidates downstream artifacts by itself,
  and no consumer code change or extra cache mechanism is needed.
- The split plan stays two physical launches; `ReduceStrategy`, `ReductionPlan`,
  `split_policy` and `MIN_CHUNK_PLANES` keep their shapes and values, so tenferro-rs needs no
  code change for any of 1–4.
- Sub-warp widths (8/16/32) keep working and remain the only widths `Unit` may use; `Prod`
  keeps today's plan.
- No device allocation, no stream creation, no host-wide or stream-wide synchronization.

## 6. Evidence protocol (frozen before the first candidate timing)

**This is a warm, native/native, batch-throughput experiment.** It supersedes nothing that
the CubeCL parity contract states about CubeCL; it explicitly leaves unmeasured: cold
preparation, NVRTC/module load, per-operation round-trip latency, and any batch-16 observation.

**Harness work in this slice.**
`examples/bench_kernels.rs::measure` gains a host-clock measurement of its existing
100-iteration region (single barrier at the boundary, unchanged) reported next to the device
time, and a declared clock-settling warm-up (2000 iterations before the profiled region).
`bench/` gains a native/native paired runner: same harness source built against two library
revisions, release mode, one process per case per side per repetition, AB/BA order alternating
between repetitions, fresh campaign directory with no reuse, exclusive GPU (refuses to start
while another compute process is present), and the existing launch/allocation/transfer/wait
counters, input digests and output checks retained.

**Pairing.** Baseline `t4a-cuda-kernels@5019c713` (the revision tenferro-rs pins today) versus
candidate = this branch, 3 repetitions, plus one A/A repetition pair per side to report the
noise floor. A case with IQR/median > 0.05 is reported and the campaign is repeated as a whole.

**Matrix (frozen).**

| Group | Cases |
| --- | --- |
| Frozen reduce cases | `reduce-sum-f32-axis-511`, `-512`, `-513`, `-600`, `reduce-sum-f32-4096sq-axis0`, `-axis1`, `reduce-sum-f64-4096sq-axis0`, `reduce-sum-c64-4096sq-axis0`, `reduce-max-i32-4096sq-axis0`, `reduce-sum-f32-full-1000000`, `reduce-sum-f32-full-10000000` |
| Full reduce, new | `reduce-{sum,max,min}-{f32,f64}-full-{1048576,4194304,16777216}`, `reduce-sumsquares-{f32,f64}-full-{1048576,16777216}`, `reduce-prod-f32-full-{1048576,16777216}`, `reduce-prod-f64-full-1048576` |
| Wide combine, new | `reduce-sum-c64-full-1048576`, `reduce-sum-i32-full-1048576`, `reduce-max-f32-4x131072-axis1` (output_len 4, 512 partials), `reduce-max-f32-256x65536-axis1` (output_len 256) |
| Rank above `MAX_RANK`, new | `reduce-max-f32-rank9-2x2x2x2x2x2x2x2x131072-axis8` (the `*_any` path, same 512-partial combine) |
| Elementwise extrema (shared helper is untouched, so this is a guard) | `elementwise-maximum-f32-1000000`, `elementwise-minimum-f32-1000000` |

Case names are added to `examples/bench_kernels.rs` in this slice; the shapes above are the
frozen declaration.

**Primary metric.** Median host nanoseconds per operation of the eight selection cells
(`reduce_{sum,prod,max,min}_full_{f32,f64}_n1048576`, the cells tagged `2406` in
`data/instances/gpu_isolated_ops.json`).

**Mechanism 4 (isolated).** With 1–3 fixed, one extra A/B pair of builds that differ only in
the reduction extrema helper, over these twelve cells:
`reduce-{max,min}-{f32,f64}-full-{1048576,4194304,16777216}`. Statistic: the median of the
twelve cells' median host per-operation times, helper-off versus helper-on, under the same
repetition, AB/BA and invalid-campaign rules as the main campaign. Retention requires (a) the
median improves by at least 3 % and by at least the A/A spread this run reports, and (b) no
one of the twelve cells is worse than 1.03x. Elementwise-extrema timings are **not** part of
this decision; the shared elementwise helper is unchanged and its existing correctness tests
already guard it.

**Adoption rule (all three must hold).**
1. The median over the eight selection cells improves by at least 10 %.
2. The extrema cells alone (`reduce_max_full_f32/f64`, `reduce_min_full_f32/f64`) improve by
   at least 20 %.
3. No case in the whole matrix is worse than 1.03x the baseline median host time. A case
   between 1.03x and 1.10x must be removed by adjusting the slice; above 1.10x is a failed
   gate. Recording a residual is reporting, never a way to pass.

**Reporting.** Per case: median and IQR of host time, device time from the nsys kernel
summary, pass-1 versus pass-2 device time, kernel count, and the counters. Aggregate
control-versus-candidate numbers are reported always; per-mechanism attribution is claimed
only where the capture separates the passes (mechanism 1 is pass 2; mechanisms 3 and 4 are
both pass 1 and are reported as one pass-1 delta plus the instruction evidence).

**Identity.** GPU model, driver, toolkit, compute capability, NVRTC/compiler options, harness
commit and both library revisions travel with the campaign.

**End-to-end.** The same eight cells measured through tenferro (`direct`/`eager`/PyTorch/JAX,
one session, `GPU_BENCH_SKIP_LATEST=1`) travel in the follow-up tenferro-rs revision-bump
pull request, in a benchmark checkout of its own.

## 7. Acceptance

- `cargo fmt --check`, `cargo clippy -j 4 --all-targets -- -D warnings` (the repository's
  documented clippy gate; the `gpu-tests` feature set already had unrelated pre-existing
  clippy failures at the baseline, and the new gpu-test files are clean),
  `cargo test -j 4`, `cargo test -j 4 --features gpu-tests` (local A100) green, including the
  ABI/header/manifest agreement tests, `tests/gpu_reduce.rs` and `tests/gpu_dynamic_rank.rs`.
- ABI revision 12 recorded consistently in `src/lib.rs`, the C header, the manifest and
  `docs/abi/c-kernel-abi.md` (revision list + launch geometry + reassociation policy).
- The §6 adoption rule holds, with the campaign artifacts and the worklog table.
- Coverage added: rank-9 split through the `*_any` body; wide complex and wide integer
  combine; multi-output wide combine with distinct expected values; a wide combine whose only
  large values sit in the last partials (review-checklist §2: every participating lane must
  influence the result); the two recorded F32 `Sum` reassociation fixtures (§4.3 A and B);
  NaN and signed-zero operand orders for both extrema dtypes through the reduction path.
- Worklog with the campaign table, the instruction evidence for mechanism 4 (or its negative
  result), and the residuals of §8.

## 8. Not doing (residuals, reported not accepted)

- **No vector loads.** Lane `t` reads `start + t + k*W`, so "axis stride is one" does not make
  a lane's consecutive elements contiguous, and alignment depends on buffer pointer, decode
  base, offset and split start (a 1028-long axis gives chunk starts at 257). A vectorized
  pass 1 needs a different lane-to-element mapping and is its own slice.
- **No `Prod` regrouping** and no change to how many partials `Prod` produces.
- **No third split level / recursive combine** (changes the plan shape and needs caller
  coordination). Keeping the two-pass plan is this crate's own choice.
- **No new entry point** for a specialized combine kernel, and no struct-layout or
  signature change (the revision bump is for the launch contract only).
- **No change to `Unit`**, to layout/stride decoding, or to the shared elementwise extrema
  helper.
- The ~28 µs per-call floor and the split workspace's allocation zeroing are consumer/runtime
  responsibilities: fix 5 waits on an uninitialised allocation entry in `t4a-cuda-runtime`
  (tensor4all/t4a-gpu-backend#13); fix 6 (deferring the in-API device synchronize) changes
  public reduction semantics and is not made by this slice.

## 9. Outcome (measured)

### 9.1 Shipped: mechanism 1, the wide combine pass

`plan_reduction` gives a split reduction's combine pass a wide `Plane` block of 64-512 threads
for `SUM`, `SUM_SQUARES`, `MAX` and `MIN` with at least 64 partials, and the combine-pass fold
crosses warps through 16 static shared-memory accumulators. `PROD` and every sub-64-partials
case keep the baseline geometry. Paired native/native campaign, 21 cases x 3 repetitions,
A100 80GB PCIe, baseline `t4a-cuda-kernels@d44b57b` (`origin/main`), A/A noise floor 1.2%: the
eight cells tagged `2406` improve to a median of **0.549** of the baseline time, the four
extrema cells to **0.389-0.395** (for example `reduce-max-f32-full-1048576` 46.6 us -> 18.4 us),
and no case regresses beyond the declared 1.03x limit (worst 1.011). The campaign measured 21 of
the 43 declared cases; the measured-scope limitation is recorded in the worklog, which also
holds the full table: [../worklogs/2026-11-03-reduction-throughput.md](../worklogs/2026-11-03-reduction-throughput.md).

### 9.2 Withdrawn: mechanism 3 (pass-1 grouped walk)

Implemented, measured, and removed. The grouped walk regressed two cells that its own path
cannot explain, which is why it is not in this change:

| case | baseline | with the grouped walk |
| --- | ---: | ---: |
| `reduce-sum-f32-4096sq-axis1` (strided axis) | 194.3 us | 217.3 us (**1.13x**) |
| `reduce-sum-f32-axis-600` (2 partials, no group runs) | 16.0 us | 16.9 us (1.05x) |

Restricting the grouped walk to a contiguous axis did not move the strided row (1.131x), and
building the same change with the grouped path forced off still showed 1.115x on the strided
case: the regression is code generation in pass 1 caused by the branch's presence, not the
branch executing. Per the plan's non-regression rule it was withdrawn, and pass 1 is unchanged
in this change.

### 9.3 Not adopted: mechanism 4 (reduction-only extrema fold)

The isolated A/B the plan requires — mechanism 1 fixed, only the extrema helper differing,
6 extrema cells x 3 repetitions — gives a median ratio of **0.998** with a worst cell of
**1.005**: no measurable win. Recorded as a negative result; the shared elementwise helpers
were never touched.

### 9.4 The one regression the first implementation caused, and its fix

The first version of §4.1 put the wide step in the fold shared by every reduce entry. That
regressed a case the wide block never serves: `reduce-sum-f32-4096sq-axis1` (4096x4096, axis
1, 16 partials, pass-1 bound at 192 us) measured **1.125x** the baseline. Splitting the fold —
pass 1 keeps the narrow-only fold, only `t4a_cuda_reduce`/`_any` can widen — restores it to
**0.987x** with the wins unchanged, and is what shipped.

### 9.5 Residual

- **Pass-1 streaming is unchanged** and is now the largest remaining term for full
  reductions: by the issue's own accounting, 16M still costs ~112 us of pass-1 streaming for
  `max`, which this change does not touch. It needs the vectorized lane-to-element mapping
  of §8, a slice of its own.
- `MAX_SPLITS` stays 4096: with mechanism 1 in place the combine no longer scales with
  `splits`, so re-tuning it bought nothing measurable and is not part of this change.
- The per-call floor (~28 us) and the split workspace's zero fill (~9 us) remain
  consumer/runtime responsibilities (fixes 5 and 6 of the issue).
