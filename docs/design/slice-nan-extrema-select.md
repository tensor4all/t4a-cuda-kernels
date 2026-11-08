# Slice: NaN-propagating extrema as selects (tenferro-rs #2406, part 3)

## Problem

After #23, full `MAX`/`MIN` reductions are still slower than `SUM` on the same plan and the
same bytes. On an A100 (nsys, `3c8d82c`), the combine pass (`t4a_cuda_reduce`, 4096 partials,
512 threads, 8 folds per lane) takes 6.85 µs for `max` f32 at 1M, against 5.25 µs for `sum`.

## Cause

`t4a_cuda_nan_max`/`t4a_cuda_nan_min` (`cuda/include/t4a/abi.cuh`) were written with early
returns:

```cpp
if (lhs != lhs) return lhs;
if (rhs != rhs) return rhs;
return fmaxf(lhs, rhs);
```

When the accumulator is NaN, `rhs` is unused, so the compiler is free to sink the load that
produces `rhs` under a branch on the accumulator. In the strided loop of
`plane_reduce_body` (`cuda/src/reduce.cu`, the combine pass), every load then waits for the
previous fold: the SASS is `FSETP.NAN acc` → `@P BRA` → `LDG` → `FMNMX`, once per element.
`SUM` unrolls the same loop four times and issues four `LDG`s before its `FADD`s — stated
here as the contrast, not shown in the logs. The extrema loop itself is in both:
`sass-plane-loop.log` in a probe kernel, `sass-production.log` on the entries,
`t4a_cuda_reduce` at `reduce.cu:252-253` (double `MIN`) and `t4a_cuda_binary`.

## Change

The four functions compute every candidate and choose with selects:

```cpp
const float ordered = fmaxf(lhs, rhs);
const float right = rhs != rhs ? rhs : ordered;
return lhs != lhs ? lhs : right;
```

- Same result for every input: a NaN `lhs` wins, then a NaN `rhs`, then the ordered value,
  including NaN payloads and signed zeros (compared on the GPU against the old form over all
  pairs of 16 special values and 2^26 random bit patterns per type). That comparison was the
  2026-11-08 harness, which ignored every CUDA return value; the checked re-run is still
  outstanding (worklog, "Correctness").
- No change to the ABI, the plan, launch geometry, symbols, signatures, structs or the ABI
  revision. Integer extrema are untouched.
- Affected entries (`scripts/sass-diff.sh 3c8d82c`): `t4a_cuda_reduce`, `t4a_cuda_reduce_any`,
  `t4a_cuda_reduce_partials`, `t4a_cuda_reduce_partials_any` and `t4a_cuda_binary`
  (elementwise `MAX`/`MIN`). The other 89 entries are identical.

## Acceptance

- Bit identity of the new and old functions on the GPU, and the full GPU suite passing.
- Paired nsys and host time for the full-reduction cells, with `SUM` and `PROD` as controls,
  reported per case. A control may not be slower than 1.03x on host time per operation.
