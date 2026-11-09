# Slice: vector width and tile for the flat element-wise entries

Status: design, 2026-11-09. Driven by
[tenferro-rs#2529](https://github.com/tensor4all/tenferro-rs/issues/2529):
isolated CUDA elementwise operations are 2.5-15.7x behind PyTorch on an A100.

**This slice is not yet justified by a measurement.** The issue's own
root-cause section is a hypothesis. Its numbers fit a per-call host cost at
least as well: at `n = 1048576` the direct arm's gap to PyTorch is almost
constant (add 18.9 us, exp 17.7 us, tanh 18.7 us). An nsys comparison of kernel
time (this crate's entries against PyTorch's elementwise kernels) decides
whether the slice merges. If the kernel-time gap is above 2x, the slice
proceeds. The design and the CPU-side selection rule do not depend on that
result, so they are written first.

## 1. Current geometry

`flat_launch` (`src/lib.rs`) plans every flat entry as one thread per element:
`block = FLAT_BLOCK = 256` and `grid = ceil(len / 256)`. Each kernel body takes
its element through `t4a_cuda_tail_index` and issues one scalar load per
operand and one scalar store. `plan_elementwise`, `plan_unary`, `plan_compare`
and `plan_clamp` all route through it, and the ABI has no width or tile
parameter.

## 2. Scope

| Entry | This slice |
| --- | --- |
| `t4a_cuda_add/sub/mul/div` | vectorized |
| `t4a_cuda_unary` | vectorized (every dtype pair, including complex `ABS`) |
| `t4a_cuda_compare` | vectorized (T inputs, Bool output) |
| `t4a_cuda_clamp` | vectorized |
| `t4a_cuda_binary`, `t4a_cuda_select`, fill/copy, fusion emitter | **deferred** |

`t4a_cuda_binary` is deferred because of its scalar-side operands and its
`2 * len` component mode. `t4a_cuda_select` is not in the issue's rows. Fill and
copy are byte movement, and a consumer can serve compact fill and copy with
`cuMemset`/`cuMemcpy`. The fusion emitter has its own launch (`FusionPlan`).
Each deferred entry keeps its current geometry and ABI.

## 3. Selection rule (algorithm configuration, decided on the host)

Inputs: the byte size of each operand's element, the element count `len`, and
`alignment`. `alignment` is the largest power of two that divides every operand
address the launch will pass; `address_alignment` computes it.

```
width = 4
while width > 1:
    if widest_operand_size * width <= MAX_VECTOR_BYTES   # one 128-bit access per operand
       and alignment % (widest_operand_size * width) == 0 # every operand's pack is aligned
       and len >= width:                                  # at least one whole pack
        break
    width /= 2
```

- `MAX_VECTOR_BYTES = 16` is the widest native load/store on the GPUs this
  crate targets (`ld.global.v4.u32` / `v2.u64`), not a tuning value. The set
  `{1, 2, 4}` matches `TransposeConfig`'s vector widths.
- Resulting widths with aligned allocations: f32/i32/Bool → 4,
  f64/i64/c32 → 2, c64 → 1. Compare reads a T and writes a byte, so its widest
  operand is T. Unary `ABS` of complex reads c32/c64 and writes f32/f64, so its
  widest operand is the complex input.
- Because every size and the alignment are powers of two, an alignment that
  serves the widest operand's pack serves every narrower one.
- Misaligned addresses (a rebased view, an odd byte offset) or a short `len`
  degrade to width 1. Width 1 is exactly the old one-element-per-thread
  geometry and body. **It is the fallback path, not a separate code path.**
- Non-contiguous operands never reach these entries: strided work goes through
  `t4a_cuda_combined_strided` / movement entries, which this slice does not
  change.

## 4. Tile (from the element count)

With `vectors = len / width` and `tail = len % width`, each thread owns either
one whole pack or one tail element, so `threads = vectors + tail`.

- `block = min(FLAT_BLOCK, round_up(threads, WARP_SIZE))`: a tensor smaller than
  one block launches the warps it needs instead of a whole 256-thread block.
- `grid = ceil(threads / block)`, with the existing `MAX_GRID_X` limit.
- A tile is therefore `block * width` elements, and both factors come from
  `len` and the dtype. No new tuning constant is introduced. `WARP_SIZE = 32` is
  the hardware warp.

Each thread still does bounded work (at most `width <= 4` elements), so the
launch-domain rule of review-checklist section 2 holds. The grid scales with
`len`, and no thread loops over the tensor.

## 5. Kernel form

`abi.cuh` gains:

- `t4a_cuda_pack<T, W>`: `alignas(sizeof(T) * W) struct { T lane[W]; }`. A pack
  load is a single vector access.
- `t4a_cuda_flat_slot<W>(len, &first)`: returns `W` (a pack starting at
  `first = thread * W`), `1` (the tail element
  `first = vectors * W + (thread - vectors)`), or `0`. For `W = 1` this is
  `t4a_cuda_tail_index`.
- `t4a_cuda_flat_map1/2/3<W, TO, TA, ...>(…, op)`: load packs, apply `op`
  lane by lane, store one pack, or do the scalar tail.

Each entry body becomes one functor (the same scalar expression as before)
plus a runtime switch on `vector_width ∈ {1, 2, 4}`. A width the dtype cannot
take (`sizeof(T) * W > 16`) is not instantiated and, like an unknown dtype,
performs no work. The planner never produces it.

Every lane applies the same scalar expression the old body applied, in the same
type, with no reassociation. Two freedoms remain, both already in the ABI's
arithmetic semantics:

- **NaN payload.** `NaN + NaN` may return either operand's payload, because the
  compiler may swap a commutative operation's operands. As for the reductions,
  only "is NaN" is part of the contract. Observed on an A100: `Add F64` width 2
  returned the lhs payload where width 1 returned the rhs payload.
- **FMA contraction.** The ABI leaves contraction to the compiler, and the pack
  lanes and the scalar tail are separately compiled code. Complex
  multiplication (`a.re*b.re - a.im*b.im`) can therefore round differently
  between them. Observed: `Mul C32`, len 5, width 2, 1 ULP on the tail element.
  The difference is bounded normwise by
  `2 eps (|a.re b.re| + |a.im b.im|)` (and the imaginary analogue), not in ULPs
  of a result that may cancel.

Everything else (real add/sub/mul/div, complex add/sub, every unary pair
tested, compare, clamp, integers) was bit-identical across widths in the A100
run. The GPU test asserts exactly that split. Making complex multiplication
bit-stable would mean pinning its formula (for example `__fmul_rn` /
`__fadd_rn`, or an explicit `fma`). That is a revision-wide numerics decision
and is not part of this slice.

For a given input, length and alignment, the width and so the bits are fixed, so
tenferro's reproducibility promise (same build, same input) is unaffected. An
owned tensor's allocation base is always aligned.

## 6. ABI

The vectorized entries take a trailing `int vector_width` after `len`.
`ABI_REVISION` 13 → 14. The planners for these entries take
`alignment: u64` and return `FlatLaunch { launch, vector_width }`, so a caller
cannot launch with a width the planner did not choose. The caller passes
`vector_width` verbatim as the trailing argument.

`alignment` is a caller contract like the pointers themselves: it must describe
the addresses actually passed. A caller that does not know its addresses passes
`1` and gets the old geometry.

## 7. Tests

CPU (`tests/planning.rs`, red before the rule exists):

- the width per dtype for aligned addresses; the degradation by alignment
  (1, 2, 4, 8, 16 bytes) and by `len < width`; compare and unary-`ABS` use the
  widest operand;
- the tile: `threads = len / w + len % w`, block warp-rounded and capped,
  grid covering `threads`; a prime length (1_000_003) and `len = 1`;
- `address_alignment` on mixed addresses and on address zero;
- width 1 (`alignment = 1`) reproduces the old plan for large lengths.

GPU (`tests/gpu_elementwise.rs`, `--features gpu-tests`, A100):

- for every vectorized entry and dtype, widths 1/2/4 (where the dtype allows)
  over lengths `0, 1, 2, 3, 4, 5, 7, 65_537, 1_000_003`. Each width must leave
  the guard of a sentinel-filled destination untouched and match width 1 under
  the contract of section 5. NaN payloads, signed zero, infinities, subnormals
  and integer extremes are among the inputs. `Add F64` is also checked against
  a CPU reference;
- a misaligned operand (base + one element) plans width 1 and matches.

## 8. Deferred / out of scope

- `t4a_cuda_binary`, `t4a_cuda_select`, fusion emitter widths.
- Grid-stride or multi-pack-per-thread unrolling: only with a measurement
  showing it is needed.
- `docs/benchmarks/kernel-parity/` is the recorded ABI-13 parity harness (its
  own crate against a pinned tenferro checkout); it stays at the revision its
  recorded runs used and is not updated here.
- tenferro-rs consumes the new planner signature after this crate's release
  (repin, pass `alignment` for allocation bases, append `vector_width`).
