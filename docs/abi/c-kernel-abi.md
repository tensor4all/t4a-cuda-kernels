# C kernel ABI (revision 3)

Every entry is an `extern "C" __global__` CUDA function in `cuda/src/`. A caller
loads the compiled module into its own context, looks the symbol up by the name
below, supplies its own stream, device buffers and completion management, and
launches with the geometry from the Rust facade (`t4a_cuda_kernels::plan_*`) or
the equivalent host arithmetic.

No entry takes a stream, allocates memory, frees memory, synchronizes, creates
runtime objects or allocates scratch. Nothing in this ABI returns a status
value: metadata errors are refused by the facade before planning and are caller
UB at the raw entry, whose kernels keep only the bounds guards of the ported
bodies. Launch success is not proof of asynchronous completion or numerical
validity.

Header: [`cuda/include/t4a/abi.cuh`](../../cuda/include/t4a/abi.cuh). It includes
nothing, so the same text compiles under NVCC and NVRTC.

## Revision

`T4A_CUDA_ABI_REVISION == 3`. Revision 2 added `t4a_cuda_axis_values` and the
diagonal, windowing and validation-status entries; revision 3 adds the reduction
entries. Nothing in revisions 1-2 changed. Any change to a struct layout, an entry signature, an element type
mapping or a launch contract requires a further revision bump and an entry in the
compilation/artifact key of any consumer that caches compiled code.

## Types

| C type | Size | Align | Notes |
| --- | --- | --- | --- |
| `t4a_cuda_layout` | 144 | 8 | by-value launch argument |
| `t4a_cuda_transpose_config` | 16 | 4 | by-value launch argument |
| `t4a_cuda_axis_values` | 72 | 8 | per-axis runtime values (revision 2) |
| `t4a_cuda_c32` | 8 | 4 | interleaved `re, im`; matches `num_complex::Complex32`, not `float2` (align 8) |
| `t4a_cuda_c64` | 16 | 8 | interleaved `re, im`; matches `num_complex::Complex64`, not `double2` (align 16) |

```c
typedef struct {              /* offsets: rank 0, reserved 4, offset 8, extents 16, strides 80 */
  int rank;                   /* 0..=8; 0 is a scalar view */
  int reserved;               /* must be 0 */
  long long offset;          /* element offset, may be negative */
  long long extents[8];
  long long strides[8];
} t4a_cuda_layout;

typedef struct {
  int tile;                   /* 8 | 16 | 32 */
  int block_rows;             /* 4 | 8 | 16 */
  int padding;                /* 0 | 1 */
  int vector_width;           /* 1 | 2 | 4, tile % vector_width == 0 */
} t4a_cuda_transpose_config;

typedef struct {              /* offsets: rank 0, reserved 4, values 8 */
  int rank;                   /* must equal the operation rank */
  int reserved;               /* must be 0 */
  long long values[8];
} t4a_cuda_axis_values;
```

`t4a_cuda_kernels::CLayout` / `CTransposeConfig` are the `#[repr(C)]` Rust
mirrors; both sides assert size, alignment and field offsets.

Element types:

| Tag | C type | Bytes | Alignment |
| --- | --- | --- | --- |
| `T4A_CUDA_F32` = 0 | `float` | 4 | 4 |
| `T4A_CUDA_F64` = 1 | `double` | 8 | 8 |
| `T4A_CUDA_I32` = 2 | `int` | 4 | 4 |
| `T4A_CUDA_I64` = 3 | `long long` | 8 | 8 |
| `T4A_CUDA_BOOL` = 4 | `unsigned char` | 1 | 1 |
| `T4A_CUDA_C32` = 5 | `t4a_cuda_c32` | 8 | 4 |
| `T4A_CUDA_C64` = 6 | `t4a_cuda_c64` | 16 | 8 |

The dtype tag is a runtime argument; one entry serves every supported dtype.
Passing an unsupported or unknown tag performs no work.

## Logical indexing

Axis 0 is the fastest-varying logical axis. A flat logical index `f` decodes as
`coordinate = f % extent`, `f /= extent`, from axis 0 upward, and the element
address is `offset + sum(coordinate * stride)`. Strides and offsets are signed
element counts. Stride zero is a broadcast axis.

For all whole-view entries, `len` equals the extent product, including the
empty (`len == 0`) and rank-0 (`len == 1`) cases. The two layouts of
`t4a_cuda_strided_to_strided` must have identical extents.

## Entries

| Symbol | Arguments | Semantics |
| --- | --- | --- |
| `t4a_cuda_fill_zero` | `(void* out, int dtype, long long len)` | `out[i] = 0` for `i < len` |
| `t4a_cuda_fill_zero_view` | `(void* out, int dtype, t4a_cuda_layout dst, long long len)` | zero every logical coordinate of `dst` |
| `t4a_cuda_copy_exact` | `(void* dst, const void* src, int dtype, long long len)` | bit-preserving copy of `len` elements |
| `t4a_cuda_materialize_strided` | `(void* dst, const void* src, int dtype, t4a_cuda_layout src, long long len)` | gather the strided source into a compact destination |
| `t4a_cuda_contiguous_to_view` | `(void* dst, const void* src, int dtype, t4a_cuda_layout dst, long long len)` | scatter a compact source into the strided destination |
| `t4a_cuda_strided_to_strided` | `(void* dst, const void* src, int dtype, t4a_cuda_layout src, t4a_cuda_layout dst, long long len)` | copy between two layouts with the same extents |
| `t4a_cuda_tiled_transpose` | `(void* dst, const void* src, int dtype, long long src_offset, long long batch_stride, long long dst_fast_extent, long long src_fast_extent, t4a_cuda_transpose_config config)` | see below |
| `t4a_cuda_add` / `t4a_cuda_sub` / `t4a_cuda_mul` | `(void* out, const void* lhs, const void* rhs, int dtype, long long len)` | element-wise, `out` distinct from both inputs |
| `t4a_cuda_div` | same | F32/F64 only |

Supported dtypes are `DType::ALL` for the movement entries; add/sub/mul serve
F32/F64/I32/I64/C32/C64 (no Bool, which the baseline dispatch does not have);
div serves F32/F64 only.

### Tiled transpose

`t4a_cuda_tiled_transpose` reads a compact row-major `(M, N)` matrix per batch
and writes its transpose to a compact destination at element offset zero:

```
M = dst_fast_extent, N = src_fast_extent, S = M * N
dst[b*S + i + j*M] = src[src_offset + b*S + i*N + j]
```

This is the same coordinate map in column-major storage, not a relabelled
`(N, M)` transpose, matching `structural.rs:197-216`. `src_offset` must be
non-negative and `batch_stride` must equal `M*N` in revision 1. The kernel uses
dynamic shared memory: pass `tile * (tile + padding) * element_size` bytes.
Shared-memory padding is deliberate: it removes bank conflicts between the
transposed read and write patterns.

### Arithmetic semantics

- Float and complex arithmetic are IEEE; no fast-math, no flush-to-zero. FMA
  contraction is the compiler default, so results match a CPU reference within
  tolerance, not bit-for-bit.
- Integer add/sub/mul wrap (two's-complement via unsigned arithmetic). No other
  integer operation is exported.
- Bool is a byte holding 0 or 1; only movement entries accept it.
- Division is float-only; complex division must port the scaled Baudin–Smith
  form and is not part of revision 1.

### Revision 2: diagonal, windowing and validation entries

| Symbol | Arguments | Semantics |
| --- | --- | --- |
| `t4a_cuda_extract_diagonal` | `(void* out, const void* in, int dtype, t4a_cuda_layout out, t4a_cuda_layout in, int axis_a, int axis_b, long long len)` | gather the diagonal over `axis_a`/`axis_b` |
| `t4a_cuda_embed_diagonal_copy` | `(void* out, const void* in, int dtype, t4a_cuda_layout out, t4a_cuda_layout in, int axis_a, int axis_b, long long len)` | write `in` onto the diagonal of a pre-zeroed `out` |
| `t4a_cuda_tril` / `t4a_cuda_triu` | `(void* out, const void* in, int dtype, long long k, long long rows, long long cols, long long len)` | triangle mask on axes 0/1, batched over higher axes |
| `t4a_cuda_slice` | `(void* out, const void* in, int dtype, t4a_cuda_layout out, t4a_cuda_layout in, t4a_cuda_axis_values starts, t4a_cuda_axis_values steps, long long len)` | static starts and steps |
| `t4a_cuda_dynamic_slice` | `(void* out, const void* in, const void* starts, int dtype, int index_dtype, t4a_cuda_layout out, t4a_cuda_layout in, long long len)` | device-resident starts, window = output extents |
| `t4a_cuda_pad` | `(void* out, const void* in, int dtype, t4a_cuda_layout out, t4a_cuda_layout in, t4a_cuda_axis_values edge_low, t4a_cuda_axis_values edge_high, t4a_cuda_axis_values interior, long long len)` | zero-padded copy; `edge_high` only defines the output extent |
| `t4a_cuda_init_validation` | `(unsigned int* flag, void* value, int dtype)` | `flag = UINT32_MAX`, `value = 0`; one thread |
| `t4a_cuda_validate_float_indices` | `(const void* indices, unsigned int* flag, int dtype, double max_exact_integer, long long len)` | NaN, Inf, non-integral and out-of-range positions; lowest wins |
| `t4a_cuda_extract_invalid_float_index` | `(const void* indices, const unsigned int* flag, void* value, int dtype)` | copy the offending index value; one thread |
| `t4a_cuda_validate_real_cast` | `(const void* input, unsigned int* flag, int dtype, double min, double max, long long stride, int max_inclusive, long long len)` | sample every `stride`-th element; NaN, Inf and range violations; the flag holds the **sampling position** (element index / stride) |
| `t4a_cuda_extract_invalid_real_cast` | `(const void* input, const unsigned int* flag, void* value, int dtype, long long stride)` | copy the offending sampled value; one thread |

Contract notes:

- **Compactness.** The baseline stores at the flat position while decoding
  coordinates from a layout, so it requires compact operands there. In revision 2
  `out` is compact for extract-diagonal, tril/triu, slice, dynamic-slice and pad;
  `in` is compact for embed-diagonal and tril/triu. The strided operand keeps
  signed strides.
- **Diagonal extraction corrects a baseline defect.** The baseline consumes the
  diagonal output coordinate where a remaining non-diagonal coordinate is
  required: it never skips the diagonal output axis while walking the other
  output coordinates, so any rank >= 3 layout with at least two remaining output
  axes reads the wrong input coordinate (input `[2,3,4]` with axes `(0,2)` and
  output coordinate `(0,1)` reads input `(0,0,0)` instead of `(0,1,0)`). This ABI
  implements the intended map; the diagonal output position is
  `axis_a - (axis_b < axis_a)` and is derived, not passed.
- **Embedding** uses `axis_a` in the input rank and `axis_b` as an insertion
  position in `[0, in.rank]`, so `axis_a == axis_b` is valid. Only diagonal
  positions are written: the caller must zero-fill `out` first, and the two
  launches are ordered on the caller's stream.
- **tril/triu** accept any rank >= 2: `rows` and `cols` are the extents of axes
  0 and 1, and `len` is a whole number of `rows * cols` matrices. The whole
  output is written. Comparisons use `col - row` so that extreme `k` values
  (`i64::MIN`, `i64::MAX`) are well defined instead of overflowing.
- **slice** requires `0 <= starts[a] <= in.extents[a]`, `steps[a] >= 1`, and
  `out.extents[a] <= ceil((in.extents[a] - starts[a]) / steps[a])`. The output
  extent is the slice length, which lets a caller stop before the input edge (the
  baseline expresses the same thing with an explicit `limit`).
- **dynamic_slice** requires `out.extents[a] <= in.extents[a]` and a
  device-resident starts vector of `in.rank` values. Starts must already be
  validated (integral and representable), matching the baseline, which runs the
  validator first; clamping is `start <= 0 -> 0`, else
  `min(trunc(start), extent - window)`.
- **pad** requires `interior >= 0` and
  `out.extents[a] == edge_low[a] + edge_high[a] + (in.extents[a] == 0 ? 0 :
  (in.extents[a]-1)*(interior[a]+1) + 1)`. Negative edge lows keep the baseline's
  unsigned-magnitude treatment; positions outside the input write zero.
- **Validation flags** are one 4-byte, 4-byte-aligned `uint32_t` word holding the
  lowest invalid position, initialized to `UINT32_MAX`. The value slot is
  `element_size` bytes with the dtype's alignment. Both must be distinct
  allocations, must not overlap the validated input, and are written only by
  `init_validation` / the validation entry; the caller orders
  initialize -> validate -> extract on one stream and decides when to read the
  value back with its own synchronization. The flag holds a *sample or element
  position*, so the domain must satisfy `samples <= UINT32_MAX`: for
  `validate_float_indices` the element count, and for `validate_real_cast` the
  sample count `ceil(len / stride)`; the planner rejects larger domains. The
  extracted value is the offending source value, not its position.
- **Validation predicates** reject NaN and +/-Infinity explicitly, then apply the
  range test (and, for float indices, the integrality test). Infinite bounds do
  not weaken this: an infinite value always reports invalid, as in the baseline.

### Revision 3: reductions

```c
typedef enum {
  T4A_CUDA_REDUCE_SUM = 0, T4A_CUDA_REDUCE_PROD = 1, T4A_CUDA_REDUCE_SUM_SQUARES = 2,
  T4A_CUDA_REDUCE_MAX = 3, T4A_CUDA_REDUCE_MIN = 4,
} t4a_cuda_reduce_op;

typedef enum {
  T4A_CUDA_REDUCE_UNIT = 0, T4A_CUDA_REDUCE_PLANE = 1,
} t4a_cuda_reduce_strategy;
```

| Symbol | Arguments |
| --- | --- |
| `t4a_cuda_reduce` | `(void* out, const void* in, int dtype, int op, int strategy, t4a_cuda_layout out, t4a_cuda_layout in, int axis, long long output_len)` |
| `t4a_cuda_reduce_partials` | `(void* partials, const void* in, int dtype, int op, t4a_cuda_layout partials, t4a_cuda_layout in, int axis, long long output_len, long long splits, long long chunk_len)` |

Semantics:

- **Keepdims.** `out` has the input rank with `out.extents[axis] == 1`;
  `output_len == out.element_count()`. The reduced axis is one axis and
  `in.extents[axis] >= 1`.
- **First-element seeding.** The accumulator starts from the axis' first element
  (or its square for `SUM_SQUARES`), so a length-1 axis returns `x` for sum, prod,
  max and min (`SUM([-0.0])` keeps its sign, a complex `PROD` never acquires a NaN
  component from a complex identity) and `x * x` for sum-of-squares.
- **Unit** (one thread per output) is only valid when
  `reduce_len <= plane_width`; **plane** (one block per output, lanes striding by
  `plane_width`) only when `reduce_len >= plane_width`. The planner enforces both
  bounds; a unit reduction never walks a long axis.
- **Plane collectives** use `__shfl_xor_sync` with ascending offsets
  `1, 2, 4, ... < plane_width` and a mask of exactly the participating lanes.
  The block is exactly `plane_width` (8, 16 or 32), every lane participates, and
  lane 0 writes.
- **Split pass 1** runs one block per `(output, split)` with
  `cube = output_index * splits + split`, reduces the half-open axis range
  `[split*chunk_len, end)` with `end = axis_len` for the last split and
  `start + chunk_len` otherwise, and writes one partial at reduce-axis
  coordinate `split`. The partials layout keeps the output's non-axis extents and
  sets the axis extent to `splits`; its strides are the caller's own. A chunk
  must hold at least one plane (`chunk_len >= plane_width`) and must be the floor
  split (`chunk_len == reduce_len / splits`), so every lane seeds inside its own
  slice; the last chunk absorbs `reduce_len % splits`.
- **Pass 2** is `t4a_cuda_reduce` over the partials, with `SUM` when pass 1 was
  `SUM_SQUARES` and the same op otherwise, and is never split again.
- **Arithmetic.** Integer `SUM`/`PROD` wrap. `SUM_SQUARES` squares each element
  through `fma(v, v, 0)` so the square rounds before the addition and cannot be
  contracted into it. Complex `SUM`/`PROD` use the revision-1 element-wise
  complex forms.
- **NaN.** Float `MAX`/`MIN` propagate NaN (a NaN operand wins) both inside a
  lane and across the plane, so a NaN anywhere in the axis yields NaN. The NaN
  payload is not part of the contract; only "is NaN" is.
- **Launch geometry.** Unit: `grid = ceil(output_len/plane_width)`. Plane:
  `grid = output_len`. Pass 1: `grid = output_len * splits`, `block =
  plane_width`. No shared memory is used.

Supported dtypes: `SUM`/`PROD` F32/F64/I32/I64/C32/C64, `SUM_SQUARES`
F32/F64, `MAX`/`MIN` F32/F64/I32/I64.

## Overlap and aliasing

- Reduction destinations (output and partials) must be injective and disjoint
  from the source of their own pass. Partials are pass 2's source, so they must
  be disjoint from the original input and from the final output.
- Movement destinations must be injective and must not overlap their source.
  `t4a_cuda_fill_zero_view` is included: two threads storing to one address is a
  data race even when both store zero.
- Arithmetic outputs must not overlap either input; in-place use is not part of
  revision 1.
- `t4a_cuda_tiled_transpose` requires disjoint source and destination.

The facade rejects any destination axis that has stride zero with extent greater
than one, for every destination including `t4a_cuda_fill_zero_view`. That is a
necessary condition only: full injectivity of a destination remains the caller's
proof, as it already is in the consumer planners.

## Launch geometry

| Entry | Grid | Block | Shared |
| --- | --- | --- | --- |
| flat entries | `(ceil(len/256), 1, 1)` | `(256, 1, 1)` | 0 |
| reductions | see the revision-3 section | `(plane_width, 1, 1)` | 0 |
| one-thread entries (`init_validation`, `extract_invalid_*`) | `(1, 1, 1)` | `(1, 1, 1)` | 0 |
| `validate_real_cast` | `(ceil(len/stride/256), 1, 1)` | `(256, 1, 1)` | 0 |
| `t4a_cuda_tiled_transpose` | `(ceil(M/tile), ceil(N/tile), batches)` | `(tile/vector_width, block_rows, 1)` | `tile*(tile+padding)*element_size` |

`grid.x <= 2^31-1` and `grid.y, grid.z <= 65535`, each checked by the facade.
A validated empty request (`len == 0`, or a zero extent, or zero batches) plans
a zero grid; the caller skips the launch, so it is zero physical launches.

## Compiler options

The tests compile with `nvcc -std=c++17 -arch=sm_80` (no `--use_fast_math`, no
`-ftz=true`). NVRTC consumers must pass the same options and the include path
for `cuda/include`. CUBIN is architecture-specific; ship or cache artifacts
with an explicit target, never a single architecture's CUBIN as universal.

## Integration note

A by-value POD argument needs a launcher that can marshal more than 8 bytes per
argument. `t4a-cuda-runtime`'s `Arg` currently exposes scalar and pointer
arguments only, so connecting this ABI to that runtime requires generic
by-value argument support (or a raw launch) there. This crate deliberately does
not reshape the metadata into a device upload to avoid that work.
