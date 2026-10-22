# C kernel ABI (revision 6)

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

`T4A_CUDA_ABI_REVISION == 6`. Revision 2 added `t4a_cuda_axis_values` and the
diagonal, windowing and validation-status entries; revision 3 added the reduction
entries; revision 4 added gather and additive scatter; revision 5 added the dtype
conversion entry; revision 6 adds the elementwise, comparison and
strided-combination entries. Nothing in revisions 1-5 changed. Any change to a struct layout, an entry signature, an element type
mapping or a launch contract requires a further revision bump and an entry in the
compilation/artifact key of any consumer that caches compiled code.

## Types

| C type | Size | Align | Notes |
| --- | --- | --- | --- |
| `t4a_cuda_layout` | 144 | 8 | by-value launch argument |
| `t4a_cuda_transpose_config` | 16 | 4 | by-value launch argument |
| `t4a_cuda_axis_values` | 72 | 8 | per-axis runtime values (revision 2) |
| `t4a_cuda_gather_plan` | 280 | 8 | gather dimension maps (revision 4) |
| `t4a_cuda_scatter_plan` | 216 | 8 | scatter dimension maps (revision 4) |
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

### Revision 4: gather and additive scatter

| Symbol | Arguments |
| --- | --- |
| `t4a_cuda_gather` | `(void* out, const void* operand, const void* indices, int dtype, int index_dtype, t4a_cuda_layout out, t4a_cuda_layout operand, t4a_cuda_layout indices, t4a_cuda_gather_plan plan, long long output_len)` |
| `t4a_cuda_scatter_add` | `(void* out, const void* indices, const void* updates, int dtype, int index_dtype, t4a_cuda_layout out, t4a_cuda_layout indices, t4a_cuda_layout updates, t4a_cuda_scatter_plan plan, long long update_len)` |

```c
typedef struct {              /* offsets: rank fields 0-20, window_dims 24, ... */
  int operand_rank;           /* 1..=8 */
  int out_rank;               /* 0..=8; a scalar gather has rank 0 */
  int index_vector_dim;       /* 0..=start_indices_rank; == rank when absent */
  int window_rank;            /* number of window positions */
  int start_map_len;          /* equal to the index vector size */
  int reserved;               /* 0 */
  long long window_dims[8];     /* operand axis per window position */
  long long offset_dims[8];     /* output axis per window position */
  long long start_index_map[8]; /* operand axis per start component */
  long long slice_sizes[8];     /* window extent per operand axis */
} t4a_cuda_gather_plan;

typedef struct {
  int operand_rank;
  int updates_rank;
  int index_vector_dim;
  int window_rank;
  int scatter_map_len;        /* equal to the index vector size */
  int reserved;
  long long window_dims[8];         /* operand axis per window position */
  long long update_window_dims[8];  /* updates axis per window position */
  long long scatter_dims[8];        /* operand axis per start component */
} t4a_cuda_scatter_plan;

void t4a_cuda_gather(void* out, const void* operand, const void* indices,
                     int dtype, int index_dtype, t4a_cuda_layout out_layout,
                     t4a_cuda_layout operand_layout, t4a_cuda_layout indices_layout,
                     t4a_cuda_gather_plan plan, long long output_len);
void t4a_cuda_scatter_add(void* out, const void* indices, const void* updates,
                          int dtype, int index_dtype, t4a_cuda_layout out_layout,
                          t4a_cuda_layout indices_layout, t4a_cuda_layout updates_layout,
                          t4a_cuda_scatter_plan plan, long long update_len);
```

- **Gather.** Window offsets fill the `window_dims` positions of the output
  coordinate, the remaining output coordinates fill the index batch shape in
  order, and each start component `c` (read at `index_vector_dim`, or from the
  flat batch position when that axis is absent) sets operand axis
  `start_index_map[c]` through `clamp_window_start`: `start <= 0 -> 0`, else
  `min(trunc(start), extent - window)`. The operand is read directly at its
  strided address. A rank-0 output is a valid scalar gather.
- **Scatter add.** One thread per update element. The window shape is the
  updates extent at each `update_window_dims` position (one on every other
  operand axis); the batch coordinates are the remaining updates axes, which must
  equal the index batch shape; the operand base is the clamped start per
  `scatter_dims`; a window that does not fit is skipped, per element, so scatter
  does not reject a window larger than the operand. The update is added with
  `atomicAdd` (complex updates add both components). The entry takes no operand
  pointer: the caller has already initialized `out` with an operand copy, and
  `out` carries every extent the fit check needs.
- **Index tensors** are compact, device-resident F32/F64/I32/I64. Their values
  must be validated integral starts within `|v| <= 2^24` (F32) and `<= 2^53`
  (F64); the entry does not re-validate and never reads them on the host.
- **Collisions.** Concurrent additive updates to one destination are
  nondeterministic, as in the baseline; the caller needs device atomics for the
  element type.
- **Initialization and empty cases.** The additive scatter must be preceded by a
  copy of the operand into `out` on the same stream. Use `t4a_cuda_copy_exact`
  only when both buffers are compact; use `t4a_cuda_contiguous_to_view` or
  `t4a_cuda_strided_to_strided` when either side is a view (the exact copy
  ignores offsets and strides). A zero update length plans no additive launch but
  still needs the copy when the destination is nonempty.
- **Launch geometry**: flat, `block = (256, 1, 1)`, `grid = ceil(len/256)`.

Supported dtypes: gather moves every data dtype with F32/F64/I32/I64 indices;
additive scatter serves F32/F64/C32/C64 with the same index dtypes.

### Revision 5: dtype conversions

| Symbol | Arguments |
| --- | --- |
| `t4a_cuda_convert` | `(void* out, const void* in, int dst_dtype, int src_dtype, long long len)` |

Flat (`block = (256,1,1)`, `grid = ceil(len/256)`), `len` logical elements,
source capacity `len * src_size`, destination capacity `len * dst_size`. Every
unequal pair of the seven dtypes is defined; an identity pair is not a
conversion (use `t4a_cuda_copy_exact` for a same-dtype copy).

- **Cast** follows the language cast `Out::cast_from(In)` lowers to: integer
  narrowing keeps the low bits, integer widening sign-extends, floats round to
  nearest-even, float→integer truncates toward zero, and Bool is a byte holding
  0 or 1.
- **Real→complex** writes `(cast(x), 0)`; **complex→real** writes `re`, never the
  magnitude; **C32↔C64** casts both components.
- **Bool destination** is a predicate: `x != 0` for numeric sources and
  `re != 0 || im != 0` for complex sources, writing exactly `1u8` or `0u8`.
  **Bool source** casts the 0/1 byte.
- **Floating-component → signed-integer precondition.** C++ float→int casts are
  undefined for NaN, infinity and out-of-range values, so the caller must run
  `t4a_cuda_validate_real_cast` and see no invalid position before converting:

  | Route | Validated buffer | Stride | Acceptance |
  | --- | --- | --- | --- |
  | F32/F64 → I32 | `len` | 1 | finite, `[-2^31, 2^31)` (F32 source) or `[-2^31, 2^31-1]` (F64 source) |
  | F32/F64 → I64 | `len` | 1 | finite, `[-2^63, 2^63)` |
  | C32/C64 → I32/I64 | `2 * len` | 2 | the same bounds on the real component; the imaginary slots are not read |

  The kernels do not re-validate: the planner cannot inspect values.
- **Aliasing**: source and destination byte spans must be disjoint, because a
  conversion can change the element width.

### Revision 6: elementwise, comparison and strided combination

| Symbol | Arguments |
| --- | --- |
| `t4a_cuda_unary` | `(void* out, const void* in, int dst_dtype, int src_dtype, int op, long long len)` |
| `t4a_cuda_binary` | `(void* out, const void* lhs, const void* rhs, int dtype, int op, int scalar_side, long long len, unsigned int* err)` |
| `t4a_cuda_compare` | `(void* out, const void* lhs, const void* rhs, int dtype, int cmp, long long len)` |
| `t4a_cuda_select` | `(void* out, const void* pred, const void* on_true, const void* on_false, int dtype, long long len)` |
| `t4a_cuda_clamp` | `(void* out, const void* in, const void* lower, const void* upper, int dtype, long long len)` |
| `t4a_cuda_combined_strided` | `(void* out, const void* a, const void* b, const void* coefficients, t4a_cuda_layout out_layout, t4a_cuda_layout a_layout, t4a_cuda_layout b_layout, int dtype, int op, long long len)` |

All except `t4a_cuda_combined_strided` are flat (`block = (256,1,1)`,
`grid = ceil(len/256)`) over `len` logical elements. Op tags are stable and
append-only; an unsupported `(op, dtype[, side])` triple is the caller's error and
is rejected by the facade planner. At the raw entry such a triple is caller UB and
the kernels make no promise about what they write, but the contract is the
facade's rejection, never that write.

`t4a_cuda_unary` — `len` elements of `src_dtype` into `dst_dtype`:

| Tag | Op | Dtype pair | Semantics |
| --- | --- | --- | --- |
| 0 | `NEG` | same, F32/F64/I32/I64/C32/C64 | `-x`; integer negation wraps |
| 1-10 | `EXP`, `LOG`, `SIN`, `COS`, `TANH`, `SQRT`, `RSQRT`, `EXPM1`, `LOG1P`, `ERF` | same, F32/F64 | CUDA math library; `RSQRT` is `rsqrtf`/`rsqrt` (~2 ulp), not `1/sqrt` |
| 11 | `ABS` | same F32/F64/I32/I64; F32←C32, F64←C64 | `fabs`; integers wrap on `INT_MIN`; complex is `hypotf`/`hypot` |
| 12 | `SIGN` | same, F32/F64/I32/I64/C32/C64 | float: NaN→NaN, **either zero → `+0`**, else `±1`; integer `0`/`±1`; complex `0` or `z/|z|` |
| 13 | `CONJ` | same, C32/C64 | imaginary part negated (`+0 → -0`) |

`t4a_cuda_binary` — `scalar_side` is `0` both elementwise, `1` `lhs` is one
element read at index 0, `2` `rhs` is one element read at index 0:

| Tag | Op | `dtype` | Sides | Semantics |
| --- | --- | --- | --- | --- |
| 0 | `MAX` | F32/F64/I32/I64 | 0,1,2 | NaN-propagating in lhs-then-rhs order; integer plain `>=` |
| 1 | `MIN` | F32/F64/I32/I64 | 0,1,2 | as `MAX` with `<=` |
| 2 | `POW` | F32/F64 | 0,1,2 | `powf`/`pow` |
| 3 | `REM` | F32/F64 | 0,1,2 | `fmodf`/`fmod`; a zero result keeps the dividend's sign |
| 4 | `MUL` | F32/F64/I32/I64/C32/C64 | 1,2 | `lhs*rhs`, integers wrap; the both-elementwise form is `t4a_cuda_mul` |
| 5 | `DIV_COMPLEX` | C32/C64 | 0 | scale-robust division over `2 * len` interleaved components |
| 6 | `DIV_CHECKED` | I32/I64 | 0,1,2 | `y==0` → `err=1`, out `0`; `y==-1` → wrapping `-x`; else `x/y` |
| 7 | `REM_CHECKED` | I32/I64 | 0,1,2 | `y==0` → `err=1`, out `0`; `y==-1` → out `0`; else `x - (x/y)*y` |
| 8 | `POW_CHECKED` | I32/I64 | 0,1,2 | `rhs<0` → `err=1`, out `0`; else square-and-multiply (`0^0 == 1`) |
| 9-12 | `MIXED_ADD`, `MIXED_SUB`, `MIXED_MUL`, `MIXED_DIV` | F32/F64 component | 1,2 | one real operand (1 element) with one complex operand (`2 * len` components); `1` = real is `lhs`. Only `MIXED_DIV` uses the robust form, with the real operand promoted to `(x, +0)` |
| 13 | `DIV` | F32/F64 | 1,2 | ordinary division; the both-elementwise form is `t4a_cuda_div` |

- `err` is one 4-byte, 4-byte-aligned word the caller zeroes before the first
  checked launch of a batch. A rejected element stores `1u` with `atomicExch`, so
  the word means "at least one element was invalid" and is never cleared; it is
  required for tags 6-8, must not overlap the operands, and is untouched
  otherwise (NULL is allowed there). This is deliberately **not** the revision-2
  flag, which records the lowest invalid position; the baseline's checked integer
  ops record only the boolean.
- `err` is not a synchronization point: the caller decides when to observe it.

`t4a_cuda_compare` — `cmp` is `0 EQ`, `1 LT`, `2 LE`, `3 GT`, `4 GE`; `dtype` is
F32/F64/I32/I64; the output is one byte per element holding `1` or `0`. An unknown
`cmp` writes `0`, as the baseline's default arm does (the facade cannot express
one).

`t4a_cuda_select` — `pred` is one byte per element (nonzero selects `on_true`);
`dtype` is F32/F64/I32/I64. Complex is not a ternary dtype in the baseline.

`t4a_cuda_clamp` — F32/F64 only. `out = x < lo ? lo : (x > hi ? hi : x)` with
ordered comparisons: a NaN `x` propagates and a NaN bound disables its comparison.
Inverted bounds (`lo > hi`) are not rejected: the statement order decides, so a
value below `lo` yields `lo` and any other value yields `hi`. The baseline left
this to CubeCL's `clamp` expansion, so this expression is the pinned contract.

`t4a_cuda_combined_strided` — one logical domain of `len = product(out.extents)`
elements, decoded through the destination extents and re-addressed through each
operand layout:

| Tag | Op | `dtype` | Semantics |
| --- | --- | --- | --- |
| 0 | `MUL` | F32/F64/I32/I64/C32/C64 | `out = a * b` (the baseline broadcast multiply); integers wrap |
| 1 | `AXPBY` | F32/F64/C32/C64 | `out = coefficients[0] * a + coefficients[1] * b` |

- `out` must be a destination (no stride-zero axis with extent greater than one).
  `a` and `b` may carry stride-zero broadcast axes, negative strides and their own
  offsets; all three layouts share the destination extents.
- `coefficients` is a two-element device buffer of `dtype`; `MUL` ignores it and
  may be given NULL, `AXPBY` requires it.
- **Aliasing**: `out` may alias an input only when that input's element type and
  layout (extents, strides, offset) are identical to `out`'s, so the thread that
  writes an element is the thread that read it. `AXPBY` may therefore run in place
  on `b` and `MUL` may scale in place through `a`. A shifted, broadcast,
  negative-stride or width-changing overlap is caller UB, so `unary`'s `ABS` route
  (C32→F32, C64→F64) and `compare`'s Bool destination must be disjoint from their
  inputs even when the element counts match: a narrower write can reach a byte
  another thread has not read. The scalar operand of `t4a_cuda_binary` and
  `coefficients` must be disjoint from `out`; the other operands and `pred` must be
  distinct allocations.

`t4a_cuda_binary`, `t4a_cuda_compare` and `t4a_cuda_select` may read one operand at
`len` elements while writing one element per position of a same-width type; their
buffer capacities are `len` elements of the named tag, or `2 * len` components for
the component-length tags, which the facade cannot see and therefore documents
rather than checks. `t4a_cuda_combined_strided` receives the caller's allocation
lengths and rejects a layout that reaches outside them.

## Overlap and aliasing

- Gather writes only its destination: the destination must be disjoint from the
  operand and from the index tensor, and it must be injective.
- Additive scatter reads `indices` and `updates` while other threads write `out`
  atomically, so both must be disjoint from `out` (logical collision in the
  scatter does not make aliasing safe). The destination must be injective for the
  preceding operand copy.
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
| gather / scatter add / convert | `(ceil(len/256), 1, 1)` | `(256, 1, 1)` | 0 |
| unary / binary / compare / select / clamp / combined strided | `(ceil(len/256), 1, 1)` | `(256, 1, 1)` | 0 |
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
