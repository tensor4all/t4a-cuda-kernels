# C kernel ABI (revision 1)

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

`T4A_CUDA_ABI_REVISION == 1`. Any change to a struct layout, an entry signature,
an element type mapping or a launch contract requires a revision bump and an
entry in the compilation/artifact key of any consumer that caches compiled code.

## Types

| C type | Size | Align | Notes |
| --- | --- | --- | --- |
| `t4a_cuda_layout` | 144 | 8 | by-value launch argument |
| `t4a_cuda_transpose_config` | 16 | 4 | by-value launch argument |
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

## Overlap and aliasing

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
