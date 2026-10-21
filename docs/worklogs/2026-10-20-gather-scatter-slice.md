# Gather and additive-scatter slice — local evidence

Date: 2026-10-20. Plan: [slice-gather-scatter.md](../design/slice-gather-scatter.md).

## What landed

ABI revision 4 adds the XLA-style indexing family to the migration:

- `t4a_cuda_gather` with the by-value `t4a_cuda_gather_plan` (window/offset dims,
  start-index map, slice sizes, index-vector dim).
- `t4a_cuda_scatter_add` with `t4a_cuda_scatter_plan` (window dims,
  update-window dims, scatter dims); the window shape comes from the updates
  extents, and the entry takes no operand because the caller initializes the
  destination with an operand copy first.
- Host planners `plan_gather` / `plan_scatter_add` (shape derivation, index
  vector size, batch equality, compact index tensors, spans, grid limits) and a
  `window_dims` helper for the collapsed-axis derivation.
- `scatter_copy_kernel` is covered by `t4a_cuda_copy_exact` for compact buffers;
  the strided initialization cases reuse `contiguous_to_view` /
  `strided_to_strided`, as documented.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe (`nvidia-smi`) |
| Driver | 580.126.09 |
| CUDA toolkit | 12.6 installed; the tests invoke `nvcc` 12.0.140 from `PATH` |
| Test compile flags | `nvcc -std=c++17 -cubin -arch=sm_80 -I cuda/include` |
| Rust | 1.97.1 |

## Commands and results

```bash
cargo fmt --check                                                      # pass
cargo clippy -j 16 --all-targets --features gpu-tests -- -D warnings   # pass
cargo test -j 16                                                       # 42 planning tests pass, no CUDA
cargo test -j 16 --features gpu-tests                                  # 42 planning + 16 + 13 + 16 + 11 GPU tests pass
```

GPU coverage: gather against an independent reference built from the output's
cartesian product (not a flat-cursor walk) for a collapsed-axis case, a
middle-axis index vector, two window dimensions with reversed offset dims, a
strided operand with clamped starts at both ends, a strided destination whose
holes and allocation tail stay sentinel, a Bool operand, a scalar (rank-0) output
with float indices, and a 2^16-output domain; scatter against an independent
reference with collisions, two window dimensions with reversed update-window
dims, complex updates through two atomics, a window that can never fit (nothing
written), a strided destination initialized through `contiguous_to_view` with
sentinel holes, a 2^16-update collision domain, the ordered copy+add composition
enqueued with `launch_async` (two launches, one synchronization, nonzero
operand), and the zero-update case (one copy launch only).

## Review round

An independent review of the diff found and this branch fixes: gather validated
slice bounds only on window axes (a collapsed axis with slice 0 could read out of
bounds — now every collapsed axis must carry slice 1); gather was grouped with
the reduction dtype list and rejected Bool data; an empty scatter destination
returned a launch instead of none; the "unsorted offset/update-window" fixtures
had only one window dimension and could not test ordering; the claimed strided
destination and sentinel-hole coverage was absent; the ABI's overlap section
omitted the gather/scatter disjointness rules; the reconstructed batch/window
product check could overflow on empty metadata; `CGatherPlan` reported
`out_rank = 0` for non-scalar outputs; and the new C prototypes shadowed the
pointer name with the layout name. Coverage now includes reversed two-dimension
maps, strided destinations with verified holes, and a Bool gather.

## Findings

- Three test-side mistakes were caught and corrected before the slice closed:
  the references enumerated coordinates row-major while the repository's logical
  order has axis 0 fastest; a compact index tensor's vector axis is the *last*
  axis, so a fixture that assumed per-batch-contiguous components addressed
  different starts; and the "window does not fit" case needed a window larger
  than the destination extent (clamping otherwise makes any window fit).
- `index_vector_size` is 1 when the index-vector axis is absent (a scalar start
  per batch), not the product of the index extents; the baseline's helper defines
  it that way and the facade now matches.

## Not done in this slice

Conversions, linalg/RRQR helpers, the fused-expression emitter and the NVRTC
prepare/cache path. No hosted CI was used and no performance comparison against
the CubeCL baseline is claimed.
