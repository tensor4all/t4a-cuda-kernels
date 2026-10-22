# Slice G1: linalg auxiliary kernels

Status: design (revised after the plan review), 2026-10-21. Seventh implementation
slice of [the migration design](cubecl-kernel-port.md), after
[slice F](slice-elementwise.md).

Baseline source: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
`crates/tenferro-linalg/src/gpu/kernels.rs` (491 lines) and the LU orchestration in
`crates/tenferro-linalg/src/gpu/linalg.rs` (`build_lu_outputs_device` L3891,
`build_lu_parity_device` L3958).

## 1. Scope

This slice ports the supporting kernels tenferro's CUDA linalg paths run around
its vendor calls: QR/Householder assembly and phase handling, the upper
trapezoidal check, and LU output extraction. The rank-revealing QR primitives in
`rank_revealing_qr.rs` are the next slice (G2).

| Baseline symbol | This slice |
| --- | --- |
| `fill_one_kernel` (L73) | `t4a_cuda_fill_one` |
| `matrix_adjoint_real` (L81), `matrix_adjoint_complex` (L93) | `t4a_cuda_matrix_adjoint` |
| `householder_explicit_v` (L119) | `t4a_cuda_householder_explicit_v` |
| `householder_q_columns_identity` (L135), `initialize_q` (RRQR L595) | `t4a_cuda_identity_shifted` |
| `positive_phase_real`/`_c32`/`_c64` (L152-L183), `qr_phase_real`/`_c32`/`_c64` (L217-L251) | `t4a_cuda_qr_phase` |
| `qr_apply_phase_real` (L253), `qr_apply_phase_complex` (L276) | `t4a_cuda_qr_apply_phase` |
| `upper_trapezoidal_violation` (L298) | `t4a_cuda_upper_trapezoidal_violation` |
| `householder_from_factors_assemble` (L315) | `t4a_cuda_householder_assemble` |
| `lu_extract_outputs` (L345) | `t4a_cuda_lu_extract_outputs` |
| `lu_parity` (L420) | `t4a_cuda_lu_parity` |
| `lu_apply_pivots` (L442) | `t4a_cuda_lu_apply_pivots` |
| `complex32_magnitude` / `complex64_magnitude` (L105, L112) | **already covered** by `t4a_cuda_unary` op `ABS` (revision 6); neither has a caller in the pinned source |
| `extract_r` (RRQR L609) | next slice (a triangle-masked copy out of the RRQR work matrix) |

Eleven entries, one per operation family, for 21 baseline entry symbols.

## 2. Operand conventions

Every matrix operand is a `t4a_cuda_layout`: axis 0 is the row, axis 1 the
column, axes 2.. are batch axes. Index arithmetic is on logical coordinates, so a
strided operand behaves exactly like a compact one (the baseline indexes `Tensor`
values, which are layout-aware).

**Logical bounds come from the layouts.** Each entry derives its launch domain and
every per-operand bound from the extent products; the caller's allocation lengths
are a host-side validation input consumed by the `plan_*` functions, never a kernel
bound. A kernel that used a capacity as a tail bound would wrap coordinates and
repeat writes, which is why the ABI does not pass them.

`phase` is the one operand that does not carry the matrix rank: its axis 0 is the
diagonal index and its axes 1.. are the matrix's batch axes (the baseline's
`qr_phase_offset` / `qr_diagonal_offset` skip the matrix row axis). A caller
therefore passes the phase layout with rank `matrix_rank - 1`.

Coefficient arrays (`coeff_q`, `coeff_out`) are **flat**: the baseline addresses
them as `coeff[pos]`, so no layout applies and each carries an explicit length.

Pivot arrays are two different things and keep their baseline types:

| Array | Type | Indexing |
| --- | --- | --- |
| LU `pivots` | `int32_t`, `1`-based row position (`pivot - 1` is the index) | `pivots[step + batch * k]`, `k = min(rows, cols)` |
| RRQR `permutation` / `pivots` | `int64_t`, `0`-based | next slice |

No entry in this slice takes a status word: none of the eleven baseline kernels
reports nonfinite input. The revision-6 sticky boolean stays the only status
convention until the RRQR slice adds its per-batch words.

## 3. ABI additions (revision 7)

`T4A_CUDA_ABI_REVISION == 7`. Revisions 1-6 are unchanged. No new struct.

```c
void t4a_cuda_fill_one(void* out, int32_t dtype, t4a_cuda_layout out_layout);

void t4a_cuda_matrix_adjoint(void* out, const void* v, int32_t dtype,
                             t4a_cuda_layout out_layout, t4a_cuda_layout v_layout);

void t4a_cuda_householder_explicit_v(void* out, const void* packed, int32_t dtype,
                                     t4a_cuda_layout out_layout, t4a_cuda_layout packed_layout);

void t4a_cuda_identity_shifted(void* out, int32_t dtype, t4a_cuda_layout out_layout,
                               int64_t start);

void t4a_cuda_qr_phase(void* phase, const void* r, int32_t dtype,
                       t4a_cuda_layout phase_layout, t4a_cuda_layout r_layout);

void t4a_cuda_qr_apply_phase(void* q, void* r, const void* phase, int32_t dtype,
                             t4a_cuda_layout q_layout, t4a_cuda_layout r_layout,
                             t4a_cuda_layout phase_layout, int64_t q_start);

void t4a_cuda_upper_trapezoidal_violation(void* violation, const void* input, int32_t dtype,
                                          t4a_cuda_layout violation_layout,
                                          t4a_cuda_layout input_layout);

void t4a_cuda_householder_assemble(void* packed_out, void* coeff_out,
                                   const void* packed_q, const void* coeff_q,
                                   const void* folded_r, int32_t dtype,
                                   t4a_cuda_layout packed_out_layout,
                                   t4a_cuda_layout packed_q_layout,
                                   t4a_cuda_layout folded_r_layout,
                                   int64_t coeff_out_len, int64_t factor_width);

void t4a_cuda_lu_extract_outputs(void* p_out, void* l_out, void* u_out, void* parity_out,
                                 const void* work, const void* pivots, int32_t dtype,
                                 t4a_cuda_layout p_layout, t4a_cuda_layout l_layout,
                                 t4a_cuda_layout u_layout, t4a_cuda_layout work_layout,
                                 int64_t k, int64_t batches);

void t4a_cuda_lu_parity(void* parity_out, const void* pivots, int32_t dtype, int64_t k,
                        int64_t batches);

void t4a_cuda_lu_apply_pivots(void* out, const void* input, const void* pivots, int32_t dtype,
                              t4a_cuda_layout out_layout, t4a_cuda_layout in_layout,
                              int64_t k, int32_t inverse);
```

All eleven are flat launches (`block = (256,1,1)`, `grid = ceil(domain/256)`); the
domain is stated per entry. `len == 0` plans no launch.

### 3.1 Fills and copies

- `fill_one`: `out[logical i] = 1` over `out_len = out.element_count()` logical
  positions, every dtype (Bool stores the byte `1`). The layout form subsumes the
  baseline's compact `out[pos]`, which already indexes logical positions.
- `matrix_adjoint`: `out(i, j, batch) = v(j, i, batch)`, conjugated for C32/C64.
  Domain = the output count.
- `householder_explicit_v`: **rank 2 only** (the baseline addresses row and column
  and never a batch axis): `out(i, j) = 0` for `i < j`, `1` for `i == j`,
  `packed(i, j)` for `i > j`. `out` rows must equal `packed` rows and `out` columns
  must not exceed `packed` columns, because the entry reads only the strict-lower
  coordinates of the requested columns — an explicit-`V` extraction may therefore
  be narrower than the packed factor matrix. Domain = the output count.
- `identity_shifted`: `out(i, j, batch) = (i == start + j) ? 1 : 0`. `start = 0` is
  the baseline `initialize_q`, which is batched. `start > 0` is the baseline
  2-D `householder_q_columns_identity` extended to the batch axes, and columns
  beyond `row_extent - start` are entirely zero — that is the baseline's
  complement-column behaviour, so `start + columns` may exceed the row extent.
  Domain = the output count.

### 3.2 QR phase

`qr_phase` writes one entry per phase position: `phase[i, batch] = positive_phase(R[i, i, batch])`,
domain = the phase's logical count. For real dtypes the phase is `-1` if the
diagonal is negative and `+1` otherwise, so a NaN diagonal yields `+1` because the
baseline's `diagonal < 0` test is false. For complex dtypes it is `1` when the
magnitude is zero, else `diagonal / |diagonal|`.

`qr_apply_phase` scales the trailing columns: for every `q` position,
`index = q_start + q.column`, and if `index < phase.extent[0]` then
`q *= phase[index, batch]`; for every `r` position, `index = r.row` and
`r *= phase[index, batch]`, conjugated for C32/C64. Domain = the larger logical
count of `q` and `r`, as the baseline covers both operands in one launch. The `q` side is guarded; the
`r` side relies on `r.rows <= phase.extent[0]`, which the planner enforces so no
entry can read past the phase array.

### 3.3 Assembly and validation

- `upper_trapezoidal_violation`: `violation(i, j) = (i > j && input(i, j) != 0) ? 1 : 0`
  as `i32`; domain = `violation_len`, and `input` must cover the same coordinates.
- `householder_assemble`: **rank 2 for the matrix operands** (the baseline reads
  row/column only). One launch over `max(packed_out_len, coeff_out_len)`.
  `packed_out(i, j)` is `folded_r(i, j)` when `i < factor_width && i <= j`, else
  `packed_q(i, j)` when `j < factor_width && i > j`, else zero; `coeff_out[p]` is
  `coeff_q[p]` for `p < factor_width`, else zero. The planner requires
  `factor_width >= 0`, `folded_r` to cover rows `< min(factor_width, packed rows)`
  across all packed columns, and `packed_q` to cover all packed rows across columns
  `< min(factor_width, packed columns)`, plus `coeff_q_len >= min(factor_width, coeff_out_len)`.
  That is exactly the region the guarded branches read; requiring full extent
  equality would reject the baseline's own narrower sources.
- `qr_phase`: `phase.rank == r.rank - 1`, batch extents equal,
  `phase.extent[0] <= min(r.rows, r.cols)` so every diagonal it reads exists.
- `qr_apply_phase`: same rank/batch rule, `r.rows <= phase.extent[0]`, `q.rows`
  and `r.cols` unconstrained, `q_start >= 0` and `q_start + q.cols` computed with
  checked arithmetic (overflow is a planning error, not a wrapped index).
- `matrix_adjoint`: both operands rank `>= 2`, equal batch extents, `out.rows ==
  v.cols` and `out.cols == v.rows`.
- `identity_shifted`: `start >= 0`, extent product computed with checked
  arithmetic.
- `upper_trapezoidal_violation`: both operands rank `>= 2` with equal extents.

### 3.4 LU

Shapes follow the pinned caller: `work = [m, n, batch]`, `P = [m, m, batch]`,
`L = [m, k, batch]`, `U = [k, n, batch]`, `parity = [batch]`, `k = min(m, n)`, and
`pivots` holds `k` `int32_t` per batch (1-based). The planner requires exactly
those relationships (a `P` column or a `U` row outside them would index outside
`work`, and a `k` other than `min(m, n)` would let a `U` row leave it), plus
`batches == product(batch extents)`, `k.checked_mul(batches) <= pivots_len` and
`parity_len >= batches`.

- `lu_extract_outputs`: one launch over `max(p_len, l_len, u_len, batches)`.
  `p_out(i, j, batch) = (final_row == i) ? 1 : 0`, where `final_row` starts at `j`
  and, for each step in `0..k`, is exchanged with `pivots[step + batch*k] - 1` when
  it equals either side of that exchange. `l_out` is unit lower triangular with
  `work` in the strict lower part, `u_out` is upper triangular including the
  diagonal with zeros below, and `parity_out[batch]` is `-1` per pivot that moved a
  row and `+1` otherwise. Dtypes F32/F64/C32/C64 (the baseline requires negation).
  The parity loop is serial over `k` **per batch**, with batches running in
  parallel threads and no cross-thread dependency; that is the baseline form, and
  `k` is the factorisation's own step count.
- `lu_parity`: the same parity from `pivots[batch*k ..]` alone, one thread per
  batch, domain = `batches`.
- `lu_apply_pivots`: `out(i, j, batch) = input(source_row, j, batch)`, where
  `source_row` starts at `i` and is exchanged per pivot, applied in ascending step
  order for `inverse != 0` and descending order otherwise, exactly as the baseline.
  Domain = the output count; `input` must have the same extents as `out`, `k <= rows`,
  and the planner requires `k * product(batch extents) <= pivots_len`.
  Pivot values must lie in `[1, rows]` — a raw-call obligation, as the baseline's
  is; the kernels do not clamp.

## 4. Aliasing and overlap

- `qr_apply_phase` writes `q` and `r` **in place**: each thread reads and writes its
  own `q` or `r` position (the phase operand is read-only), so `q` and `r` may be
  separate in-place buffers. Both must still be destinations (no broadcast axis) and
  must not overlap each other or `phase`.
- Every other writable operand must be a destination (no stride-zero axis with
  extent greater than one, as in revisions 1-6) and must be disjoint from all
  read operands: `lu_extract_outputs` writes `P`, `L`, `U` and `parity` while
  reading `work` and `pivots`; `householder_assemble` writes `packed_out` and
  `coeff_out` while reading `packed_q`, `coeff_q` and `folded_r`;
  `householder_explicit_v`, `matrix_adjoint`, `identity_shifted`, `qr_phase`,
  `upper_trapezoidal_violation` and `lu_apply_pivots` write one output each.
  `lu_extract_outputs`' four outputs are mutually disjoint.
- Full injectivity of a destination remains the caller's proof, as in
  revisions 1-6.

## 5. Dtype matrices

| Entry | F32 | F64 | I32 | I64 | Bool | C32 | C64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fill_one` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `matrix_adjoint` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| `householder_explicit_v` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| `identity_shifted` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| `qr_phase`, `qr_apply_phase` | ✓ | ✓ | – | – | – | ✓ | ✓ |
| `upper_trapezoidal_violation` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `householder_assemble` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| `lu_extract_outputs`, `lu_parity` | ✓ | ✓ | – | – | – | ✓ | ✓ |
| `lu_apply_pivots` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |

The violation matrix is an `i32` payload whatever the input dtype; its tag names
the *input* type. `lu_parity`'s tag names the parity payload type; its pivots are
always `int32_t`.

## 6. Facade

One `plan_*` per entry, taking the layouts, the caller's allocation lengths (and
the pivot-array length for the two LU entries) and returning the flat launch. The
kernels take only the layouts, so a bound can never come from a capacity. `Error` needs no new variant: `RankOutOfRange`,
`ShapeMismatch`, `NegativeCount`, `Overflow`, `BroadcastDestination` and
`UnsupportedDType` already express every failure. `require_destination` is applied
to every layout-backed output.

## 7. Tests

Planning: the dtype matrix (every unsupported combination rejected), each shape
rule of §3 including the narrower explicit-`V` and assembly sources, phase
diagonal bounds, `q_start` overflow, the LU P/L/U/parity relationships, empty
domains, and the launch geometry across a block boundary.

GPU (local A100): every entry against a CPU reference for a compact and a strided
batched operand, with sizes above one block; phase rules at `+0`, `-0`, negative,
zero-magnitude and NaN diagonals; `qr_apply_phase` with a complement column left
untouched and with `q_start > 0`; LU pivots applied forwards and backwards
(`apply` with `inverse` then without restores the operand) and parity against a
direct permutation-parity count on 1×1, 2×2 and a pivot-heavy 8×5 case;
`householder_assemble` and `identity_shifted` at their boundaries
(`factor_width` 0, 1 and larger than the matrix; `start` 0 and `start > 0` with
zero complement columns); `matrix_adjoint` of an asymmetric strided view;
`upper_trapezoidal_violation` with NaN and negative zeros; and one physical launch
per call.

## 8. Deliberate exclusions

- The RRQR primitives and `extract_r`: slice G2.
- `complex32_magnitude` / `complex64_magnitude`: no caller in the pinned source,
  and revision 6's `unary`/`ABS` already covers the operation.
- Host orchestration (the step loop, buffer allocation, pivot validation) stays
  with the consumer, per the migration design.
