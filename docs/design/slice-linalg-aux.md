# Slice G1: linalg auxiliary kernels

Status: design, 2026-10-21. Seventh implementation slice of
[the migration design](cubecl-kernel-port.md), after
[slice F](slice-elementwise.md).

Baseline source: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
`crates/tenferro-linalg/src/gpu/kernels.rs` (491 lines).

## 1. Scope

This slice ports the supporting kernels tenferro's CUDA linalg paths run around
its vendor calls: QR/Householder assembly and phase handling, the upper
trapezoidal check, and LU output extraction. The rank-revealing QR primitives in
`rank_revealing_qr.rs` are the next slice (G2), and `initialize_q` /
`extract_r` are split between the two as the table shows.

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
| `complex32_magnitude` / `complex64_magnitude` (L105, L112) | **already covered** by `t4a_cuda_unary` op `ABS` (revision 6); they have no caller in the pinned source |
| `extract_r` (RRQR L609) | next slice (it is a triangle-masked copy out of the RRQR work matrix) |

Eleven entries, one per operation family, for 21 baseline entry symbols.

## 2. Matrix operands and the phase operand

Every matrix operand is a `t4a_cuda_layout`: axis 0 is the row, axis 1 the
column, axes 2.. are batch axes; `len` is the logical element count. The row,
column and batch extents come from the layout, so no separate `m`/`n`/`batches`
argument is needed except where the baseline itself passes them.

`phase` is the one operand that does not carry the matrix rank: its axis 0 is the
diagonal index and its axes 1.. are the matrix's batch axes (the baseline's
`qr_phase_offset` / `qr_diagonal_offset` skip the matrix row axis). The kernels
build addresses with exactly that rule, so a caller passes the phase layout with
rank `matrix_rank - 1`.

Conventions carried from the baseline:

- Index arithmetic is on logical coordinates, so a strided operand works the same
  way a compact one does; the entries do not require compactness.
- `pivots` is a flat `i64` array of the per-batch LU/RRQR pivot sequence, holding
  **1-based** row positions (`pivot - 1` is the index); `k` is the number of steps.
- Status words are `i64`, one per batch or one per launch, initialized to `0` by
  the caller; `1` means "a participating value was nonfinite". No entry
  synchronizes or reads status back.

## 3. ABI additions (revision 7)

`T4A_CUDA_ABI_REVISION == 7`. Revisions 1-6 are unchanged. No new struct: the
existing `t4a_cuda_layout` covers every matrix operand.

```c
void t4a_cuda_fill_one(void* out, int32_t dtype, t4a_cuda_layout out, int64_t len);

void t4a_cuda_matrix_adjoint(void* out, const void* v, int32_t dtype,
                             t4a_cuda_layout out, t4a_cuda_layout v, int64_t len);

void t4a_cuda_householder_explicit_v(void* out, const void* packed, int32_t dtype,
                                     t4a_cuda_layout out, t4a_cuda_layout packed, int64_t len);

void t4a_cuda_identity_shifted(void* out, int32_t dtype, t4a_cuda_layout out,
                               int64_t start, int64_t len);

void t4a_cuda_qr_phase(void* phase, const void* r, int32_t dtype,
                       t4a_cuda_layout phase, t4a_cuda_layout r, int64_t len);

void t4a_cuda_qr_apply_phase(void* q, void* r, const void* phase, int32_t dtype,
                             t4a_cuda_layout q, t4a_cuda_layout r, t4a_cuda_layout phase,
                             int64_t q_start, int64_t q_len, int64_t r_len);

void t4a_cuda_upper_trapezoidal_violation(void* violation, const void* input, int32_t dtype,
                                          t4a_cuda_layout violation, t4a_cuda_layout input,
                                          int64_t len);

void t4a_cuda_householder_assemble(void* packed_out, void* coeff_out, const void* packed_q,
                                   const void* coeff_q, const void* folded_r, int32_t dtype,
                                   t4a_cuda_layout packed_out, t4a_cuda_layout coeff_out,
                                   t4a_cuda_layout packed_q, t4a_cuda_layout coeff_q,
                                   t4a_cuda_layout folded_r, int64_t factor_width,
                                   int64_t packed_len, int64_t coeff_len);

void t4a_cuda_lu_extract_outputs(void* p_out, void* l_out, void* u_out, void* parity_out,
                                 const void* work, const void* pivots, int32_t dtype,
                                 t4a_cuda_layout p_out, t4a_cuda_layout l_out,
                                 t4a_cuda_layout u_out, t4a_cuda_layout work,
                                 int64_t k, int64_t p_len, int64_t l_len, int64_t u_len,
                                 int64_t batches);

void t4a_cuda_lu_parity(void* parity_out, const void* pivots, int32_t dtype, int64_t k,
                        int64_t len);

void t4a_cuda_lu_apply_pivots(void* out, const void* input, const void* pivots, int32_t dtype,
                              t4a_cuda_layout out, t4a_cuda_layout input, int64_t k,
                              int32_t inverse, int64_t len);
```

All eleven are flat launches (`block = (256,1,1)`, `grid = ceil(domain/256)`); the
domain is stated per entry below. `len == 0` plans no launch.

### 3.1 Fills and copies

- `fill_one`: `out[logical i] = 1` for `len = out.element_count()` logical
  elements, every dtype (Bool stores the byte `1`). The layout form subsumes the
  baseline's compact `out[pos]`, which already indexes logical positions.
- `matrix_adjoint`: `out(i, j, batch) = v(j, i, batch)`, conjugated for C32/C64.
  `out` and `v` have the same rank `>= 2` and the same extents with axes 0 and 1
  swapped; `len` is the output count. Dtypes F32/F64/I32/I64/C32/C64.
- `householder_explicit_v`: `out(i, j) = 0` for `i < j`, `1` for `i == j`, and
  `packed(i, j)` for `i > j` (all batch axes shared). `len` is the output count.
- `identity_shifted`: `out(i, j, batch) = (i == start + j) ? 1 : 0`; `start = 0` is
  the RRQR `initialize_q` identity. Columns beyond `row_extent - start` are all
  zero, which is the baseline's complement-column behaviour. `len` is the output
  count.

### 3.2 QR phase

`qr_phase` writes one entry per phase position: `phase[i, batch] = positive_phase(R[i, i, batch])`, with `len` = the phase count. For real dtypes the phase is `-1` if the diagonal is negative and `+1` otherwise (including NaN, as in the baseline's `diagonal < 0` test). For complex dtypes it is `1` when the magnitude is zero, otherwise `diagonal / |diagonal|`.

`qr_apply_phase` scales the trailing columns: for every `q` position,
`index = q_start + q.column`, and if `index < phase.extent[0]` then
`q *= phase[index, batch]`; for every `r` position, `index = r.row` and
`r *= phase[index, batch]`, conjugated for C32/C64. The launch domain is
`max(q_len, r_len)` so both operands are covered by one launch exactly as the
baseline does; a full-`Q` complement column (`index >= phase.extent[0]`) is left
untouched.

### 3.3 Assembly, validation, LU

- `upper_trapezoidal_violation`: `violation(i, j) = (i > j && input(i, j) != 0) ? 1 : 0`
  as `i32`; `len` is the violation count, and `input` shares its extents.
- `householder_assemble`: one launch over `max(packed_len, coeff_len)`.
  `packed_out(i, j)` is `folded_r(i, j)` when `i < factor_width && i <= j`, else
  `packed_q(i, j)` when `j < factor_width && i > j`, else zero; `coeff_out[p]` is
  `coeff_q[p]` for `p < factor_width` and zero otherwise. `factor_width` is the
  baseline's trailing-factor count.
- `lu_extract_outputs`: one launch over `max(p_len, l_len, u_len, batches)`.
  `p_out(i, j, batch) = (final_row == i) ? 1 : 0` where `final_row` starts at `j`
  and, for each step in `0..k`, is exchanged with `pivots[step + batch*k] - 1`
  when it equals either side of that exchange. `l_out` is unit lower triangular
  with `work` in the strict lower part, `u_out` is upper triangular (including
  the diagonal) with zeros below, and `parity_out[batch]` is `-1` per pivot that
  moved a row, `+1` otherwise. Dtypes F32/F64/C32/C64 (the baseline requires
  negation and is only launched for those).
- `lu_parity`: `parity_out[batch]` from `pivots[batch*k ..]` alone, `len = batches`.
- `lu_apply_pivots`: `out(i, j, batch) = input(source_row, j, batch)`, where
  `source_row` starts at `i` and is exchanged per pivot, applied in ascending
  step order for `inverse != 0` and descending order otherwise, matching the
  baseline. `len` is the output count; every dtype.

## 4. Dtype matrices

| Entry | F32 | F64 | I32 | I64 | Bool | C32 | C64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fill_one` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `matrix_adjoint` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| `householder_explicit_v` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| `identity_shifted` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| `qr_phase` | ✓ | ✓ | – | – | – | ✓ | ✓ |
| `qr_apply_phase` | ✓ | ✓ | – | – | – | ✓ | ✓ |
| `upper_trapezoidal_violation` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `householder_assemble` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| `lu_extract_outputs`, `lu_parity` | ✓ | ✓ | – | – | – | ✓ | ✓ |
| `lu_apply_pivots` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | 

The violation matrix is an `i32` payload whatever the input dtype; its dtype tag
names the *input* type. `lu_parity`'s `dtype` names the parity payload type
(F32/F64/C32/C64); the pivots are `i64` whatever it is.

## 5. Facade

Each entry gets one `plan_*` returning the flat launch plus the shape checks the
kernel relies on:

```rust
pub fn plan_fill_one(dtype, out: &Layout, out_len) -> Result<Launch, Error>;
pub fn plan_matrix_adjoint(dtype, out: &Layout, out_len, v: &Layout, v_len) -> Result<Launch, Error>;
pub fn plan_householder_explicit_v(dtype, out: &Layout, out_len, packed: &Layout, packed_len) -> Result<Launch, Error>;
pub fn plan_identity_shifted(dtype, out: &Layout, out_len, start: i64) -> Result<Launch, Error>;
pub fn plan_qr_phase(dtype, phase: &Layout, phase_len, r: &Layout, r_len) -> Result<Launch, Error>;
pub fn plan_qr_apply_phase(dtype, q: &Layout, q_len, r: &Layout, r_len, phase: &Layout, phase_len, q_start: i64) -> Result<Launch, Error>;
pub fn plan_upper_trapezoidal_violation(dtype, violation: &Layout, violation_len, input: &Layout, input_len) -> Result<Launch, Error>;
pub fn plan_householder_assemble(dtype, packed_out, coeff_out, packed_q, coeff_q, folded_r, factor_width) -> Result<Launch, Error>;
pub fn plan_lu_extract_outputs(dtype, p_out, l_out, u_out, work, k, batches) -> Result<Launch, Error>;
pub fn plan_lu_parity(dtype, batches: i64, k: i64) -> Result<Launch, Error>;
pub fn plan_lu_apply_pivots(dtype, out, out_len, input, input_len, k: i64) -> Result<Launch, Error>;
```

Checks: rank `>= 2` for matrices (except `fill_one`, `lu_parity`), the
swapped-axes rule for `matrix_adjoint`, equal extents for the element pairs,
`phase.rank == r.rank - 1` with `phase.extents[1..] == r.extents[2..]`,
`start >= 0` and `start + column_extent <= row_extent` for `identity_shifted`,
`k >= 0` and `batches >= 0`, the dtype matrix above, and the caller's allocation
span for every layout-carrying operand. `Error` gains
`MatrixRankTooSmall`, `InvalidDiagonalRange` and `PivotCountMismatch`.

## 6. Tests

Planning: the dtype matrix (rejection of every unsupported triple), all shape
checks above, empty domains, and the launch geometry across a block boundary.

GPU (local A100): every entry against a CPU reference for a batched strided
operand and for a compact one, with sizes above one block; the phase rules at
`+0`, `-0`, negative, zero-magnitude and NaN diagonals; `qr_apply_phase` with a
complement column (`index >= phase.extent[0]`) left untouched; the LU pivot
round trip (`apply` then `apply` with `inverse` restores the operand) and the
`parity` sign against a direct permutation-parity count; `householder_assemble`
at the `factor_width` boundaries; `identity_shifted` with `start > 0`;
`matrix_adjoint` with an asymmetric strided view; and one launch per call.

## 7. Deliberate exclusions

- The RRQR primitives and `extract_r`: slice G2.
- `complex32_magnitude` / `complex64_magnitude`: no caller in the pinned source,
  and revision 6's `unary`/`ABS` already covers the operation.
- Host orchestration (the step loop, buffer allocation, when to read status) stays
  with the consumer, per the migration design.
