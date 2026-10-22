# Slice G2: rank-revealing QR primitives

Status: design, 2026-10-21. Eighth implementation slice of
[the migration design](cubecl-kernel-port.md), after
[slice G1](slice-linalg-aux.md). This slice closes the custom-kernel inventory
that revision 1 opened.

Baseline source: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
`crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs` (762 lines), with the
step loop in `crates/tenferro-linalg/src/gpu/linalg/rank_revealing_qr.rs`.

## 1. Scope

| Baseline symbol | This slice |
| --- | --- |
| `initialize_permutation` (L24) | `t4a_cuda_initialize_permutation` |
| `column_norms_real` (L32), `_c32` (L79), `_c64` (L128) | `t4a_cuda_column_norms` |
| `select_pivot` (L177) | `t4a_cuda_select_pivot` |
| `swap_columns` (L232) | `t4a_cuda_swap_columns` |
| `reflector_real` (L265), `_c32` (L335), `_c64` (L394) | `t4a_cuda_reflector` |
| `apply_reflector_real` (L453), `_c32` (L499), `_c64` (L547) | `t4a_cuda_apply_reflector` |
| `extract_r` (L609) | `t4a_cuda_extract_r` |
| `prefix_rank_real` (L630), `_c32` (L676), `_c64` (L720) | `t4a_cuda_prefix_rank` |

Eight entries for 15 baseline symbols. `initialize_q` is already served by
revision 7's `identity_shifted` (`start = 0`), and the bounded host step loop stays
with the consumer, as the migration design requires.

## 2. Conventions

- Matrix operands are layouts: axis 0 the row, axis 1 the column, axes 2.. batch.
- `norms`, `permutation` and `coeff` are the one-axis-shorter operands: axis 0 is
  the column (or factorization step) and axes 1.. are the matrix's axes 2.., the
  same rule revision 7 uses for `phase`.
- `pivots`, `rank` and `status` are flat `int64_t` arrays with one entry per batch.
  The RRQR pivot is a **0-based column index** (the LU `int32_t` 1-based pivots of
  revision 7 are a different convention and are not mixed here).
- `status[batch]` is a caller-zeroed `int64_t` word: `1` means "a participating
  magnitude, or the norm computed from it, was nonfinite". It is written by the
  kernel and never read back by it (revision 6's word is the same kind of sticky
  boolean; revision 2's *lowest invalid position* flag is a different convention and
  is not reused). The consumer decides when to observe it.
- Every flat array (`permutation`, `pivots`, `rank`, `status`) gets an explicit
  element count in its `plan_*` function and is checked against the batch product,
  exactly as revision 7 does for the LU pivots and parity.
- Logical bounds again come from the layouts; capacities are host-side validation
  inputs.

## 3. Plane topology

Every reduction entry in this slice uses **one 32-lane plane per block**
(`block = (32,1,1)`) with the full warp mask, matching the baseline's `PLANE_DIM`
and the caller's `RRQR_PLANE_WIDTH = 32`. Lanes stride the row axis by 32, so the
per-thread work is `ceil(rows / 32)`; a lane with no rows contributes its zero
accumulator, and every collective runs for the whole plane. Lanes never return
before a shuffle.

## 4. ABI additions (revision 8)

`T4A_CUDA_ABI_REVISION == 8`. Revisions 1-7 are unchanged. No new struct.

```c
void t4a_cuda_initialize_permutation(void* permutation, int64_t n, int64_t len);

void t4a_cuda_column_norms(const void* work, void* norms, int32_t dtype,
                           t4a_cuda_layout work, t4a_cuda_layout norms, int64_t step);

void t4a_cuda_select_pivot(const void* norms, const void* permutation, void* pivots,
                           void* status, int32_t dtype, t4a_cuda_layout norms,
                           t4a_cuda_layout permutation, int64_t step);

void t4a_cuda_swap_columns(void* work, void* permutation, const void* pivots, int32_t dtype,
                           t4a_cuda_layout work, t4a_cuda_layout permutation, int64_t step);

void t4a_cuda_reflector(void* work, void* coeff, int32_t dtype, t4a_cuda_layout work,
                        t4a_cuda_layout coeff, int64_t step);

void t4a_cuda_apply_reflector(const void* packed, void* target, const void* coeff,
                              int32_t dtype, t4a_cuda_layout packed, t4a_cuda_layout target,
                              t4a_cuda_layout coeff, int64_t step, int64_t first_column,
                              int64_t column_count);

void t4a_cuda_extract_r(const void* work, void* r, int32_t dtype, t4a_cuda_layout work,
                        t4a_cuda_layout r);

void t4a_cuda_prefix_rank(const void* r, void* rank, void* status, int32_t dtype,
                          t4a_cuda_layout r, double rtol, double atol);
```

### 4.1 `initialize_permutation`

Flat over `len` entries: `permutation[pos] = pos % n`. No dtype tag: the payload is
`int64_t`. `n >= 1`; the planner rejects `n <= 0` and `len < 0`.

### 4.2 `column_norms`

One block per `(batch, column)` task, `grid = batches * (n - step)`. Task index
`t`: `batch = t / (n - step)`, `column = step + t % (n - step)`. The norm is the
2-norm of `work(step..rows-1, column, batch)` computed with the baseline's max-scale
form (a first pass over the magnitudes for the plane maximum, a second over
`(magnitude / scale)^2`, then `scale * sqrt(plane_sum)`), so no intermediate
overflows. `norms(column, batch)` receives that norm, or `-1` when any magnitude is
nonfinite or the norm itself is nonfinite — the negative sentinel is what
`select_pivot` reads.

`dtype` names the work payload: F32/F64 (norms F32/F64) or C32/C64 (norms F32/F64).
The planner requires `norms.rank == work.rank - 1`, matching batch extents,
`norms.extent[0] == work.extent[1]`, `0 <= step <= n`, and `rows > step` so every
lane participates in the shuffles.

### 4.3 `select_pivot`

One block per batch, `grid = batches`. Lane `l` scans columns `step + l`,
`step + l + 32`, ... below `n`. A negative norm (the sentinel) raises the invalid
flag. The local best is the largest norm, ties broken by the smaller original column
`permutation(column, batch)`. Across lanes the plane maximum of the local best is
taken, then the plane minimum of the original column among lanes holding it, then
the plane minimum of the column among lanes matching that original — the baseline's
deterministic tie rule. Lane 0 writes `pivots[batch]` (the selected column, or `step`
when no lane selected) and sets `status[batch] = 1` when any norm was invalid.

`dtype` names the norms payload (F32/F64). The planner requires `norms.rank >= 1`,
`permutation.rank == norms.rank`, `norms.extent[0] == permutation.extent[0]`, matching
batch extents, `0 <= step <= n`, and `pivots_len >= batches` / `status_len >= batches`
against the batch extent product.

### 4.4 `swap_columns`

Flat over `rows * batches` threads, `block = (256,1,1)`: `row = pos % rows`,
`batch = pos / rows`, `pivot = pivots[batch]`. When `pivot != step` the thread
exchanges `work(row, step, batch)` with `work(row, pivot, batch)`, and the `row == 0`
thread also exchanges `permutation(step, batch)` with `permutation(pivot, batch)`.
The planner requires `work.rank >= 2`, `permutation.rank == work.rank - 1`,
`permutation.extent[0] == work.extent[1]`, matching batch extents,
`permutation_len >= permutation logical count`, and `0 <= step < n`. `pivot`
**values** must lie in `[0, n)` — a raw-call obligation the kernel does not clamp,
exactly as the baseline.

### 4.5 `reflector`

One block per batch, `grid = batches`. Thread 0 computes nothing alone: the whole
plane computes the norm of `work(step..rows-1, step, batch)` with the same max-scale
form, `x0 = work(step, step, batch)`, and

- `norm == 0` → `coeff(step, batch) = 0` and nothing else is written;
- otherwise the phase is `-1` for a negative real `x0` (and `+1` for `x0 >= 0`,
  including NaN) or `x0 / |x0|` for complex (or `1` for a zero `x0`), `alpha = -phase * norm`,
  `denominator = x0 - alpha`, and because lane 0 later overwrites `x0`'s slot,
  `x0` is **broadcast from lane 0 before any store** (`__shfl_sync`), so every lane
  divides by the value read from memory. Every row below `step` is divided by `denominator`,
  and thread 0 writes `work(step, step, batch) = alpha` and
  `coeff(step, batch) = 1 + |x0| / norm`.

The complex division `work(row, step) / denominator` is the ordinary component form,
which is what the baseline does here. The planner requires `work.rank >= 2`,
`coeff.rank == work.rank - 1`, matching batch extents and
`0 <= step < min(rows, cols, coeff.extent[0])`, so neither the work column nor the
coefficient slot can leave its buffer. `coeff_len` must cover the coefficient count.

### 4.6 `apply_reflector`

One block per `(batch, column)` task, `grid = batches * column_count`, `column = first_column + task % column_count`. The plane builds `vector(row) = row == step ? 1 : packed(row, step, batch)`
for `row` in `step..rows-1`, computes `dot = Σ conj(vector) * target(row, column, batch)`,
multiplies it by `coeff(step, batch)` (for complex payloads the two plane sums are
recombined as `re + i * im` and then multiplied by the complex coefficient, matching
the baseline's product), and
subtracts `vector * dot` from the same target positions. `packed_columns` and
`target_columns` come from the two layouts.

The baseline needs a one-element imaginary-unit buffer only because its DSL cannot
build a complex value from two plane sums; the two sums are recombined here directly,
so this ABI has no such operand. The planner requires
`packed.rank == target.rank >= 2`, equal row and batch extents,
`coeff.rank == target.rank - 1` with matching batches and `coeff.extent[0] > step`,
`0 <= step < min(rows, packed columns)` (the reflector vector lives in packed column
`step`), `first_column >= 0`, `column_count >= 0` and
`first_column + column_count <= target.extent[1]` (checked arithmetic). `target` must
be a destination and its write set must be disjoint from `packed`'s reflector column
and from `coeff`; the baseline caller passes distinct buffers, so this slice simply
requires disjoint allocations.

### 4.7 `extract_r`

Flat over `r`'s logical count: `r(row, column, batch) = row <= column ? work(row, column, batch) : 0`,
with `r = [k, n, batch]` and `work = [rows, n, batch]`. The planner requires equal
column and batch extents and `r.rows <= work.rows`, with `r.rows >= 1` when the batch
product is positive.

### 4.8 `prefix_rank`

One block per batch, `grid = batches`. `leading = |r(0, 0, batch)|`,
`threshold = max(rtol * leading, atol)`; lane `l` walks the diagonal `d = l, l+32, ...`
below `k`, raises the invalid flag for a nonfinite `|r(d,d,batch)|`, and records the
smallest `d` with `!(value > threshold)`; the plane minimum gives `first_failure`,
lane 0 writes `rank[batch] = first_failure` (or `k` when the rank is full) and sets
`status[batch] = 1` on any nonfinite value. `k = r.extent[0]`, `n = r.extent[1]`;
`rtol` and `atol` arrive as `double` and are cast to the payload precision, which
preserves every exactly representable tolerance. The threshold is the baseline's
`atol > relative ? atol : relative`, not an `fmax`, so a NaN tolerance propagates the
same way. The planner requires `r.rank >= 2`, `1 <= k <= n`, and
`rank_len >= batches` / `status_len >= batches`.

## 5. Dtype matrix

| Entry | F32 | F64 | I32 | I64 | Bool | C32 | C64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `initialize_permutation` | payload is always `int64_t`; no tag | | | | | | |
| `column_norms` | ✓ (norms F32) | ✓ (F64) | – | – | – | ✓ (F32) | ✓ (F64) |
| `select_pivot` | ✓ | ✓ | – | – | – | – | – |
| `swap_columns` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| `reflector`, `apply_reflector`, `extract_r` | ✓ | ✓ | – | – | – | ✓ | ✓ |
| `prefix_rank` | ✓ | ✓ | – | – | – | ✓ | ✓ |

## 6. Facade

```rust
pub fn plan_initialize_permutation(n: i64, len: i64) -> Result<Launch, Error>;
pub fn plan_column_norms(dtype, work: &Layout, work_len, norms: &Layout, norms_len, step: i64) -> Result<Launch, Error>;
pub fn plan_select_pivot(dtype, norms: &Layout, norms_len, permutation: &Layout, permutation_len, pivots_len, status_len, step: i64) -> Result<Launch, Error>;
pub fn plan_swap_columns(dtype, work: &Layout, work_len, permutation: &Layout, permutation_len, pivots_len: i64, step: i64) -> Result<Launch, Error>;
pub fn plan_reflector(dtype, work: &Layout, work_len, coeff: &Layout, coeff_len, step: i64) -> Result<Launch, Error>;
pub fn plan_apply_reflector(dtype, packed: &Layout, packed_len, target: &Layout, target_len, coeff: &Layout, coeff_len, step: i64, first_column: i64, column_count: i64) -> Result<Launch, Error>;
pub fn plan_extract_r(dtype, work: &Layout, work_len, r: &Layout, r_len) -> Result<Launch, Error>;
pub fn plan_prefix_rank(dtype, r: &Layout, r_len, rank_len: i64, status_len: i64, rtol: f64, atol: f64) -> Result<Launch, Error>;
```

Each returns a `Launch` with `block = (32,1,1)` for the reduction entries and the
flat `(256,1,1)` otherwise, plus the span and shape checks of §4. `Error` needs no
new variant. `batches` is never a planner argument: it is the batch extent product
of the operands, and every flat array is required to hold `batches` words.
`swap_columns` takes `step` and requires `0 <= step < n`. `prefix_rank` casts
`rtol`/`atol` to the payload precision without rejecting non-finite values, so a NaN
tolerance propagates through `atol > relative ? atol : relative` exactly as in the
baseline.

## 7. Tests

Planning: the dtype matrix, every shape/rank/batch rule of §4, the `step` bounds,
`batches` versus the extent product, empty domains, and the block geometry (32 for
the plane entries).

GPU (local A100): `column_norms` against a CPU 2-norm for a batched strided matrix,
including a column of zeros, a column with a huge and a tiny magnitude in the same
column (the max-scale form), a NaN column (`-1` sentinel) and an infinity;
`select_pivot` on hand-built norm arrays covering the tie-break (equal norms, the
smaller original column wins), an invalid sentinel raising `status`, and `step > 0`;
`swap_columns` on a batched matrix with a no-op pivot and a real swap, checking the
permutation; `reflector` for real, C32 and C64 with a zero column, a negative
`x0`, a positive `x0` and a complex phase, checking the reflector's defining
identity `Q v = alpha e_0` numerically; `apply_reflector` applying the built
reflector back (orthogonality: applying it twice restores the operand) for real and
complex, with `first_column > 0` and several columns per block; `extract_r` on a
batched strided work matrix; `prefix_rank` for full rank, zero rank, a rank-limited
case, `rtol`/`atol` dominance, and a NaN diagonal raising `status`; and one physical
launch per call.

## 8. Deliberate departures

- **Widened tie-break domain.** The baseline narrows the original column and the
  diagonal position to `u32` and uses `u32::MAX` as "none". This ABI keeps both in
  `int64_t`: with a permutation value of `2^32` the baseline's cast makes two
  distinct columns compare equal, and a widened domain has no sentinel-collision
  case. Column counts below `2^32` behave identically.

## 9. Deliberate exclusions

- The host step loop, the workspace allocation and the decision when to read
  `status` stay with the consumer.
- `initialize_q` stays revision 7's `identity_shifted`.
- No GPU-graph, NCCL or multi-device work.
