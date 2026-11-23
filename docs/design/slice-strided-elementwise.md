# Slice plan: layout-taking elementwise entries (revision 15)

Consumer: tenferro-rs#2701. The tenferro CUDA engine reads an operand that is a view (a transposed eager value, a
broadcast reduced row, a zero-stride constant) by staging it into a compact copy, because `t4a_cuda_unary`,
`t4a_cuda_binary`, `t4a_cuda_add/sub/mul/div`, `t4a_cuda_compare` and `t4a_cuda_select` address `in[index]`. That
copy is an allocation and a launch inside every timed call. Revision 15 removes the need for it.

## Entries

| Entry | Mirrors | Dtypes |
| --- | --- | --- |
| `t4a_cuda_strided_unary` | `t4a_cuda_unary` (every `UnaryOp`, same dtype pairs) | as `UnaryOp::accepts` |
| `t4a_cuda_strided_binary` | `t4a_cuda_add/sub/mul/div`, `t4a_cuda_binary` (`MAX`, `MIN`, `POW`, `REM`, `DIV_COMPLEX`, checked integer forms) | per `StridedBinaryOp::accepts` |
| `t4a_cuda_strided_compare` | `t4a_cuda_compare` | F32/F64/I32/I64 |
| `t4a_cuda_strided_select` | `t4a_cuda_select` | F32/F64/I32/I64 |

Each has an `_any` twin (revision 9 general path). Arguments: the flat entry's pointers, one `t4a_cuda_layout` per
buffer (destination first), then the flat entry's scalar arguments. `StridedBinaryOp` is a new tag space rather than
`BinaryOp` reused, because the flat set splits addition, subtraction, multiplication and real division across four
entries and distinguishes real from complex division by tag; one strided entry needs one tag per operation.

## Classification (review checklist section 1)

- Runtime values: extents, strides, offsets of every layout; `len`; buffer addresses.
- Operation attributes: `dtype` (or `dst_dtype`/`src_dtype`), `op`/`cmp`. Runtime tags, as in revision 6; nothing is
  folded into a compiled variant.
- Algorithm configuration: none. Launch geometry is the flat `(ceil(len/256),1,1) x (256,1,1)`.

## Parallelism (section 2)

One thread per logical destination element; `len = product(out.extents)`; the planner derives the grid from it. Each
thread decodes its flat index through the destination extents (axis 0 fastest, the revision-6 `combined_strided` walk)
and computes every operand address in the same loop. No barrier, no shared memory, no per-thread loop over the tensor.

## Numerics

Every element is computed by the same device value function the flat entry calls, so non-NaN results are
bit-identical. Two refinements were needed and are part of the revision:

1. `t4a_cuda_mul` for complex operands left the fused-multiply-add choice to the compiler, which fused the flat and the
   strided body differently. `t4a_cuda_mul_elementwise` spells the contraction out in the form the flat entry already
   compiled to, and both entries use it.
2. A NaN result's sign and payload follow operand position in `DMUL`/`DFMA`, and the compiler may commute a product's
   operands; the ABI leaves them unspecified (as the revision 14 golden already does for reductions). Tests compare
   bits, treating NaN components as equal to NaN components.

## Validation

Same as `plan_combined_strided`: the destination must be a destination, every operand covers the destination extents,
every buffer span (including the lowest address a negative stride reaches) lies inside the caller's allocation length.

## Scalar-side `ADD` / `SUB` on `t4a_cuda_binary`

Tags 14 and 15, `ScalarSide::Left`/`Right` only (the elementwise forms stay with `t4a_cuda_add`/`t4a_cuda_sub`), all six
numeric dtypes, per element `t4a_cuda_add`/`t4a_cuda_sub`. A compact tensor plus a resident one-element constant then
takes the flat kernel with no layout.

## Tests

`tests/gpu_strided_elementwise.rs`: every accepted `(op, dtype)` on permuted, padded-and-offset, broadcast and
reversed operands into compact or permuted destinations, against the flat entry on host-gathered compact copies; the
status word of the checked forms; one 103 000-element case per family; rank 9 through the `_any` symbols and the `_any`
symbols at rank 3; scalar-side `ADD`/`SUB` against `t4a_cuda_add`/`t4a_cuda_sub` on a materialized splat, both sides,
all dtypes, special values as the scalar. `tests/planning.rs`: acceptance and rejection of the planners.
