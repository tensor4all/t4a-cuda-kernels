import sys
p, which = sys.argv[1], sys.argv[2]
s = open(p).read()
muts = {
 "drop-tail": ("  if (lane < len - tail_start) {\n    acc = fold(acc, chunk[tail_start + lane]);\n  }\n", ""),
 "batch-overlap": ("    for (int b = 0; b < B; ++b) {\n      batch[b] = t4a_cuda_load_group<T, V, Aligned>(chunk, index + b * stride);", "    for (int b = 0; b < B; ++b) {\n      batch[b] = t4a_cuda_load_group<T, V, Aligned>(chunk, index + b);"),
 "first-batch-short": ("  for (int b = 1; b < B; ++b) {\n    if (lane + b * stride < groups) {", "  for (int b = 1; b < B - 1; ++b) {\n    if (lane + b * stride < groups) {"),
 "last-batch-short": ("  for (int b = 0; b < B - 1; ++b) {\n    if (index + b * stride < groups) {\n      acc", "  for (int b = 0; b < B - 2; ++b) {\n    if (index + b * stride < groups) {\n      acc"),
 "aligned-test-too-weak": ("& 15ull) == 0) {", "& 3ull) == 0) {"),
 "scalar-group-load-wrong": ("      x.lane[c] = chunk[group * V + c];", "      x.lane[c] = chunk[group * V];"),
 "prod-regrouped": ("struct ProdFold {\n  static constexpr bool kRegroup = false;", "struct ProdFold {\n  static constexpr bool kRegroup = true;"),
}
old, new = muts[which]
assert s.count(old) == 1, which
open(p, 'w').write(s.replace(old, new))
