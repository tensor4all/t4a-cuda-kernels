"""Apply one mutant to cuda/src/reduce.cu in the #29 checkout, run the golden test, restore."""
import subprocess, sys, os
REPO = "/home/shinaoka/work/t4a-cuda-kernels-combine"; SRC = f"{REPO}/cuda/src/reduce.cu"
MUTANTS = {
 # #29's combine walk: fold each batch in reverse order (loads unchanged).
 "combine-reverse-fold": ("""#pragma unroll
    for (int b = 0; b < B; ++b) {
      if (index + b * stride < axis_len) {
        acc = fold(acc, batch[b]);""", """#pragma unroll
    for (int b = B - 1; b >= 0; --b) {
      if (index + b * stride < axis_len) {
        acc = fold(acc, batch[b]);"""),
 # Item 3: the lane ignores the plane (warp) offset inside a multi-warp block.
 "pass1-lane-not-per-warp": ("""          static_cast<int>(threadIdx.x & 31u)};""",
                             """          static_cast<int>(threadIdx.x)};"""),
 # Item 3: every warp of a block reduces the block's first cube.
 "pass1-cube-ignores-warp": ("""  const unsigned int cube = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);""",
                             """  const unsigned int cube = blockIdx.x * (blockDim.x >> 5);"""),
 # Pass 1: swap the operands of the batch fold (same value for sum, different NaN payload).
 "pass1-operand-swap": ("""    acc = fold.combine(acc, fold.combine(low, high));""",
                        """    acc = fold.combine(fold.combine(low, high), acc);"""),
 # Extrema: swap max's operands in the lane fold (changes NaN and signed-zero selection).
 "max-operand-swap": ("""    return t4a_cuda_max<T>(lhs, rhs);
  }
  template <typename T>
  __device__ __forceinline__ T combine(T lhs, T rhs) const {
    return t4a_cuda_max<T>(lhs, rhs);""", """    return t4a_cuda_max<T>(rhs, lhs);
  }
  template <typename T>
  __device__ __forceinline__ T combine(T lhs, T rhs) const {
    return t4a_cuda_max<T>(lhs, rhs);"""),
}
name = sys.argv[1]; old, new = MUTANTS[name]
orig = open(SRC).read(); assert orig.count(old) == 1, name
try:
    open(SRC, "w").write(orig.replace(old, new))
    env = dict(os.environ, PATH="/usr/local/cuda-12.6/bin:" + os.environ["PATH"])
    r = subprocess.run(["cargo", "test", "-j8", "--release", "--features", "gpu-tests", "--test",
                        "gpu_reduce_golden", "--", "--nocapture"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
finally:
    open(SRC, "w").write(orig)
assert subprocess.run(["git", "diff", "--quiet", "cuda/src/reduce.cu"], cwd=REPO).returncode == 0
print(f"### mutant {name}: exit {r.returncode}\n" + "\n".join(l for l in out.splitlines() if "Compiling" not in l))
