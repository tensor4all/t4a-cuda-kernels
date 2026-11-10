// Probe driver: launches the instrumented t4a_cuda_reduce_partials (F32 SUM) with one-plane
// and 4-plane blocks and dumps (cube, t0, t1, smid) of the last of 20 launches.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "reduce_probe.cu"
static void ck(cudaError_t e, const char* w) { if (e) { std::printf("%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); } }
int main(int argc, char** argv) {
  // args: name n0 n1 axis splits planes out
  const char* name = argv[1]; long long e0 = atoll(argv[2]), e1 = atoll(argv[3]);
  int axis = atoi(argv[4]); long long splits = atoll(argv[5]); int planes = atoi(argv[6]);
  long long n = e0 * e1, axis_len = axis == 0 ? e0 : e1, outputs = n / axis_len;
  long long chunk = axis_len / splits, cubes = outputs * splits;
  t4a_cuda_layout in{}, pl{};
  in.rank = 2; in.extents[0] = e0; in.extents[1] = e1; in.strides[0] = 1; in.strides[1] = e0;
  pl.rank = 2; pl.extents[0] = axis == 0 ? splits : e0; pl.extents[1] = axis == 1 ? splits : e1;
  pl.strides[0] = 1; pl.strides[1] = pl.extents[0];
  float* x; float* p; ck(cudaMalloc(&x, n * 4), "malloc"); ck(cudaMalloc(&p, cubes * 4), "malloc");
  std::vector<float> h(n); for (long long i = 0; i < n; ++i) h[i] = (i % 7) * 0.25f;
  ck(cudaMemcpy(x, h.data(), n * 4, cudaMemcpyHostToDevice), "copy");
  dim3 block(32 * planes), grid((cubes + planes - 1) / planes);
  for (int it = 0; it < 20; ++it)
    t4a_cuda_reduce_partials<<<grid, block>>>(p, x, 0, 0, pl, in, axis, outputs, splits, chunk);
  ck(cudaDeviceSynchronize(), "run");
  std::vector<unsigned long long> g(3 * cubes);
  ck(cudaMemcpyFromSymbol(g.data(), g_probe, 3 * cubes * 8), "probe");
  FILE* f = std::fopen(argv[7], "w");
  std::fprintf(f, "# %s cubes=%lld chunk=%lld planes=%d grid=%u block=%u\n", name, cubes, chunk, planes, grid.x, block.x);
  for (long long c = 0; c < cubes; ++c) std::fprintf(f, "%lld %llu %llu %llu\n", c, g[3*c], g[3*c+1], g[3*c+2]);
  std::fclose(f); return 0;
}
