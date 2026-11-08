// Bit-identity of the select-form NaN extrema (kf abi.cuh) against the early-return form (3c8d82c).
//
// Every CUDA call goes through CUDA_CHECK. Before that, the return values were ignored, so a run
// that could not allocate, copy or launch still printed "mismatches 0" and exited 0: an evidence
// program that passed without comparing anything. --legacy keeps that old behaviour runnable, so
// the hole can be measured instead of described.
#include "../../../../cuda/include/t4a/abi.cuh"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static bool legacy = false;

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    cudaError_t err_ = (call);                                                      \
    if (!legacy && err_ != cudaSuccess) {                                           \
      std::fprintf(stderr, "FAILED %s:%d %s -> %s\n", __FILE__, __LINE__, #call,    \
                   cudaGetErrorString(err_));                                       \
      std::exit(2);                                                                 \
    }                                                                               \
  } while (0)

template <typename T> __device__ T old_max(T l, T r) { if (l != l) return l; if (r != r) return r; return sizeof(T) == 4 ? (T)fmaxf((float)l, (float)r) : (T)fmax((double)l, (double)r); }
template <typename T> __device__ T old_min(T l, T r) { if (l != l) return l; if (r != r) return r; return sizeof(T) == 4 ? (T)fminf((float)l, (float)r) : (T)fmin((double)l, (double)r); }
template <typename T, typename U> __global__ void check(const U* a, const U* b, long long n, unsigned long long* bad) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return;
  T l, r; memcpy(&l, &a[i], sizeof(T)); memcpy(&r, &b[i], sizeof(T));
  T x[4] = {old_max(l, r), t4a_cuda_nan_max(l, r), old_min(l, r), t4a_cuda_nan_min(l, r)};
  U y[4]; memcpy(y, x, sizeof(x));
  if (y[0] != y[1] || y[2] != y[3]) atomicAdd(bad, 1ull);
}
static uint64_t s = 0x9E3779B97F4A7C15ull; static uint64_t rnd() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

// copy_fails copies from a null host pointer: cudaMemcpy rejects it with cudaErrorInvalidValue
// without reading anything. (A size one past the allocation would overrun the host buffer
// instead of failing, so it is not a test of the failure path.)
template <typename T, typename U> unsigned long long run(const U* special, int ns, long long nrand, bool copy_fails) {
  long long n = (long long)ns * ns + nrand;
  U *da = nullptr, *db = nullptr; unsigned long long *dbad = nullptr, bad = 0;
  const size_t bytes = (size_t)n * sizeof(U);
  // Device allocations come first: with a large --n this fails here, on cudaMalloc, and nothing
  // touches an uninitialised pointer afterwards.
  CUDA_CHECK(cudaMalloc(&da, bytes));
  CUDA_CHECK(cudaMalloc(&db, bytes));
  CUDA_CHECK(cudaMalloc(&dbad, sizeof bad));
  U* ha = new U[n]; U* hb = new U[n]; long long k = 0;
  for (int i = 0; i < ns; ++i) for (int j = 0; j < ns; ++j) { ha[k] = special[i]; hb[k] = special[j]; ++k; }
  for (; k < n; ++k) { ha[k] = (U)rnd(); hb[k] = (U)rnd(); if (k % 3 == 0) hb[k] = ha[k] ^ 1; }
  CUDA_CHECK(cudaMemcpy(da, copy_fails ? nullptr : ha, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dbad, 0, sizeof bad));
  check<T, U><<<(n + 255) / 256, 256>>>(da, db, n, dbad);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&bad, dbad, sizeof bad, cudaMemcpyDeviceToHost));
  printf("%s: %lld pairs (%d special^2 + %lld random), mismatches %llu\n", sizeof(T) == 4 ? "f32" : "f64", n, ns, nrand, bad);
  CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dbad));
  delete[] ha; delete[] hb;
  return bad;
}
int main(int argc, char** argv) {
  long long nrand = 1ll << 26; bool copy_fails = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--legacy")) legacy = true;
    else if (!strcmp(argv[i], "--copy-fails")) copy_fails = true;
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) nrand = atoll(argv[++i]);
    else { fprintf(stderr, "usage: %s [--legacy] [--copy-fails] [--n <random pairs>]\n", argv[0]); return 2; }
  }
  // 2^44 pairs is 128 TB per buffer: far past any device, and small enough that the byte count
  // and the special-value square below cannot overflow.
  if (nrand < 0 || nrand > (1ll << 44)) { fprintf(stderr, "--n out of range: 0 to 2^44\n"); return 2; }
  printf("mode %s, %lld random pairs per type, copy %s\n", legacy ? "legacy (every CUDA return value ignored)" : "checked",
         nrand, copy_fails ? "from a null pointer" : "as allocated");
  const uint32_t s32[] = {0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u, 0x00000001u, 0x80000001u, 0x7f7fffffu, 0xff7fffffu, 0x7fc00000u, 0xffc00000u, 0x7fc00001u, 0x7f800001u, 0xff812345u, 0x7fffffffu};
  const uint64_t s64[] = {0ull, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull, 0x7ff0000000000000ull, 0xfff0000000000000ull, 1ull, 0x8000000000000001ull, 0x7fefffffffffffffull, 0xffefffffffffffffull, 0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff8000000000001ull, 0x7ff0000000000001ull, 0xfff0123456789abcull, 0x7fffffffffffffffull};
  unsigned long long bad = run<float, uint32_t>(s32, 16, nrand, copy_fails) + run<double, uint64_t>(s64, 16, nrand, copy_fails);
  printf("total mismatches %llu\n", bad); return bad != 0;
}
