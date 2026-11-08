// Bit-identity of the select-form NaN extrema (kf abi.cuh) against the early-return form (3c8d82c).
//
// Every CUDA call goes through CUDA_CHECK. Before that, the return values were ignored, so a run
// that could not allocate, copy or launch still printed "mismatches 0" and exited 0: an evidence
// program that passed without comparing anything. --legacy keeps that old behaviour runnable, so
// the hole can be measured instead of described.
//
// In checked mode the host counters also start at a sentinel, not 0, and the kernel counts the
// pairs it compared: a download that did not happen, or a launch that did not cover every pair,
// exits 2 instead of printing "mismatches 0". Exit codes: 0 = every pair compared and none
// differed, 1 = mismatch, 2 = CUDA failure, incomplete comparison or bad arguments.
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

static const unsigned long long kSentinel = ~0ull;

template <typename T> __device__ T old_max(T l, T r) { if (l != l) return l; if (r != r) return r; return sizeof(T) == 4 ? (T)fmaxf((float)l, (float)r) : (T)fmax((double)l, (double)r); }
template <typename T> __device__ T old_min(T l, T r) { if (l != l) return l; if (r != r) return r; return sizeof(T) == 4 ? (T)fminf((float)l, (float)r) : (T)fmin((double)l, (double)r); }

// counters[0]: mismatching pairs; counters[1]: pairs compared.
template <typename T, typename U> __global__ void check(const U* a, const U* b, long long n, unsigned long long* counters) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return;
  T l, r; memcpy(&l, &a[i], sizeof(T)); memcpy(&r, &b[i], sizeof(T));
  T x[4] = {old_max(l, r), t4a_cuda_nan_max(l, r), old_min(l, r), t4a_cuda_nan_min(l, r)};
  U y[4]; memcpy(y, x, sizeof(x));
  if (y[0] != y[1] || y[2] != y[3]) atomicAdd(&counters[0], 1ull);
  atomicAdd(&counters[1], 1ull);
}
static uint64_t s = 0x9E3779B97F4A7C15ull; static uint64_t rnd() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

// copy_fails copies from a null host pointer: cudaMemcpy rejects it with cudaErrorInvalidValue
// without reading anything. (A size one past the allocation would overrun the host buffer
// instead of failing, so it is not a test of the failure path.)
template <typename T, typename U> unsigned long long run(const U* special, int ns, long long nrand, bool copy_fails) {
  const char* name = sizeof(T) == 4 ? "f32" : "f64";
  long long n = (long long)ns * ns + nrand;
  U *da = nullptr, *db = nullptr; unsigned long long* dcounters = nullptr;
  // The legacy harness started its counter at 0; the checked one starts at the sentinel.
  unsigned long long counters[2] = {legacy ? 0ull : kSentinel, legacy ? 0ull : kSentinel};
  const size_t bytes = (size_t)n * sizeof(U);
  // Device allocations come first: with a large --n this fails here, on cudaMalloc, and nothing
  // touches an uninitialised pointer afterwards.
  CUDA_CHECK(cudaMalloc(&da, bytes));
  CUDA_CHECK(cudaMalloc(&db, bytes));
  CUDA_CHECK(cudaMalloc(&dcounters, sizeof counters));
  U* ha = new U[n]; U* hb = new U[n]; long long k = 0;
  for (int i = 0; i < ns; ++i) for (int j = 0; j < ns; ++j) { ha[k] = special[i]; hb[k] = special[j]; ++k; }
  for (; k < n; ++k) { ha[k] = (U)rnd(); hb[k] = (U)rnd(); if (k % 3 == 0) hb[k] = ha[k] ^ 1; }
  CUDA_CHECK(cudaMemcpy(da, copy_fails ? nullptr : ha, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dcounters, 0, sizeof counters));
  check<T, U><<<(unsigned)((n + 255) / 256), 256>>>(da, db, n, dcounters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(counters, dcounters, sizeof counters, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dcounters));
  delete[] ha; delete[] hb;
  if (legacy) {
    printf("%s: %lld pairs (%d special^2 + %lld random), mismatches %llu\n", name, n, ns, nrand, counters[0]);
    return counters[0];
  }
  if (counters[0] == kSentinel || counters[1] != (unsigned long long)n) {
    fprintf(stderr, "%s: incomplete comparison: compared %llu of %lld pairs (mismatch counter %llu)\n",
            name, counters[1], n, counters[0]);
    exit(2);
  }
  printf("%s: %lld pairs (%d special^2 + %lld random), compared %llu, mismatches %llu\n", name, n, ns,
         nrand, counters[1], counters[0]);
  return counters[0];
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
  printf("total mismatches %llu\n", bad); return bad != 0 ? 1 : 0;
}
