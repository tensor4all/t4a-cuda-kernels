// Bit-identity of the select-form NaN extrema (kf abi.cuh) against the early-return form (3c8d82c).
//
// Fails loudly: every CUDA API call and the launch are checked, and any failure exits 2 before a
// result line is printed. The host mismatch counter starts at a sentinel (not 0) and the kernel
// also counts the pairs it compared, so a download that did not happen, or a launch that did not
// cover every pair, can never print "mismatches 0". Exit 0 only when every pair was compared and
// none differed; exit 1 on a mismatch.
#include "../../../../cuda/include/t4a/abi.cuh"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                                       \
  do {                                                                                         \
    cudaError_t cuda_check_status = (call);                                                    \
    if (cuda_check_status != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA failure at %s:%d: %s -> %s\n", __FILE__, __LINE__, #call,          \
              cudaGetErrorString(cuda_check_status));                                          \
      exit(2);                                                                                 \
    }                                                                                          \
  } while (0)

static const unsigned long long kSentinel = ~0ull;

template <typename T> __device__ T old_max(T l, T r) { if (l != l) return l; if (r != r) return r; return sizeof(T) == 4 ? (T)fmaxf((float)l, (float)r) : (T)fmax((double)l, (double)r); }
template <typename T> __device__ T old_min(T l, T r) { if (l != l) return l; if (r != r) return r; return sizeof(T) == 4 ? (T)fminf((float)l, (float)r) : (T)fmin((double)l, (double)r); }

// counters[0]: mismatching pairs; counters[1]: pairs compared.
template <typename T, typename U> __global__ void check(const U* a, const U* b, long long n, unsigned long long* counters) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) {
    T l, r; memcpy(&l, &a[i], sizeof(T)); memcpy(&r, &b[i], sizeof(T));
    T x[4] = {old_max(l, r), t4a_cuda_nan_max(l, r), old_min(l, r), t4a_cuda_nan_min(l, r)};
    U y[4]; memcpy(y, x, sizeof(x));
    if (y[0] != y[1] || y[2] != y[3]) atomicAdd(&counters[0], 1ull);
    atomicAdd(&counters[1], 1ull);
  }
}

static uint64_t s = 0x9E3779B97F4A7C15ull; static uint64_t rnd() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

// Returns the mismatch count; exits on any CUDA failure or incomplete comparison.
template <typename T, typename U> unsigned long long run(const U* special, int ns, long long nrand) {
  const char* name = sizeof(T) == 4 ? "f32" : "f64";
  long long n = (long long)ns * ns + nrand; U* ha = new U[n]; U* hb = new U[n]; long long k = 0;
  for (int i = 0; i < ns; ++i) for (int j = 0; j < ns; ++j) { ha[k] = special[i]; hb[k] = special[j]; ++k; }
  for (; k < n; ++k) { ha[k] = (U)rnd(); hb[k] = (U)rnd(); if (k % 3 == 0) hb[k] = ha[k] ^ 1; }
  U *da = nullptr, *db = nullptr; unsigned long long* dcounters = nullptr;
  unsigned long long counters[2] = {kSentinel, kSentinel};
  CUDA_CHECK(cudaMalloc(&da, n * sizeof(U)));
  CUDA_CHECK(cudaMalloc(&db, n * sizeof(U)));
  CUDA_CHECK(cudaMalloc(&dcounters, sizeof(counters)));
  CUDA_CHECK(cudaMemcpy(da, ha, n * sizeof(U), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb, n * sizeof(U), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dcounters, 0, sizeof(counters)));
  check<T, U><<<(unsigned)((n + 255) / 256), 256>>>(da, db, n, dcounters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(counters, dcounters, sizeof(counters), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dcounters));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(da));
  delete[] ha; delete[] hb;
  if (counters[0] == kSentinel || counters[1] != (unsigned long long)n) {
    fprintf(stderr, "%s: incomplete comparison: compared %llu of %lld pairs (mismatch counter %llu)\n",
            name, counters[1], n, counters[0]);
    exit(2);
  }
  printf("%s: %lld pairs (%d special^2 + %lld random), compared %llu, mismatches %llu\n", name, n, ns,
         nrand, counters[1], counters[0]);
  return counters[0];
}

int main() {
  const uint32_t s32[] = {0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u, 0x00000001u, 0x80000001u, 0x7f7fffffu, 0xff7fffffu, 0x7fc00000u, 0xffc00000u, 0x7fc00001u, 0x7f800001u, 0xff812345u, 0x7fffffffu};
  const uint64_t s64[] = {0ull, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull, 0x7ff0000000000000ull, 0xfff0000000000000ull, 1ull, 0x8000000000000001ull, 0x7fefffffffffffffull, 0xffefffffffffffffull, 0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff8000000000001ull, 0x7ff0000000000001ull, 0xfff0123456789abcull, 0x7fffffffffffffffull};
  unsigned long long bad = run<float, uint32_t>(s32, 16, 1ll << 26) + run<double, uint64_t>(s64, 16, 1ll << 26);
  printf("total mismatches %llu\n", bad);
  return bad != 0 ? 1 : 0;
}
