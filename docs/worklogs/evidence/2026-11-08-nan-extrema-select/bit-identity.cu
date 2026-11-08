// Bit-identity of the select-form NaN extrema (kf abi.cuh) against the early-return form (3c8d82c).
#include "../../../../cuda/include/t4a/abi.cuh"
#include <cstdio>
#include <cstdint>
#include <cstring>
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
template <typename T, typename U> unsigned long long run(const U* special, int ns, long long nrand) {
  long long n = (long long)ns * ns + nrand; U* ha = new U[n]; U* hb = new U[n]; long long k = 0;
  for (int i = 0; i < ns; ++i) for (int j = 0; j < ns; ++j) { ha[k] = special[i]; hb[k] = special[j]; ++k; }
  for (; k < n; ++k) { ha[k] = (U)rnd(); hb[k] = (U)rnd(); if (k % 3 == 0) hb[k] = ha[k] ^ 1; }
  U *da, *db; unsigned long long *dbad, bad = 0; cudaMalloc(&da, n * sizeof(U)); cudaMalloc(&db, n * sizeof(U)); cudaMalloc(&dbad, 8);
  cudaMemcpy(da, ha, n * sizeof(U), cudaMemcpyHostToDevice); cudaMemcpy(db, hb, n * sizeof(U), cudaMemcpyHostToDevice); cudaMemset(dbad, 0, 8);
  check<T, U><<<(n + 255) / 256, 256>>>(da, db, n, dbad); cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost);
  printf("%s: %lld pairs (%d special^2 + %lld random), mismatches %llu, cuda %s\n", sizeof(T) == 4 ? "f32" : "f64", n, ns, nrand, bad, cudaGetErrorString(cudaGetLastError()));
  return bad;
}
int main() {
  const uint32_t s32[] = {0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u, 0x00000001u, 0x80000001u, 0x7f7fffffu, 0xff7fffffu, 0x7fc00000u, 0xffc00000u, 0x7fc00001u, 0x7f800001u, 0xff812345u, 0x7fffffffu};
  const uint64_t s64[] = {0ull, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull, 0x7ff0000000000000ull, 0xfff0000000000000ull, 1ull, 0x8000000000000001ull, 0x7fefffffffffffffull, 0xffefffffffffffffull, 0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff8000000000001ull, 0x7ff0000000000001ull, 0xfff0123456789abcull, 0x7fffffffffffffffull};
  unsigned long long bad = run<float, uint32_t>(s32, 16, 1ll << 26) + run<double, uint64_t>(s64, 16, 1ll << 26);
  printf("total mismatches %llu\n", bad); return bad != 0;
}
