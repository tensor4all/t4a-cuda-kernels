// Old-vs-new comparison of the reduce combine entries (docs/design/slice-combine-load-ahead.md).
//
// Loads two cubins of cuda/src/reduce.cu (the baseline and the candidate), launches
// `t4a_cuda_reduce` and `t4a_cuda_reduce_any` with strategy PLANE on identical inputs, and
// compares every output element bit for bit. Each floating component that differs is classified:
//   - value:   at least one side is not NaN and the bits differ (an in-contract difference);
//   - nanness: one side NaN, the other not (an in-contract difference);
//   - payload: both NaN, different bits (outside the ABI contract, c-kernel-abi.md "NaN").
// Integers have only the `value` class.
//
// Exit codes: 0 every planned output was compared and no value/nanness difference was found;
// 1 a value/nanness difference; 2 any CUDA failure or an incomplete comparison.
//
// Build: g++ -O2 -std=c++17 -I/usr/local/cuda-12.6/include combine-old-vs-new.cpp
//        -L/usr/local/cuda-12.6/lib64/stubs -lcuda -o combine-old-vs-new
// Run:   ./combine-old-vs-new old.cubin new.cubin

#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kMaxRank = 8;
struct Layout {
  int rank;
  int reserved;
  long long offset;
  long long extents[kMaxRank];
  long long strides[kMaxRank];
};
struct LayoutAny {
  int rank;
  int reserved;
  long long offset;
  const long long* extents;
  const long long* strides;
};
static_assert(sizeof(Layout) == 144, "t4a_cuda_layout");
static_assert(sizeof(LayoutAny) == 32, "t4a_cuda_layout_any");

enum DType { F32 = 0, F64 = 1, I32 = 2, I64 = 3, C32 = 5, C64 = 6 };
enum Op { SUM = 0, PROD = 1, SUM_SQUARES = 2, MAX = 3, MIN = 4 };
constexpr int kPlane = 1;

void check(CUresult status, const char* what, int line) {
  if (status != CUDA_SUCCESS) {
    const char* name = nullptr;
    cuGetErrorName(status, &name);
    std::fprintf(stderr, "CUDA failure at line %d: %s -> %s\n", line, what,
                 name ? name : "unknown");
    std::exit(2);
  }
}
#define CK(call) check((call), #call, __LINE__)

// splitmix64
struct Rng {
  uint64_t state;
  uint64_t next() {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  double unit() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int scalar_bytes(DType dtype) {
  return dtype == F32 || dtype == I32 || dtype == C32 ? 4 : 8;
}
int element_bytes(DType dtype) {
  return dtype == C32 || dtype == C64 ? 2 * scalar_bytes(dtype) : scalar_bytes(dtype);
}
bool floating(DType dtype) { return dtype != I32 && dtype != I64; }
const char* dtype_name(DType dtype) {
  switch (dtype) {
    case F32: return "F32";
    case F64: return "F64";
    case I32: return "I32";
    case I64: return "I64";
    case C32: return "C32";
    case C64: return "C64";
  }
  return "?";
}
const char* op_name(int op) {
  static const char* names[] = {"SUM", "PROD", "SUM_SQUARES", "MAX", "MIN"};
  return names[op];
}

// One floating scalar under input `mode`:
//   0 near 1 (products stay finite; contraction-sensitive complex products),
//   1 mixture of finite magnitudes and specials, 2 raw random bits,
//   3 near 1 with sparse NaNs of random payloads (several may meet in one lane),
//   4 near-overflow magnitudes with random signs, 5 wide exponents (overflow/underflow mid-walk).
uint64_t float_bits(Rng& rng, bool f32, int mode) {
  auto from = [&](double v) -> uint64_t {
    if (f32) {
      float x = static_cast<float>(v);
      uint32_t b;
      std::memcpy(&b, &x, 4);
      return b;
    }
    uint64_t b;
    std::memcpy(&b, &v, 8);
    return b;
  };
  auto nan_bits = [&]() -> uint64_t {
    uint64_t r = rng.next();
    if (f32) return 0x7f800000u | (r & 0x807fffffu) | 1u;  // any sign, nonzero mantissa
    return 0x7ff0000000000000ull | (r & 0x800fffffffffffffull) | 1ull;
  };
  const double max_finite = f32 ? 3.4028234663852886e38 : 1.7976931348623157e308;
  switch (mode) {
    case 0: return from(1.0 + (rng.unit() - 0.5) * 0.01);
    case 1: {
      uint64_t pick = rng.next() % 16;
      if (pick == 0) return from(0.0);
      if (pick == 1) return from(-0.0);
      if (pick == 2) return from(rng.unit() < 0.5 ? 1.0 / 0.0 : -1.0 / 0.0);
      if (pick == 3) return nan_bits();
      if (pick == 4) return f32 ? 1u : 1ull;  // smallest subnormal
      if (pick == 5) return from(rng.unit() < 0.5 ? max_finite : -max_finite);
      return from((rng.unit() - 0.5) * 1e3);
    }
    case 2: return f32 ? (rng.next() & 0xffffffffu) : rng.next();
    case 3: return rng.next() % 61 == 0 ? nan_bits() : from(1.0 + (rng.unit() - 0.5) * 0.01);
    case 4: return from((rng.unit() < 0.5 ? -1.0 : 1.0) * max_finite * (0.5 + 0.5 * rng.unit()) / 16);
    default: {
      int span = f32 ? 25 : 200;
      int e = static_cast<int>((rng.unit() - 0.5) * 2 * span);
      double m = 1.0 + rng.unit();
      double v = m;
      for (int i = 0; i < (e < 0 ? -e : e); ++i) v = e < 0 ? v * 0.5 : v * 2.0;
      return from(v);
    }
  }
}

struct Counts {
  long long compared = 0, value = 0, nanness = 0, payload = 0;
};

bool is_nan_bits(uint64_t b, bool f32) {
  if (f32) return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0;
  return (b & 0x7ff0000000000000ull) == 0x7ff0000000000000ull && (b & 0x000fffffffffffffull) != 0;
}

void classify(const uint8_t* a, const uint8_t* b, DType dtype, Counts& counts) {
  const int sb = scalar_bytes(dtype);
  const int components = element_bytes(dtype) / sb;
  for (int c = 0; c < components; ++c) {
    uint64_t x = 0, y = 0;
    std::memcpy(&x, a + c * sb, sb);
    std::memcpy(&y, b + c * sb, sb);
    if (x == y) continue;
    if (!floating(dtype)) { ++counts.value; continue; }
    const bool f32 = sb == 4;
    const bool xn = is_nan_bits(x, f32), yn = is_nan_bits(y, f32);
    if (xn && yn) ++counts.payload;
    else if (xn != yn) ++counts.nanness;
    else ++counts.value;
  }
}

struct Module {
  CUmodule module;
  CUfunction reduce, reduce_any;
};

Module load(const char* path) {
  FILE* file = std::fopen(path, "rb");
  if (!file) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
  std::vector<char> image;
  char buffer[65536];
  size_t got;
  while ((got = std::fread(buffer, 1, sizeof buffer, file)) > 0) image.insert(image.end(), buffer, buffer + got);
  std::fclose(file);
  Module m;
  CK(cuModuleLoadData(&m.module, image.data()));
  CK(cuModuleGetFunction(&m.reduce, m.module, "t4a_cuda_reduce"));
  CK(cuModuleGetFunction(&m.reduce_any, m.module, "t4a_cuda_reduce_any"));
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: %s old.cubin new.cubin\n", argv[0]); return 2; }
  CK(cuInit(0));
  CUdevice device;
  CK(cuDeviceGet(&device, 0));
  CUcontext context;
  CK(cuDevicePrimaryCtxRetain(&context, device));
  CK(cuCtxSetCurrent(context));
  Module sides[2] = {load(argv[1]), load(argv[2])};

  struct Route { DType dtype; int op; };
  const Route routes[] = {
      {F32, SUM}, {F64, SUM}, {I32, SUM}, {I64, SUM}, {C32, SUM}, {C64, SUM},
      {F32, PROD}, {F64, PROD}, {I32, PROD}, {I64, PROD}, {C32, PROD}, {C64, PROD},
      {F32, SUM_SQUARES}, {F64, SUM_SQUARES},
      {F32, MAX}, {F64, MAX}, {I32, MAX}, {I64, MAX},
      {F32, MIN}, {F64, MIN}, {I32, MIN}, {I64, MIN},
  };
  // (axis length, plane width): the planner's combines and direct-plane shapes.
  const long long geometries[][2] = {
      {4096, 32}, {4096, 64}, {4096, 128}, {4096, 256}, {4096, 512}, {3001, 32}, {3001, 512},
      {1000, 128}, {513, 512}, {512, 512}, {511, 32}, {100, 16}, {100, 8}, {40, 32}, {32, 32},
  };
  // 0 axis-major [len, outputs], 1 strided [outputs, len] over axis 1, 2 reversed axis.
  const int kinds = 3;
  const long long outputs = 48;  // 8 outputs per input mode

  long long planned = 0;
  Counts total;
  Rng rng{0x2406};
  for (const Route& route : routes) {
    Counts counts;
    const int eb = element_bytes(route.dtype);
    for (const auto& geometry : geometries) {
      const long long len = geometry[0];
      const int width = static_cast<int>(geometry[1]);
      for (int kind = 0; kind < kinds; ++kind) {
        // Host input: output `o` uses mode `o % 6`; integers use raw bits.
        std::vector<uint8_t> logical(static_cast<size_t>(len * outputs) * eb);
        for (long long o = 0; o < outputs; ++o) {
          for (long long k = 0; k < len; ++k) {
            uint8_t* element = &logical[static_cast<size_t>(o * len + k) * eb];
            const int sb = scalar_bytes(route.dtype);
            for (int c = 0; c < eb / sb; ++c) {
              uint64_t bits = floating(route.dtype)
                                  ? float_bits(rng, sb == 4, static_cast<int>(o % 6))
                                  : rng.next();
              std::memcpy(element + c * sb, &bits, sb);
            }
          }
        }
        Layout in{}, out{};
        in.rank = 2;
        out.rank = 2;
        std::vector<uint8_t> host(logical.size());
        int axis = 0;
        auto place = [&](long long o, long long k, long long at) {
          std::memcpy(&host[static_cast<size_t>(at) * eb], &logical[static_cast<size_t>(o * len + k) * eb], eb);
        };
        if (kind == 0) {
          in.extents[0] = len; in.extents[1] = outputs; in.strides[0] = 1; in.strides[1] = len;
          for (long long o = 0; o < outputs; ++o) for (long long k = 0; k < len; ++k) place(o, k, k + o * len);
        } else if (kind == 1) {
          axis = 1;
          in.extents[0] = outputs; in.extents[1] = len; in.strides[0] = 1; in.strides[1] = outputs;
          for (long long o = 0; o < outputs; ++o) for (long long k = 0; k < len; ++k) place(o, k, o + k * outputs);
        } else {
          in.extents[0] = len; in.extents[1] = outputs; in.strides[0] = -1; in.strides[1] = len;
          in.offset = len - 1;
          for (long long o = 0; o < outputs; ++o) for (long long k = 0; k < len; ++k) place(o, k, (len - 1 - k) + o * len);
        }
        for (int d = 0; d < 2; ++d) out.extents[d] = d == axis ? 1 : in.extents[d];
        out.strides[0] = 1;
        out.strides[1] = out.extents[0];

        CUdeviceptr src, geometry_buf;
        CK(cuMemAlloc(&src, host.size()));
        CK(cuMemcpyHtoD(src, host.data(), host.size()));
        long long geom[8] = {out.extents[0], out.extents[1], out.strides[0], out.strides[1],
                             in.extents[0], in.extents[1], in.strides[0], in.strides[1]};
        CK(cuMemAlloc(&geometry_buf, sizeof geom));
        CK(cuMemcpyHtoD(geometry_buf, geom, sizeof geom));
        LayoutAny out_any{2, 0, out.offset, reinterpret_cast<const long long*>(geometry_buf),
                          reinterpret_cast<const long long*>(geometry_buf + 16)};
        LayoutAny in_any{2, 0, in.offset, reinterpret_cast<const long long*>(geometry_buf + 32),
                         reinterpret_cast<const long long*>(geometry_buf + 48)};

        for (int general = 0; general < 2; ++general) {
          std::vector<uint8_t> results[2];
          for (int side = 0; side < 2; ++side) {
            CUdeviceptr dst;
            const size_t bytes = static_cast<size_t>(outputs) * eb;
            CK(cuMemAlloc(&dst, bytes));
            CK(cuMemsetD8(dst, 0xCD, bytes));
            int dtype = route.dtype, op = route.op, strategy = kPlane;
            long long output_len = outputs;
            void* args[] = {&dst, &src, &dtype, &op, &strategy,
                            general ? static_cast<void*>(&out_any) : static_cast<void*>(&out),
                            general ? static_cast<void*>(&in_any) : static_cast<void*>(&in),
                            &axis, &output_len};
            CUfunction function = general ? sides[side].reduce_any : sides[side].reduce;
            CK(cuLaunchKernel(function, static_cast<unsigned>(outputs), 1, 1, width, 1, 1, 0,
                              nullptr, args, nullptr));
            CK(cuCtxSynchronize());
            results[side].resize(bytes);
            CK(cuMemcpyDtoH(results[side].data(), dst, bytes));
            CK(cuMemFree(dst));
          }
          for (long long o = 0; o < outputs; ++o) {
            classify(&results[0][o * eb], &results[1][o * eb], route.dtype, counts);
            ++counts.compared;
          }
          planned += outputs;
        }
        CK(cuMemFree(src));
        CK(cuMemFree(geometry_buf));
      }
    }
    std::printf("%-4s %-12s outputs %6lld  value %lld  nanness %lld  payload-only %lld\n",
                dtype_name(route.dtype), op_name(route.op), counts.compared, counts.value,
                counts.nanness, counts.payload);
    total.compared += counts.compared;
    total.value += counts.value;
    total.nanness += counts.nanness;
    total.payload += counts.payload;
  }
  std::printf("total outputs %lld (planned %lld)  value %lld  nanness %lld  payload-only %lld\n",
              total.compared, planned, total.value, total.nanness, total.payload);
  if (total.compared != planned || planned == 0) return 2;
  return total.value + total.nanness == 0 ? 0 : 1;
}
