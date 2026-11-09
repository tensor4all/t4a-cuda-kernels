"""PyTorch arm of the flat vector-width kernel-time measurement (tenferro-rs#2529).

usage: python torch_flat.py <case> [len]

Same protocol as examples/bench_flat_vector.rs: 10 warm-ups, cudaProfilerStart,
200 back-to-back calls into a preallocated destination, one synchronize,
cudaProfilerStop. The calls are the tenferro-benchmark gpu/isolated_ops forms
(scripts/gpu_public_api_ops.py): clamp takes 0-d bounds there (`clamp_f64`), and
`clamp_full_f64` is the per-element-bound form t4a_cuda_clamp's ABI requires.
"""
import sys

import torch

case = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 20
dtype = torch.float32 if case.endswith("_f32") else torch.float64
g = torch.Generator(device="cuda").manual_seed(1)
x = torch.rand(n, dtype=dtype, device="cuda", generator=g) * 2 - 1
y = torch.rand(n, dtype=dtype, device="cuda", generator=g) * 2 - 1
lo0 = torch.tensor(-0.2, dtype=dtype, device="cuda")
hi0 = torch.tensor(0.3, dtype=dtype, device="cuda")
lo = torch.full((n,), -0.2, dtype=dtype, device="cuda")
hi = torch.full((n,), 0.3, dtype=dtype, device="cuda")
out = torch.empty(n, dtype=torch.bool if case.startswith("compare") else dtype, device="cuda")
calls = {
    "add_f64": lambda: torch.add(x, y, out=out),
    "add_f32": lambda: torch.add(x, y, out=out),
    "exp_f64": lambda: torch.exp(x, out=out),
    "exp_f32": lambda: torch.exp(x, out=out),
    "tanh_f64": lambda: torch.tanh(x, out=out),
    "clamp_f64": lambda: torch.clamp(x, min=lo0, max=hi0, out=out),
    "clamp_full_f64": lambda: torch.clamp(x, min=lo, max=hi, out=out),
    "compare_gt_f64": lambda: torch.gt(x, y, out=out),
}
call = calls[case]
for _ in range(10):
    call()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(200):
    call()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(f"TORCH case={case} len={n} torch={torch.__version__} cuda={torch.version.cuda}")
