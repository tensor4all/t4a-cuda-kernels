//! Kernel-time harness for the vectorized flat entries (tenferro-rs#2529).
//!
//! ```text
//! cargo run --release --features gpu-tests --example bench_flat_vector -- <case> <width> [len]
//! ```
//!
//! - `case`: `add_f64`, `exp_f64`, `tanh_f64`, `clamp_f64`, `compare_gt_f64`,
//!   `add_f32`, `exp_f32`.
//! - `width`: `1` plans the scalar geometry (alignment 1); `auto` plans with the
//!   operands' real address alignment, which is what a consumer passing its
//!   allocation bases gets.
//! - `len`: element count, default `1 << 20` (the tenferro-benchmark
//!   `gpu/isolated_ops/*_n1048576` rows).
//!
//! Protocol: 10 warm-ups, `cuProfilerStart`, 200 back-to-back launches with no
//! allocation, transfer or host wait between them, one stream synchronize,
//! `cuProfilerStop`. Run it under
//! `nsys profile --capture-range=cudaProfilerApi --trace=cuda` and read the
//! per-launch kernel durations from the `cuda_gpu_trace` report. The output of the
//! last launch is checked against a CPU reference after the range.
//!
//! `T4A_CUDA_SOURCE_ROOT=<checkout>` compiles the kernels from another checkout.
//! With `width 1` and a revision-13 checkout this measures the pre-vectorization
//! kernel with the same binary and geometry; a revision-13 entry declares one
//! parameter fewer, and the driver reads only the parameters the kernel declares.
//!
//! `clamp_f64` takes per-element bounds, because that is the entry's ABI: it reads
//! four arrays, where PyTorch's `clamp(x, min=scalar, max=scalar)` reads one.

#[path = "../tests/gpu/mod.rs"]
mod gpu;

use gpu::{download, launch_async, pub_sync, upload, written, Buf, Session, Slot};
use t4a_cuda_kernels::{
    address_alignment, plan_arithmetic, plan_clamp, plan_compare, plan_unary, CompareOp, DType,
    Entry, FlatLaunch, UnaryOp,
};

const WARMUPS: usize = 10;
const ITERS: usize = 200;

type ProfilerFn = unsafe extern "C" fn(u32) -> i32;

fn profiler(name: &[u8]) -> i32 {
    // `cuProfilerStart`/`cuProfilerStop` are not bound by cudarc; resolve them from
    // the already-loaded driver library, as `bench_kernels` does.
    let lib = unsafe { cudarc::driver::sys::culib() };
    let function = *unsafe { lib.get::<ProfilerFn>(name) }.expect("driver profiler symbol");
    unsafe { function(0) }
}

fn inputs(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn address(buf: &Buf) -> u64 {
    buf.as_ptr() as u64
}

fn close(actual: f64, expected: f64) -> bool {
    actual == expected || (actual - expected).abs() <= 1e-6 * expected.abs().max(1e-30)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let case = args.first().map(String::as_str).unwrap_or("add_f64");
    let auto = match args.get(1).map(String::as_str).unwrap_or("auto") {
        "1" => false,
        "auto" => true,
        other => panic!("width must be 1 or auto, got {other}"),
    };
    let len: usize = args.get(2).map_or(1 << 20, |n| n.parse().expect("len"));
    let n = len as i64;
    let dtype = match case {
        "add_f32" | "exp_f32" => DType::F32,
        _ => DType::F64,
    };
    let source = if case.starts_with("add") {
        "arithmetic"
    } else {
        "elementwise"
    };
    let session = Session::new(source);

    let a = inputs(len, 1);
    let b = inputs(len, 2);
    let upload_values = |values: &[f64]| -> Buf {
        if dtype == DType::F32 {
            upload(
                &session,
                &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            )
        } else {
            upload(&session, values)
        }
    };
    let a_buf = upload_values(&a);
    let b_buf = upload_values(&b);
    let lower = vec![-0.2f64; len];
    let upper = vec![0.3f64; len];
    let lower_buf = upload(&session, &lower);
    let upper_buf = upload(&session, &upper);
    let out_bytes = if case == "compare_gt_f64" {
        len
    } else {
        len * dtype.size()
    };
    let out = written(&session, &vec![0u8; out_bytes]);

    let alignment_of = |buffers: &[&Buf]| {
        if auto {
            address_alignment(&buffers.iter().map(|b| address(b)).collect::<Vec<_>>())
        } else {
            1
        }
    };
    let (function, plan, mut slots): (&str, FlatLaunch, Vec<Slot>) = match case {
        "add_f64" | "add_f32" => {
            let plan = plan_arithmetic(Entry::Add, dtype, n, alignment_of(&[&out, &a_buf, &b_buf]))
                .unwrap();
            (
                Entry::Add.name(),
                plan,
                vec![
                    Slot::ptr(&out),
                    Slot::ptr(&a_buf),
                    Slot::ptr(&b_buf),
                    Slot::i32(dtype.tag()),
                    Slot::i64(n),
                    Slot::i32(plan.vector_width as i32),
                ],
            )
        }
        "exp_f64" | "tanh_f64" | "exp_f32" => {
            let op = if case.starts_with("tanh") {
                UnaryOp::Tanh
            } else {
                UnaryOp::Exp
            };
            let plan = plan_unary(op, dtype, dtype, n, alignment_of(&[&out, &a_buf])).unwrap();
            (
                "t4a_cuda_unary",
                plan,
                vec![
                    Slot::ptr(&out),
                    Slot::ptr(&a_buf),
                    Slot::i32(dtype.tag()),
                    Slot::i32(dtype.tag()),
                    Slot::i32(op.tag()),
                    Slot::i64(n),
                    Slot::i32(plan.vector_width as i32),
                ],
            )
        }
        "clamp_f64" => {
            let plan = plan_clamp(
                dtype,
                n,
                alignment_of(&[&out, &a_buf, &lower_buf, &upper_buf]),
            )
            .unwrap();
            (
                "t4a_cuda_clamp",
                plan,
                vec![
                    Slot::ptr(&out),
                    Slot::ptr(&a_buf),
                    Slot::ptr(&lower_buf),
                    Slot::ptr(&upper_buf),
                    Slot::i32(dtype.tag()),
                    Slot::i64(n),
                    Slot::i32(plan.vector_width as i32),
                ],
            )
        }
        "compare_gt_f64" => {
            let plan = plan_compare(dtype, n, alignment_of(&[&out, &a_buf, &b_buf])).unwrap();
            (
                "t4a_cuda_compare",
                plan,
                vec![
                    Slot::ptr(&out),
                    Slot::ptr(&a_buf),
                    Slot::ptr(&b_buf),
                    Slot::i32(dtype.tag()),
                    Slot::i32(CompareOp::Gt.tag()),
                    Slot::i64(n),
                    Slot::i32(plan.vector_width as i32),
                ],
            )
        }
        other => panic!("unknown case {other}"),
    };
    let kernel = session.function(function);
    println!(
        "BENCH case={case} len={len} width={} grid={} block={} source_root={}",
        plan.vector_width,
        plan.launch.grid[0],
        plan.launch.block[0],
        std::env::var("T4A_CUDA_SOURCE_ROOT").unwrap_or_else(|_| "self".into())
    );

    for _ in 0..WARMUPS {
        launch_async(kernel, &plan.launch, session.stream(), &mut slots);
    }
    pub_sync(session.stream());
    assert_eq!(profiler(b"cuProfilerStart\0"), 0, "cuProfilerStart");
    for _ in 0..ITERS {
        launch_async(kernel, &plan.launch, session.stream(), &mut slots);
    }
    pub_sync(session.stream());
    assert_eq!(profiler(b"cuProfilerStop\0"), 0, "cuProfilerStop");

    // Reference check of the last launch, outside the measured range.
    let bad = match case {
        "compare_gt_f64" => {
            let got: Vec<u8> = download(&session, &out, len);
            (0..len)
                .filter(|&i| got[i] != u8::from(a[i] > b[i]))
                .count()
        }
        _ => {
            let got: Vec<f64> = if dtype == DType::F32 {
                download::<f32>(&session, &out, len)
                    .into_iter()
                    .map(f64::from)
                    .collect()
            } else {
                download(&session, &out, len)
            };
            let expected = |i: usize| match case {
                "add_f64" | "add_f32" => a[i] + b[i],
                "exp_f64" | "exp_f32" => a[i].exp(),
                "tanh_f64" => a[i].tanh(),
                _ => a[i].clamp(lower[i], upper[i]),
            };
            (0..len).filter(|&i| !close(got[i], expected(i))).count()
        }
    };
    assert_eq!(
        bad, 0,
        "{case}: {bad} elements differ from the CPU reference"
    );
    println!("CHECK case={case} ok");
}
