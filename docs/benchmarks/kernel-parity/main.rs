//! Paired performance comparison at kernel granularity: tenferro's CubeCL CUDA path
//! at the pinned baseline commit versus `t4a-cuda-kernels` through an equivalent
//! minimal caller.
//!
//! Protocol (frozen in `t4a-cuda-kernels/docs/design/kernel-parity-benchmark.md`
//! before the first run): release build, one process, one GPU, operands uploaded and
//! results validated outside the timed region, 3 warm-up iterations, then the median
//! of 5 repetitions of (a) a batch of 16 enqueues plus one synchronize, reported per
//! operation, (b) one operation plus synchronize, and (c) an empty synchronize.

mod harness;

use std::time::Instant;

use harness::{as_bytes, launch, pub_sync, sentinel, upload, Session, Slot};
use num_complex::Complex64;
use tenferro_gpu::cuda::{upload_tensor, CudaBackend, CudaDeviceId, CudaRuntime};
use tenferro_tensor::{BackendSessionHost, DType, Tensor, TensorRead, TensorWrite};
use t4a_cuda_kernels as kernels;
use t4a_cuda_kernels::{CLayout, DType as KernelDType, Entry, Layout, ReductionPlan, ReduceOp};

const BATCH: usize = 16;
const REPS: usize = 5;
const WARMUP: usize = 3;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
    values[values.len() / 2]
}

/// Median spread over the repetitions, as a fraction of the median.
fn spread(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
    let mid = sorted[sorted.len() / 2];
    (sorted[sorted.len() - 1] - sorted[0]) / mid
}

fn measure(
    backend: &mut CudaBackend,
    mut op: impl FnMut(&mut CudaBackend),
    mut sync_only: impl FnMut(&mut CudaBackend),
) -> (f64, f64, f64, f64) {
    for _ in 0..WARMUP {
        op(backend);
    }
    sync_only(backend);

    let mut batch = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        for _ in 0..BATCH {
            op(backend);
        }
        sync_only(backend);
        batch.push(start.elapsed().as_secs_f64() * 1e9 / BATCH as f64);
    }

    let mut round_trip = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        op(backend);
        sync_only(backend);
        round_trip.push(start.elapsed().as_secs_f64() * 1e9);
    }

    let mut empty = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        sync_only(backend);
        let start = Instant::now();
        sync_only(backend);
        empty.push(start.elapsed().as_secs_f64() * 1e9);
    }

    (median(batch.clone()), median(round_trip), median(empty), spread(&batch))
}

/// Fail loudly when the two arms did not compute the same bytes, so a timing row can
/// never report a launch that did nothing.
fn check_bytes(case: &str, baseline: &[u8], candidate: &[u8]) {
    if baseline != candidate {
        let first = baseline
            .iter()
            .zip(candidate.iter())
            .position(|(lhs, rhs)| lhs != rhs);
        panic!("{case}: arms disagree, first differing byte at {first:?}");
    }
}

/// Fail loudly when a reduction disagrees beyond a relative tolerance.
fn check_close(case: &str, reference: f64, baseline: f64, candidate: f64) {
    let scale = reference.abs().max(1.0);
    for (arm, value) in [("baseline", baseline), ("candidate", candidate)] {
        if (value - reference).abs() > 1e-9 * scale {
            panic!("{case}: {arm} {value} differs from the CPU reference {reference}");
        }
    }
}

struct Row {
    case: &'static str,
    base_b16: f64,
    cand_b16: f64,
    base_rt: f64,
    cand_rt: f64,
    base_spread: f64,
    cand_spread: f64,
    launches: usize,
    note: &'static str,
}

fn f64_vec(len: usize) -> Vec<f64> {
    (0..len).map(|index| ((index % 97) as f64 + 1.0) * 0.001).collect()
}

fn f32_vec(len: usize) -> Vec<f32> {
    (0..len).map(|index| ((index % 97) as f32 + 1.0) * 0.001).collect()
}

fn c64_vec(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| Complex64::new((index % 89) as f64 * 0.01 + 1.0, (index % 7) as f64 * 0.1))
        .collect()
}

fn main() {
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend");
    // A cloned handle keeps the uploads working while `backend` is borrowed mutably
    // for the measured session calls.
    let runtime: CudaRuntime = backend.runtime().clone();
    let session = Session::with_modules(&["arithmetic", "elementwise", "movement", "convert", "reduce"]);
    let mut rows: Vec<Row> = Vec::new();

    // Device clock ramp-up before the first measured case: spin one add launch for
    // 300 ms so every case starts at the sustained clock instead of the idle one.
    {
        let len = 1_000_000usize;
        let lhs_buf = sentinel(&session, len * 4);
        let rhs_buf = sentinel(&session, len * 4);
        let out_buf = sentinel(&session, len * 4);
        let plan = kernels::plan_elementwise(Entry::Add, KernelDType::F32, len as i64).unwrap();
        let function = session.function("t4a_cuda_add");
        let mut slots = [
            Slot::ptr(&out_buf),
            Slot::ptr(&lhs_buf),
            Slot::ptr(&rhs_buf),
            Slot::i32(KernelDType::F32.tag()),
            Slot::i64(len as i64),
        ];
        let start = Instant::now();
        while start.elapsed().as_millis() < 300 {
            for _ in 0..20 {
                launch(function, &plan, session.stream(), &mut slots);
            }
            pub_sync(session.stream());
        }
    }

    // --- 1: add, launch-bound (F32, 1e4) ----------------------------------
    {
        let len = 10_000usize;
        let lhs_host = f32_vec(len);
        let rhs_host = f32_vec(len);
        let out_host = vec![0.0f32; len];
        let lhs = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], lhs_host.clone()).unwrap()).unwrap();
        let rhs = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], rhs_host.clone()).unwrap()).unwrap();
        let mut out = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], out_host.clone()).unwrap()).unwrap();
        let (base_b16, base_rt, _, base_spread) = measure(
            &mut backend,
            |backend| {
                backend
                    .with_backend_session(|s| {
                        s.add_read_into(
                            TensorRead::from_tensor(&lhs),
                            TensorRead::from_tensor(&rhs),
                            TensorWrite::from_tensor(&mut out),
                        )
                        .map(|_| ())
                    })
                    .expect("session")
                    .expect("add");
            },
            |backend| backend.runtime().synchronize().expect("sync"),
        );

        let plan = kernels::plan_elementwise(Entry::Add, KernelDType::F32, len as i64).unwrap();
        let lhs_buf = upload(&session, as_bytes(&lhs_host));
        let rhs_buf = upload(&session, as_bytes(&rhs_host));
        let out_buf = sentinel(&session, len * 4);
        let function = session.function("t4a_cuda_add");
        let mut slots = [
            Slot::ptr(&out_buf),
            Slot::ptr(&lhs_buf),
            Slot::ptr(&rhs_buf),
            Slot::i32(KernelDType::F32.tag()),
            Slot::i64(len as i64),
        ];
        let (cand_b16, cand_rt, _, cand_spread) = measure(
            &mut backend,
            |_| launch(function, &plan, session.stream(), &mut slots),
            |_| pub_sync(session.stream()),
        );
        rows.push(Row {
            case: "add_f32_1e4",
            base_b16,
            cand_b16,
            base_rt,
            cand_rt,
            base_spread,
            cand_spread,
            launches: 1,
            note: "preallocated destination on both arms",
        });
        let base_out = tenferro_gpu::cuda::download_tensor(&runtime, &out).expect("download");
        let cand_out = harness::read_bytes(&session, &out_buf);
        check_bytes(
            "add_f32_1e4",
            as_bytes(base_out.as_slice::<f32>().unwrap()),
            &cand_out[..len * 4],
        );
    }

    // --- 2: add, bandwidth-bound (F32, 1e6) -------------------------------
    {
        let len = 1_000_000usize;
        let lhs_host = f32_vec(len);
        let rhs_host = f32_vec(len);
        let out_host = vec![0.0f32; len];
        let lhs = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], lhs_host.clone()).unwrap()).unwrap();
        let rhs = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], rhs_host.clone()).unwrap()).unwrap();
        let mut out = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], out_host.clone()).unwrap()).unwrap();
        let (base_b16, base_rt, _, base_spread) = measure(
            &mut backend,
            |backend| {
                backend
                    .with_backend_session(|s| {
                        s.add_read_into(
                            TensorRead::from_tensor(&lhs),
                            TensorRead::from_tensor(&rhs),
                            TensorWrite::from_tensor(&mut out),
                        )
                        .map(|_| ())
                    })
                    .expect("session")
                    .expect("add");
            },
            |backend| backend.runtime().synchronize().expect("sync"),
        );
        let plan = kernels::plan_elementwise(Entry::Add, KernelDType::F32, len as i64).unwrap();
        let lhs_buf = upload(&session, as_bytes(&lhs_host));
        let rhs_buf = upload(&session, as_bytes(&rhs_host));
        let out_buf = sentinel(&session, len * 4);
        let function = session.function("t4a_cuda_add");
        let mut slots = [
            Slot::ptr(&out_buf),
            Slot::ptr(&lhs_buf),
            Slot::ptr(&rhs_buf),
            Slot::i32(KernelDType::F32.tag()),
            Slot::i64(len as i64),
        ];
        let (cand_b16, cand_rt, _, cand_spread) = measure(
            &mut backend,
            |_| launch(function, &plan, session.stream(), &mut slots),
            |_| pub_sync(session.stream()),
        );
        rows.push(Row {
            case: "add_f32_1e6",
            base_b16,
            cand_b16,
            base_rt,
            cand_rt,
            base_spread,
            cand_spread,
            launches: 1,
            note: "preallocated destination on both arms",
        });
        let base_out = tenferro_gpu::cuda::download_tensor(&runtime, &out).expect("download");
        let cand_out = harness::read_bytes(&session, &out_buf);
        check_bytes(
            "add_f32_1e6",
            as_bytes(base_out.as_slice::<f32>().unwrap()),
            &cand_out[..len * 4],
        );
    }

    // --- 3: complex add (C64, 1e5) ----------------------------------------
    {
        let len = 100_000usize;
        let lhs_host = c64_vec(len);
        let rhs_host = c64_vec(len);
        let out_host = vec![Complex64::new(0.0, 0.0); len];
        let lhs = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], lhs_host.clone()).unwrap()).unwrap();
        let rhs = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], rhs_host.clone()).unwrap()).unwrap();
        let mut out = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], out_host).unwrap()).unwrap();
        let (base_b16, base_rt, _, base_spread) = measure(
            &mut backend,
            |backend| {
                backend
                    .with_backend_session(|s| {
                        s.add_read_into(
                            TensorRead::from_tensor(&lhs),
                            TensorRead::from_tensor(&rhs),
                            TensorWrite::from_tensor(&mut out),
                        )
                        .map(|_| ())
                    })
                    .expect("session")
                    .expect("add");
            },
            |backend| backend.runtime().synchronize().expect("sync"),
        );
        let plan = kernels::plan_elementwise(Entry::Add, KernelDType::C64, len as i64).unwrap();
        let lhs_buf = upload(&session, as_bytes(&lhs_host));
        let rhs_buf = upload(&session, as_bytes(&rhs_host));
        let out_buf = sentinel(&session, len * 16);
        let function = session.function("t4a_cuda_add");
        let mut slots = [
            Slot::ptr(&out_buf),
            Slot::ptr(&lhs_buf),
            Slot::ptr(&rhs_buf),
            Slot::i32(KernelDType::C64.tag()),
            Slot::i64(len as i64),
        ];
        let (cand_b16, cand_rt, _, cand_spread) = measure(
            &mut backend,
            |_| launch(function, &plan, session.stream(), &mut slots),
            |_| pub_sync(session.stream()),
        );
        rows.push(Row {
            case: "add_c64_1e5",
            base_b16,
            cand_b16,
            base_rt,
            cand_rt,
            base_spread,
            cand_spread,
            launches: 1,
            note: "preallocated destination on both arms",
        });
        let base_out = tenferro_gpu::cuda::download_tensor(&runtime, &out).expect("download");
        let cand_out = harness::read_bytes(&session, &out_buf);
        check_bytes(
            "add_c64_1e5",
            as_bytes(base_out.as_slice::<Complex64>().unwrap()),
            &cand_out[..len * 16],
        );
    }

    // --- 4: multiply (F64, 1e6) -------------------------------------------
    {
        let len = 1_000_000usize;
        let lhs_host = f64_vec(len);
        let rhs_host = f64_vec(len);
        let out_host = vec![0.0f64; len];
        let lhs = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], lhs_host.clone()).unwrap()).unwrap();
        let rhs = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], rhs_host.clone()).unwrap()).unwrap();
        let mut out = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], out_host).unwrap()).unwrap();
        let (base_b16, base_rt, _, base_spread) = measure(
            &mut backend,
            |backend| {
                backend
                    .with_backend_session(|s| {
                        s.mul_read_into(
                            TensorRead::from_tensor(&lhs),
                            TensorRead::from_tensor(&rhs),
                            TensorWrite::from_tensor(&mut out),
                        )
                        .map(|_| ())
                    })
                    .expect("session")
                    .expect("mul");
            },
            |backend| backend.runtime().synchronize().expect("sync"),
        );
        let plan = kernels::plan_elementwise(Entry::Mul, KernelDType::F64, len as i64).unwrap();
        let lhs_buf = upload(&session, as_bytes(&lhs_host));
        let rhs_buf = upload(&session, as_bytes(&rhs_host));
        let out_buf = sentinel(&session, len * 8);
        let function = session.function("t4a_cuda_mul");
        let mut slots = [
            Slot::ptr(&out_buf),
            Slot::ptr(&lhs_buf),
            Slot::ptr(&rhs_buf),
            Slot::i32(KernelDType::F64.tag()),
            Slot::i64(len as i64),
        ];
        let (cand_b16, cand_rt, _, cand_spread) = measure(
            &mut backend,
            |_| launch(function, &plan, session.stream(), &mut slots),
            |_| pub_sync(session.stream()),
        );
        rows.push(Row {
            case: "mul_f64_1e6",
            base_b16,
            cand_b16,
            base_rt,
            cand_rt,
            base_spread,
            cand_spread,
            launches: 1,
            note: "preallocated destination on both arms",
        });
        let base_out = tenferro_gpu::cuda::download_tensor(&runtime, &out).expect("download");
        let cand_out = harness::read_bytes(&session, &out_buf);
        check_bytes(
            "mul_f64_1e6",
            as_bytes(base_out.as_slice::<f64>().unwrap()),
            &cand_out[..len * 8],
        );
    }

    // --- 5: compact copy (F64, 1e6) ---------------------------------------
    {
        let len = 1_000_000usize;
        let src_host = f64_vec(len);
        let src = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], src_host.clone()).unwrap()).unwrap();
        let mut dst = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], vec![0.0f64; len]).unwrap()).unwrap();
        let (base_b16, base_rt, _, base_spread) = measure(
            &mut backend,
            |backend| {
                backend
                    .with_backend_session(|s| {
                        s.copy_read_into(
                            TensorRead::from_tensor(&src),
                            TensorWrite::from_tensor(&mut dst),
                        )
                        .map(|_| ())
                    })
                    .expect("session")
                    .expect("copy");
            },
            |backend| backend.runtime().synchronize().expect("sync"),
        );
        let plan = kernels::plan_copy_exact(KernelDType::F64, len as i64).unwrap();
        let src_buf = upload(&session, as_bytes(&src_host));
        let dst_buf = sentinel(&session, len * 8);
        let function = session.function("t4a_cuda_copy_exact");
        let mut slots = [
            Slot::ptr(&dst_buf),
            Slot::ptr(&src_buf),
            Slot::i32(KernelDType::F64.tag()),
            Slot::i64(len as i64),
        ];
        let (cand_b16, cand_rt, _, cand_spread) = measure(
            &mut backend,
            |_| launch(function, &plan, session.stream(), &mut slots),
            |_| pub_sync(session.stream()),
        );
        rows.push(Row {
            case: "copy_f64_1e6",
            base_b16,
            cand_b16,
            base_rt,
            cand_rt,
            base_spread,
            cand_spread,
            launches: 1,
            note: "preallocated destination on both arms",
        });
        let base_out = tenferro_gpu::cuda::download_tensor(&runtime, &dst).expect("download");
        let cand_out = harness::read_bytes(&session, &dst_buf);
        check_bytes(
            "copy_f64_1e6",
            as_bytes(base_out.as_slice::<f64>().unwrap()),
            &cand_out[..len * 8],
        );
    }

    // --- 6: conversion F64 -> F32 (1e6) -----------------------------------
    {
        let len = 1_000_000usize;
        let src_host = f64_vec(len);
        let src = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], src_host.clone()).unwrap()).unwrap();
        let (base_b16, base_rt, _, base_spread) = measure(
            &mut backend,
            |backend| {
                backend
                    .with_backend_session(|s| s.cast(&src, DType::F32).map(|_| ()))
                    .expect("session")
                    .expect("convert");
            },
            |backend| backend.runtime().synchronize().expect("sync"),
        );
        let plan = kernels::plan_convert(KernelDType::F32, KernelDType::F64, len as i64).unwrap();
        let src_buf = upload(&session, as_bytes(&src_host));
        let dst_buf = sentinel(&session, len * 4);
        let function = session.function("t4a_cuda_convert");
        let mut slots = [
            Slot::ptr(&dst_buf),
            Slot::ptr(&src_buf),
            Slot::i32(KernelDType::F32.tag()),
            Slot::i32(KernelDType::F64.tag()),
            Slot::i64(len as i64),
        ];
        let (cand_b16, cand_rt, _, cand_spread) = measure(
            &mut backend,
            |_| launch(function, &plan, session.stream(), &mut slots),
            |_| pub_sync(session.stream()),
        );
        rows.push(Row {
            case: "convert_f64_f32_1e6",
            base_b16,
            cand_b16,
            base_rt,
            cand_rt,
            base_spread,
            cand_spread,
            launches: 1,
            note: "baseline allocates its output; candidate writes a preallocated buffer",
        });
    }

    // --- 7-8: reductions -------------------------------------------------
    for (case, extents, axis, out_extents) in [
        ("reduce_sum_f64_1e6_axis0", vec![1_000_000i64], 0usize, vec![1i64]),
        (
            "reduce_sum_f64_1000x1000_axis1",
            vec![1000i64, 1000],
            1usize,
            // Keepdims: the reduce axis keeps extent 1.
            vec![1000i64, 1],
        ),
    ] {
        let len: usize = extents.iter().map(|&value| value as usize).product();
        let out_len: usize = out_extents.iter().map(|&value| value as usize).product();
        let src_host = f64_vec(len);
        let src = upload_tensor(
            &runtime,
            &Tensor::from_vec_col_major(
                extents
                    .iter()
                    .map(|&value| value as usize)
                    .collect::<Vec<usize>>(),
                src_host.clone(),
            )
            .unwrap(),
        )
        .unwrap();
        let (base_b16, base_rt, _, base_spread) = measure(
            &mut backend,
            |backend| {
                backend
                    .with_backend_session(|s| {
                        s.reduce_sum_read(TensorRead::from_tensor(&src), &[axis])
                            .map(|_| ())
                    })
                    .expect("session")
                    .expect("reduce");
            },
            |backend| backend.runtime().synchronize().expect("sync"),
        );

        let out_layout = Layout::compact(&out_extents).unwrap();
        let in_layout = Layout::compact(&extents).unwrap();
        let plan = kernels::plan_reduction(
            KernelDType::F64,
            ReduceOp::Sum,
            &out_layout,
            out_len as i64,
            &in_layout,
            len as i64,
            axis as u32,
            32,
        )
        .unwrap();
        let src_buf = upload(&session, as_bytes(&src_host));
        let out_buf = sentinel(&session, out_len * 8);
        let out_slot = Slot::layout(&CLayout::new(&out_layout));
        let in_slot = Slot::layout(&CLayout::new(&in_layout));
        let (cand_b16, cand_rt, launches, cand_spread) = match &plan {
            ReductionPlan::Direct { strategy, launch: direct } => {
                let function = session.function("t4a_cuda_reduce");
                let mut slots = [
                    Slot::ptr(&out_buf),
                    Slot::ptr(&src_buf),
                    Slot::i32(KernelDType::F64.tag()),
                    Slot::i32(ReduceOp::Sum.tag()),
                    Slot::i32(strategy.tag()),
                    out_slot,
                    in_slot,
                    Slot::i32(axis as i32),
                    Slot::i64(out_len as i64),
                ];
                let (a, b, _, d) = measure(
                    &mut backend,
                    |_| launch(function, direct, session.stream(), &mut slots),
                    |_| pub_sync(session.stream()),
                );
                (a, b, 1usize, d)
            }
            ReductionPlan::Split {
                split,
                pass1,
                combine_strategy,
                combine_op,
                pass2,
            } => {
                let partials_layout =
                    kernels::reduction_partials_layout(&in_layout, axis as u32, split.splits)
                        .unwrap();
                let partials_len = partials_layout.element_count().unwrap();
                let partials_buf = sentinel(&session, partials_len as usize * 8);
                let function_partials = session.function("t4a_cuda_reduce_partials");
                let function_final = session.function("t4a_cuda_reduce");
                let mut slots1 = [
                    Slot::ptr(&partials_buf),
                    Slot::ptr(&src_buf),
                    Slot::i32(KernelDType::F64.tag()),
                    Slot::i32(ReduceOp::Sum.tag()),
                    Slot::layout(&CLayout::new(&partials_layout)),
                    in_slot,
                    Slot::i32(axis as i32),
                    Slot::i64(out_len as i64),
                    Slot::i64(split.splits),
                    Slot::i64(split.chunk_len),
                ];
                let mut slots2 = [
                    Slot::ptr(&out_buf),
                    Slot::ptr(&partials_buf),
                    Slot::i32(KernelDType::F64.tag()),
                    Slot::i32(combine_op.tag()),
                    Slot::i32(combine_strategy.tag()),
                    out_slot,
                    Slot::layout(&CLayout::new(&partials_layout)),
                    Slot::i32(axis as i32),
                    Slot::i64(out_len as i64),
                ];
                let (a, b, _, d) = measure(
                    &mut backend,
                    |_| {
                        launch(function_partials, pass1, session.stream(), &mut slots1);
                        launch(function_final, pass2, session.stream(), &mut slots2);
                    },
                    |_| pub_sync(session.stream()),
                );
                (a, b, 2usize, d)
            }
        };
        rows.push(Row {
            case,
            base_b16,
            cand_b16,
            base_rt,
            cand_rt,
            base_spread,
            cand_spread,
            launches,
            note: "baseline allocates its output; candidate writes a preallocated buffer",
        });

        // Validate both arms against a CPU reference for every output. The baseline
        // arm allocates inside the timed region, so one extra call provides the
        // tensor to download here.
        let base_out = backend
            .with_backend_session(|s| s.reduce_sum_read(TensorRead::from_tensor(&src), &[axis]))
            .expect("session")
            .expect("reduce");
        runtime.synchronize().expect("sync");
        let base_out = tenferro_gpu::cuda::download_tensor(&runtime, &base_out).expect("download");
        let base_values = base_out.as_slice::<f64>().unwrap();
        let cand_bytes = harness::read_bytes(&session, &out_buf);
        let cand_values = harness::from_bytes::<f64>(&cand_bytes[..out_len * 8]);
        for index in 0..out_len {
            let reference = if extents.len() == 1 {
                src_host.iter().sum::<f64>()
            } else {
                // Column-major: output `index` is the row coordinate, and the
                // reduction runs over the trailing axis.
                (0..extents[1] as usize)
                    .map(|column| src_host[index + column * extents[0] as usize])
                    .sum::<f64>()
            };
            check_close(case, reference, base_values[index], cand_values[index]);
        }
    }

    // --- 9: reduce max (F32, 513) -----------------------------------------
    {
        let len = 513usize;
        let src_host = f32_vec(len);
        let src = upload_tensor(&runtime, &Tensor::from_vec_col_major(vec![len], src_host.clone()).unwrap()).unwrap();
        let (base_b16, base_rt, _, base_spread) = measure(
            &mut backend,
            |backend| {
                backend
                    .with_backend_session(|s| {
                        s.reduce_max_read(TensorRead::from_tensor(&src), &[0])
                            .map(|_| ())
                    })
                    .expect("session")
                    .expect("reduce max");
            },
            |backend| backend.runtime().synchronize().expect("sync"),
        );
        let out_layout = Layout::compact(&[1]).unwrap();
        let in_layout = Layout::compact(&[len as i64]).unwrap();
        let plan = kernels::plan_reduction(
            KernelDType::F32,
            ReduceOp::Max,
            &out_layout,
            1,
            &in_layout,
            len as i64,
            0,
            32,
        )
        .unwrap();
        let src_buf = upload(&session, as_bytes(&src_host));
        let out_buf = sentinel(&session, 4);
        let launched = match &plan {
            ReductionPlan::Direct {
                strategy,
                launch: direct,
            } => {
                let function = session.function("t4a_cuda_reduce");
                let mut slots = [
                    Slot::ptr(&out_buf),
                    Slot::ptr(&src_buf),
                    Slot::i32(KernelDType::F32.tag()),
                    Slot::i32(ReduceOp::Max.tag()),
                    Slot::i32(strategy.tag()),
                    Slot::layout(&CLayout::new(&out_layout)),
                    Slot::layout(&CLayout::new(&in_layout)),
                    Slot::i32(0),
                    Slot::i64(1),
                ];
                let (a, b, _, d) = measure(
                    &mut backend,
                    |_| launch(function, direct, session.stream(), &mut slots),
                    |_| pub_sync(session.stream()),
                );
                (a, b, 1usize, d)
            }
            ReductionPlan::Split {
                split,
                pass1,
                combine_strategy,
                combine_op,
                pass2,
            } => {
                let partials_layout =
                    kernels::reduction_partials_layout(&in_layout, 0, split.splits).unwrap();
                let partials_len = partials_layout.element_count().unwrap();
                let partials_buf = sentinel(&session, partials_len as usize * 4);
                let function_partials = session.function("t4a_cuda_reduce_partials");
                let function_final = session.function("t4a_cuda_reduce");
                let mut slots1 = [
                    Slot::ptr(&partials_buf),
                    Slot::ptr(&src_buf),
                    Slot::i32(KernelDType::F32.tag()),
                    Slot::i32(ReduceOp::Max.tag()),
                    Slot::layout(&CLayout::new(&partials_layout)),
                    Slot::layout(&CLayout::new(&in_layout)),
                    Slot::i32(0),
                    Slot::i64(1),
                    Slot::i64(split.splits),
                    Slot::i64(split.chunk_len),
                ];
                let mut slots2 = [
                    Slot::ptr(&out_buf),
                    Slot::ptr(&partials_buf),
                    Slot::i32(KernelDType::F32.tag()),
                    Slot::i32(combine_op.tag()),
                    Slot::i32(combine_strategy.tag()),
                    Slot::layout(&CLayout::new(&out_layout)),
                    Slot::layout(&CLayout::new(&partials_layout)),
                    Slot::i32(0),
                    Slot::i64(1),
                ];
                let (a, b, _, d) = measure(
                    &mut backend,
                    |_| {
                        launch(function_partials, pass1, session.stream(), &mut slots1);
                        launch(function_final, pass2, session.stream(), &mut slots2);
                    },
                    |_| pub_sync(session.stream()),
                );
                (a, b, 2usize, d)
            }
        };
        rows.push(Row {
            case: "reduce_max_f32_513_axis0",
            base_b16,
            cand_b16: launched.0,
            base_rt,
            cand_rt: launched.1,
            base_spread,
            cand_spread: launched.3,
            launches: launched.2,
            note: "baseline allocates its output; candidate writes a preallocated buffer",
        });
    }

    println!(
        "{:<32} {:>11} {:>11} {:>8} {:>11} {:>11} {:>8} {:>6}  {}",
        "case", "base_b16_ns", "cand_b16_ns", "ratio", "base_rt_ns", "cand_rt_ns", "spread", "launch", "note"
    );
    for row in &rows {
        println!(
            "{:<32} {:>11.0} {:>11.0} {:>8.2} {:>11.0} {:>11.0} {:>7.0}% {:>6}  {}",
            row.case,
            row.base_b16,
            row.cand_b16,
            row.cand_b16 / row.base_b16,
            row.base_rt,
            row.cand_rt,
            (row.cand_spread.max(row.base_spread)) * 100.0,
            row.launches,
            row.note
        );
    }
    let _ = (DType::F32, pub_sync as fn(_));
}
