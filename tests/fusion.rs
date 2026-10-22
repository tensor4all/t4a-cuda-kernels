//! Host-side validation of the fused-expression emitter. Needs no CUDA.

use t4a_cuda_kernels::{
    artifact_key, emit_fusion, plan_fused, DType, Error, FusionInstruction, FusionOp,
    FusionOperand, FusionPlan, ABI_REVISION, FLAT_BLOCK, MAX_FUSION_INSTRUCTIONS,
};

/// The eligibility matrix the pinned baseline's `classify.rs` states, written out
/// independently so the test cannot agree with the implementation by construction.
fn expected_eligible(op: FusionOp, dtype: DType) -> bool {
    let real = matches!(dtype, DType::F32 | DType::F64);
    let complex = matches!(dtype, DType::C32 | DType::C64);
    match op {
        FusionOp::Add | FusionOp::Mul | FusionOp::Div | FusionOp::Neg => real || complex,
        FusionOp::Conj => complex,
        _ => real,
    }
}

const ALL_OPS: [FusionOp; 20] = [
    FusionOp::Add,
    FusionOp::Mul,
    FusionOp::Div,
    FusionOp::Neg,
    FusionOp::Abs,
    FusionOp::Conj,
    FusionOp::Max,
    FusionOp::Min,
    FusionOp::Clamp,
    FusionOp::Exp,
    FusionOp::Log,
    FusionOp::Sin,
    FusionOp::Cos,
    FusionOp::Tanh,
    FusionOp::Sqrt,
    FusionOp::Rsqrt,
    FusionOp::Pow,
    FusionOp::Expm1,
    FusionOp::Log1p,
    FusionOp::Erf,
];

fn instruction(op: FusionOp, operands: &[FusionOperand]) -> FusionInstruction {
    FusionInstruction {
        op,
        operands: operands.to_vec(),
    }
}

/// A plan whose single output is `op(a0, a1, a2)` over three inputs.
fn single_op(op: FusionOp) -> FusionPlan {
    let operands = (0..op.arity() as u32)
        .map(FusionOperand::Input)
        .collect::<Vec<FusionOperand>>();
    FusionPlan {
        dtype: DType::F32,
        inputs: op.arity() as u32,
        instructions: vec![instruction(op, &operands)],
        outputs: vec![FusionOperand::Value(0)],
    }
}

#[test]
fn eligibility_matches_the_pinned_matrix() {
    for op in ALL_OPS {
        for dtype in DType::ALL {
            let plan = FusionPlan {
                dtype,
                ..single_op(op)
            };
            let result = emit_fusion(&plan);
            if expected_eligible(op, dtype) {
                assert!(result.is_ok(), "{op:?} should be eligible for {dtype:?}");
            } else {
                assert_eq!(
                    result,
                    Err(Error::FusionEligibility),
                    "{op:?} must be rejected for {dtype:?}"
                );
            }
        }
    }
    // The complex dtypes keep only the algebraic operations, as the baseline does.
    for op in [
        FusionOp::Exp,
        FusionOp::Tanh,
        FusionOp::Erf,
        FusionOp::Clamp,
    ] {
        assert!(!op.accepts(DType::C64));
    }
    for op in [
        FusionOp::Add,
        FusionOp::Mul,
        FusionOp::Div,
        FusionOp::Neg,
        FusionOp::Conj,
    ] {
        assert!(op.accepts(DType::C32) && op.accepts(DType::C64));
    }
}

#[test]
fn arity_operands_and_bounds_are_validated() {
    let bad_arity = FusionPlan {
        dtype: DType::F32,
        inputs: 2,
        instructions: vec![instruction(FusionOp::Add, &[FusionOperand::Input(0)])],
        outputs: vec![FusionOperand::Value(0)],
    };
    assert_eq!(emit_fusion(&bad_arity), Err(Error::FusionArity));

    let missing_input = FusionPlan {
        dtype: DType::F32,
        inputs: 1,
        instructions: vec![instruction(FusionOp::Neg, &[FusionOperand::Input(1)])],
        outputs: vec![FusionOperand::Value(0)],
    };
    assert_eq!(emit_fusion(&missing_input), Err(Error::FusionOperand));

    // A value operand must name an *earlier* instruction.
    let forward = FusionPlan {
        dtype: DType::F32,
        inputs: 1,
        instructions: vec![
            instruction(FusionOp::Neg, &[FusionOperand::Value(1)]),
            instruction(FusionOp::Neg, &[FusionOperand::Input(0)]),
        ],
        outputs: vec![FusionOperand::Value(1)],
    };
    assert_eq!(emit_fusion(&forward), Err(Error::FusionOperand));

    let empty_outputs = FusionPlan {
        dtype: DType::F32,
        inputs: 1,
        instructions: vec![instruction(FusionOp::Neg, &[FusionOperand::Input(0)])],
        outputs: vec![],
    };
    assert_eq!(emit_fusion(&empty_outputs), Err(Error::FusionEmpty));

    let no_inputs = FusionPlan {
        dtype: DType::F32,
        inputs: 0,
        instructions: vec![],
        outputs: vec![FusionOperand::Input(0)],
    };
    assert_eq!(emit_fusion(&no_inputs), Err(Error::FusionEmpty));

    // A plan with no instructions still validates its dtype: an instruction list is not
    // needed to emit a four-byte access over a one-byte Bool buffer.
    for dtype in [DType::I32, DType::I64, DType::Bool] {
        let identity = FusionPlan {
            dtype,
            inputs: 1,
            instructions: vec![],
            outputs: vec![FusionOperand::Input(0)],
        };
        assert_eq!(emit_fusion(&identity), Err(Error::FusionEligibility));
    }

    // The kernel-parameter budget bounds the operand count independently of the
    // instruction count.
    let wide = FusionPlan {
        dtype: DType::F32,
        inputs: 600,
        instructions: vec![],
        outputs: (0..300).map(FusionOperand::Input).collect(),
    };
    assert_eq!(emit_fusion(&wide), Err(Error::FusionTooLarge));
    // Just inside the budget: 500 operands plus `len` is 4008 bytes.
    let just_inside = FusionPlan {
        dtype: DType::F32,
        inputs: 300,
        instructions: vec![],
        outputs: (0..199).map(FusionOperand::Input).collect(),
    };
    assert!(emit_fusion(&just_inside).is_ok());

    let too_long = FusionPlan {
        dtype: DType::F32,
        inputs: 1,
        instructions: (0..=MAX_FUSION_INSTRUCTIONS)
            .map(|position| {
                let operand = if position == 0 {
                    FusionOperand::Input(0)
                } else {
                    FusionOperand::Value(position as u32 - 1)
                };
                instruction(FusionOp::Neg, &[operand])
            })
            .collect(),
        outputs: vec![FusionOperand::Value(MAX_FUSION_INSTRUCTIONS as u32)],
    };
    assert_eq!(emit_fusion(&too_long), Err(Error::FusionTooLarge));
}

#[test]
fn the_source_is_self_contained_and_canonical() {
    let plan = FusionPlan {
        dtype: DType::F64,
        inputs: 4,
        instructions: vec![
            instruction(
                FusionOp::Mul,
                &[FusionOperand::Input(0), FusionOperand::Input(1)],
            ),
            instruction(
                FusionOp::Add,
                &[FusionOperand::Value(0), FusionOperand::Input(2)],
            ),
            instruction(
                FusionOp::Div,
                &[FusionOperand::Value(1), FusionOperand::Input(3)],
            ),
            instruction(FusionOp::Tanh, &[FusionOperand::Value(1)]),
        ],
        outputs: vec![FusionOperand::Value(2), FusionOperand::Value(3)],
    };
    let emitted = emit_fusion(&plan).unwrap();
    let source = emitted.source();

    // No include path the caller might not set, and the canonical header inside.
    assert!(
        !source.contains("#include"),
        "the source must be self-contained"
    );
    assert!(source.contains("T4A_CUDA_ABI_REVISION"));
    assert!(source.contains("t4a_cuda_div_complex_parts_at"));

    // Entry, argument order (outputs, then inputs, then len) and the tail guard.
    assert!(source.contains("extern \"C\" __global__ void t4a_fused_elementwise("));
    let signature = source
        .lines()
        .find(|line| line.contains("extern \"C\" __global__ void t4a_fused_elementwise("))
        .unwrap();
    assert!(signature.contains("double* out0, double* out1, const double* in0"));
    assert!(signature.contains("long long len)"));
    assert!(source.contains("if (index >= len)"));
    assert!(source.contains("blockIdx.x) * blockDim.x + threadIdx.x"));

    // One temporary per region value, and the outputs reference the instruction
    // values (input count 4, so instruction 2 is v6).
    assert!(source.contains("const double v4 = t4a_cuda_mul(v0, v1);"));
    assert!(source.contains("const double v5 = t4a_cuda_add(v4, v2);"));
    assert!(source.contains("const double v6 = (v5 / v3);"));
    assert!(source.contains("out0[index] = v6;"));
    assert!(source.contains("out1[index] = v7;"));
    let declarations = source
        .lines()
        .filter(|line| line.trim_start().starts_with("const double v"))
        .count();
    assert_eq!(declarations, plan.inputs as usize + plan.instructions.len());

    assert_eq!(emitted.manifest().entry, "t4a_fused_elementwise");
    assert_eq!(emitted.manifest().abi_revision, ABI_REVISION);
    assert_eq!(emitted.manifest().dtype, DType::F64);
    assert_eq!(emitted.manifest().inputs, 4);
    assert_eq!(emitted.manifest().outputs, 2);
    assert_eq!(emitted.manifest().block, FLAT_BLOCK);
    assert_eq!(emitted.manifest().index_width, 64);
}

#[test]
fn complex_plans_use_the_component_forms_and_the_robust_division() {
    let plan = FusionPlan {
        dtype: DType::C64,
        inputs: 2,
        instructions: vec![
            instruction(
                FusionOp::Div,
                &[FusionOperand::Input(0), FusionOperand::Input(1)],
            ),
            instruction(FusionOp::Conj, &[FusionOperand::Value(0)]),
        ],
        outputs: vec![FusionOperand::Value(0), FusionOperand::Value(1)],
    };
    let emitted = emit_fusion(&plan).unwrap();
    let source = emitted.source().to_string();
    assert!(source.contains("t4a_cuda_c64* out0"));
    assert!(source.contains("const t4a_cuda_c64 v2 = t4a_cuda_div_complex(v0, v1);"));
    assert!(source.contains("t4a_cuda_conj(v2)"));
    // The robust division, not the ordinary `/`.
    assert!(!source.contains("v0 / v1"));
}

#[test]
fn the_key_covers_everything_that_changes_the_artifact() {
    let base = single_op(FusionOp::Add);
    let emitted = emit_fusion(&base).unwrap();
    let key = artifact_key(&emitted, "-O3", "sm_80:cubin", "nvrtc 12.0");
    assert!(key.material().len() > emitted.source().len());
    assert_eq!(
        key,
        artifact_key(&emitted, "-O3", "sm_80:cubin", "nvrtc 12.0")
    );
    // The key material is the source plus the three caller fields, each length-prefixed.
    assert_eq!(
        key.material().len(),
        emitted.key().material().len()
            + 3 * 8
            + "-O3".len()
            + "sm_80:cubin".len()
            + "nvrtc 12.0".len()
    );
    assert!(key.material().len() > emitted.source().len());

    // Compiler identity, options and target are all part of the key.
    assert_ne!(
        key,
        artifact_key(&emitted, "-O0", "sm_80:cubin", "nvrtc 12.0")
    );
    assert_ne!(
        key,
        artifact_key(&emitted, "-O3", "sm_90:cubin", "nvrtc 12.0")
    );
    assert_ne!(
        key,
        artifact_key(&emitted, "-O3", "sm_80:cubin", "nvrtc 12.6")
    );
    assert_ne!(
        key,
        artifact_key(&emitted, "-O3", "sm_80:ptx", "nvrtc 12.0")
    );

    // A different op set or dtype is a different artifact.
    let other_op = emit_fusion(&single_op(FusionOp::Mul)).unwrap();
    assert_ne!(
        key,
        artifact_key(&other_op, "-O3", "sm_80:cubin", "nvrtc 12.0")
    );
    let other_dtype = emit_fusion(&FusionPlan {
        dtype: DType::F64,
        ..single_op(FusionOp::Add)
    })
    .unwrap();
    assert_ne!(
        key,
        artifact_key(&other_dtype, "-O3", "sm_80:cubin", "nvrtc 12.0")
    );

    // The runtime length is not part of the artifact: the same plan serves any size.
    assert_eq!(emitted.key(), emit_fusion(&base).unwrap().key());
}

#[test]
fn the_launch_geometry_follows_the_runtime_length() {
    assert!(plan_fused(0).unwrap().is_empty());
    let one_block = plan_fused(FLAT_BLOCK as i64).unwrap();
    assert_eq!(one_block.grid, [1, 1, 1]);
    assert_eq!(one_block.block, [FLAT_BLOCK, 1, 1]);
    let ragged = plan_fused(513).unwrap();
    assert_eq!(ragged.grid, [3, 1, 1]);
    assert_eq!(plan_fused(-1), Err(Error::NegativeCount));
    assert_eq!(
        plan_fused(i64::from(i32::MAX) * i64::from(FLAT_BLOCK) + 1),
        Err(Error::LaunchLimit)
    );
}

#[test]
fn shared_values_are_computed_once() {
    // Eight successive squarings: eight instructions. A recursive tree would need 511
    // nodes for the same computation, which is why the plan is an instruction list.
    let mut instructions = vec![instruction(
        FusionOp::Mul,
        &[FusionOperand::Input(0), FusionOperand::Input(0)],
    )];
    for position in 1..8 {
        instructions.push(instruction(
            FusionOp::Mul,
            &[
                FusionOperand::Value(position - 1),
                FusionOperand::Value(position - 1),
            ],
        ));
    }
    let plan = FusionPlan {
        dtype: DType::F32,
        inputs: 1,
        instructions,
        outputs: vec![FusionOperand::Value(7), FusionOperand::Value(3)],
    };
    let emitted = emit_fusion(&plan).unwrap();
    let multiplications = emitted.source().matches("t4a_cuda_mul(v").count();
    assert_eq!(multiplications, 8, "one multiply per instruction");
    assert!(emitted.source().contains("out0[index] = v8;"));
    assert!(emitted.source().contains("out1[index] = v4;"));
}
