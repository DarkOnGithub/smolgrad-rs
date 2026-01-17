//! JIT Fusion - Dynamic kernel generation using cubecl's KernelBuilder
//!
//! This module generates fused kernels at runtime by building cubecl IR directly,
//! rather than using pre-defined kernel templates.

use cubecl::ir::{
    Arithmetic, BinaryOperator, Branch, Builtin, Comparison, ElemType, FloatKind, Id, If,
    IndexAssignOperator, IndexOperator, Instruction, Metadata, Operation, Operator, Scope, Type,
    UIntKind, Variable, VariableKind,
};
use cubecl::prelude::{
    Binding, ComputeClient, CubeCount, CubeDim, CubeElement, CubeKernel, KernelDefinition,
    KernelId, KernelLauncher, KernelMetadata, KernelOptions, KernelSettings, Location, Numeric,
    Runtime, StorageType, TensorArg, Visibility,
};
use std::hash::{Hash, Hasher};

use crate::core::backend::Buffer;
use crate::core::backend::executor::fusion::ir::{FusedBinaryOp, FusedInstruction, FusionPlan};

/// A dynamically generated fused kernel
#[derive(Clone)]
pub struct FusedKernel {
    /// The fusion plan this kernel implements
    pub plan: FusionPlan,
    /// Unique identifier based on the plan's structure
    pub id: u64,
    /// Cube dimension for launch
    pub cube_dim: CubeDim,
}

impl std::fmt::Debug for FusedKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedKernel")
            .field("id", &self.id)
            .field("input_count", &self.plan.input_count)
            .field("instruction_count", &self.plan.instructions.len())
            .finish()
    }
}

impl FusedKernel {
    pub fn new(plan: FusionPlan) -> Self {
        let id = Self::compute_id(&plan);
        Self {
            plan,
            id,
            cube_dim: CubeDim::new_1d(256),
        }
    }

    pub fn with_cube_dim(mut self, cube_dim: CubeDim) -> Self {
        self.cube_dim = cube_dim;
        self
    }

    fn compute_id(plan: &FusionPlan) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        plan.instructions.hash(&mut hasher);
        plan.input_count.hash(&mut hasher);
        plan.dtype.hash(&mut hasher);
        hasher.finish()
    }

    /// Build the kernel definition using cubecl IR
    fn build_kernel_definition(&self) -> KernelDefinition {
        // Storage types
        let f32_storage = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let u32_storage = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let bool_storage = StorageType::Scalar(ElemType::Bool);

        // Element types - Line of size 1 (scalar operations)
        let line_type = Type::Line(f32_storage, 1);
        let u32_type = Type::Scalar(u32_storage);
        let bool_type = Type::Scalar(bool_storage);

        let mut scope = Scope::root(false);

        // Create input tensor bindings
        let mut bindings = Vec::new();
        let mut input_vars = Vec::new();

        for i in 0..self.plan.input_count {
            let id = i as Id;
            bindings.push(Binding {
                id,
                location: Location::Storage,
                visibility: Visibility::Read,
                ty: line_type.clone(),
                size: None,
                has_extended_meta: true,
            });
            input_vars.push(Variable::new(
                VariableKind::GlobalInputArray(id),
                line_type.clone(),
            ));
        }

        // Create output tensor binding
        let output_id = self.plan.input_count as Id;
        bindings.push(Binding {
            id: output_id,
            location: Location::Storage,
            visibility: Visibility::ReadWrite,
            ty: line_type.clone(),
            size: None,
            has_extended_meta: true,
        });
        let output_var = Variable::new(
            VariableKind::GlobalOutputArray(output_id),
            line_type.clone(),
        );

        // Get ABSOLUTE_POS builtin
        let abs_pos = Variable::builtin(Builtin::AbsolutePos, u32_storage);

        // Get output length for bounds check
        let output_len_var = scope.create_local(u32_type.clone());
        scope.register(Instruction::new(
            Operation::Metadata(Metadata::BufferLength { var: output_var }),
            *output_len_var,
        ));

        // Create bounds check: if ABSOLUTE_POS < output.len()
        let in_bounds_var = scope.create_local(bool_type);
        scope.register(Instruction::new(
            Operation::Comparison(Comparison::Lower(BinaryOperator {
                lhs: abs_pos,
                rhs: *output_len_var,
            })),
            *in_bounds_var,
        ));

        // Create the if branch scope for bounds check
        let mut if_scope = scope.child();

        // Register map: index -> Variable
        let mut registers: Vec<Variable> = Vec::new();

        // Process instructions
        for inst in &self.plan.instructions {
            match inst {
                FusedInstruction::LoadInput(idx) => {
                    // Load from input tensor at ABSOLUTE_POS
                    let loaded = if_scope.create_local(line_type.clone());
                    if_scope.register(Instruction::new(
                        Operation::Operator(Operator::Index(IndexOperator {
                            list: input_vars[*idx],
                            index: abs_pos,
                            line_size: 0, // 0 means same as list
                            unroll_factor: 1,
                        })),
                        *loaded,
                    ));
                    registers.push(*loaded);
                }
                FusedInstruction::BinaryOp { op, lhs, rhs } => {
                    let lhs_var = registers[*lhs];
                    let rhs_var = registers[*rhs];
                    let result = if_scope.create_local(line_type.clone());

                    let arith_op = match op {
                        FusedBinaryOp::Add => Arithmetic::Add(BinaryOperator {
                            lhs: lhs_var,
                            rhs: rhs_var,
                        }),
                        FusedBinaryOp::Sub => Arithmetic::Sub(BinaryOperator {
                            lhs: lhs_var,
                            rhs: rhs_var,
                        }),
                        FusedBinaryOp::Mul => Arithmetic::Mul(BinaryOperator {
                            lhs: lhs_var,
                            rhs: rhs_var,
                        }),
                        FusedBinaryOp::Div => Arithmetic::Div(BinaryOperator {
                            lhs: lhs_var,
                            rhs: rhs_var,
                        }),
                    };

                    if_scope.register(Instruction::new(Operation::Arithmetic(arith_op), *result));
                    registers.push(*result);
                }
                FusedInstruction::Store(reg) => {
                    // Store to output tensor at ABSOLUTE_POS
                    let value = registers[*reg];
                    if_scope.register(Instruction::new(
                        Operation::Operator(Operator::IndexAssign(IndexAssignOperator {
                            index: abs_pos,
                            value,
                            line_size: 0, // 0 means same as output
                            unroll_factor: 1,
                        })),
                        output_var, // The output target
                    ));
                }
            }
        }

        // Register the if branch
        scope.register(Instruction::no_out(Operation::Branch(Branch::If(
            Box::new(If {
                cond: *in_bounds_var,
                scope: if_scope,
            }),
        ))));

        KernelDefinition {
            buffers: bindings,
            tensor_maps: Vec::new(),
            scalars: Vec::new(),
            cube_dim: self.cube_dim,
            body: scope,
            options: KernelOptions {
                kernel_name: format!("fused_elementwise_{}", self.id),
                debug_symbols: false,
                cluster_dim: None,
            },
        }
    }
}

impl KernelMetadata for FusedKernel {
    fn name(&self) -> &'static str {
        "fused_elementwise"
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.id)
    }

    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}

impl CubeKernel for FusedKernel {
    fn define(&self) -> KernelDefinition {
        self.build_kernel_definition()
    }
}

/// Launch a JIT-compiled fused kernel
pub fn launch_fused_kernel_jit<R: Runtime, E: CubeElement + Numeric>(
    client: &ComputeClient<R>,
    inputs: &[Buffer<R, E>],
    output: &mut Buffer<R, E>,
    plan: &FusionPlan,
) -> Result<(), String> {
    if inputs.len() != plan.input_count {
        return Err(format!(
            "Input count mismatch: expected {}, got {}",
            plan.input_count,
            inputs.len()
        ));
    }

    let num_elements = output.len();
    if num_elements == 0 {
        return Ok(());
    }

    let kernel = FusedKernel::new(plan.clone());
    let cube_count = CubeCount::new_1d((num_elements as u32 + 255) / 256);

    // Build tensor arguments
    let output_shape = output.shape().to_vec();
    let line_size = 1usize;

    // Create launcher
    let mut launcher = KernelLauncher::<R>::new(KernelSettings::default());

    // Register input tensors with broadcasting
    for buf in inputs {
        let view = if buf.shape() == output_shape.as_slice() {
            buf.view().clone()
        } else {
            buf.broadcast_view_to(&output_shape)
                .expect("fusion requires broadcastable inputs")
        };
        let handle = buf.handle();
        let tensor_arg = unsafe {
            TensorArg::from_raw_parts::<E>(handle, &view.strides, &view.shape, line_size)
        };
        launcher.register_tensor(&tensor_arg);
    }

    // Register output tensor
    let out_view = output.view().clone();
    let out_handle = output.handle();
    let out_arg = unsafe {
        TensorArg::from_raw_parts::<E>(out_handle, &out_view.strides, &out_view.shape, line_size)
    };
    launcher.register_tensor(&out_arg);

    // Launch the kernel
    launcher
        .launch(cube_count, kernel, client)
        .map_err(|e| format!("Kernel launch failed: {:?}", e))
}
