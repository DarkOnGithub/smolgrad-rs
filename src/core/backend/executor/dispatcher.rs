use crate::core::backend::{Buffer, kernels};
use crate::core::backend::executor::error::{ExecutorError, ExecutorResult};
use crate::core::backend::executor::graph::NodeId;
use crate::core::backend::executor::ops::OpType;
use cubecl::prelude::*;

pub fn execute_op<R, E>(
    client: &ComputeClient<R>,
    op: &OpType,
    inputs: &[Buffer<R, E>],
    shape: &[usize],
) -> Option<Buffer<R, E>>
where
    R: Runtime,
    E: CubeElement + CubePrimitive + Numeric,
{
    execute_op_result(client, op, inputs, shape, NodeId(0)).ok()
}

pub fn execute_op_result<R, E>(
    client: &ComputeClient<R>,
    op: &OpType,
    inputs: &[Buffer<R, E>],
    shape: &[usize],
    node_id: NodeId,
) -> ExecutorResult<Buffer<R, E>>
where
    R: Runtime,
    E: CubeElement + CubePrimitive + Numeric,
{
    if op == &OpType::Input {
        return Err(ExecutorError::KernelExecutionFailed {
            node_id,
            op: "Input".to_string(),
            reason: "Input nodes do not produce outputs".to_string(),
        });
    }

    let expected_inputs = op.num_inputs();
    if inputs.len() < expected_inputs {
        return Err(ExecutorError::InvalidInputCount {
            node_id,
            op: format!("{:?}", op),
            expected: expected_inputs,
            actual: inputs.len(),
        });
    }

    let mut output = Buffer::empty(client, shape.to_vec());

    match op {
        OpType::Add => {
            kernels::binary::launch_add(client, &inputs[0], &inputs[1], &mut output, 0);
        }
        OpType::Sub => {
            kernels::binary::launch_sub(client, &inputs[0], &inputs[1], &mut output, 0);
        }
        OpType::Mul => {
            kernels::binary::launch_mul(client, &inputs[0], &inputs[1], &mut output, 0);
        }
        OpType::Div => {
            kernels::binary::launch_div(client, &inputs[0], &inputs[1], &mut output, 0);
        }
        OpType::Input => unreachable!(),
    }

    Ok(output)
}

pub fn validate_inputs<R, E>(
    op: &OpType,
    inputs: &[Buffer<R, E>],
    expected_shape: &[usize],
    node_id: NodeId,
) -> ExecutorResult<()>
where
    R: Runtime,
    E: CubeElement + CubePrimitive,
{
    let expected_count = op.num_inputs();
    if inputs.len() != expected_count {
        return Err(ExecutorError::InvalidInputCount {
            node_id,
            op: format!("{:?}", op),
            expected: expected_count,
            actual: inputs.len(),
        });
    }

    if op.is_elementwise() {
        for input in inputs.iter() {
            let input_shape = input.shape();
            if !shapes_compatible(input_shape, expected_shape) {
                return Err(ExecutorError::ShapeMismatch {
                    node_id,
                    expected: expected_shape.to_vec(),
                    actual: input_shape.to_vec(),
                });
            }
        }
    }

    Ok(())
}

fn shapes_compatible(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() {
        let max_len = a.len().max(b.len());
        for i in 0..max_len {
            let dim_a = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
            let dim_b = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
            if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
                return false;
            }
        }
        true
    } else {
        a == b
    }
}
