use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tensor_line_size_parallel;
use std::mem;

use crate::core::backend::{Buffer, BufferView};

fn broadcast_view_or_clone<R: Runtime, E: CubeElement + CubePrimitive>(
    buffer: &Buffer<R, E>,
    target_shape: &[usize],
) -> BufferView {
    if buffer.shape() == target_shape {
        buffer.view().clone()
    } else {
        buffer
            .broadcast_view_to(target_shape)
            .expect("binary kernel requires broadcastable inputs")
    }
}

fn max_line_size_for_view<R: Runtime>(supported: &[usize], view: &BufferView) -> usize {
    let axis = view.shape.len().saturating_sub(1);
    tensor_line_size_parallel(supported.iter().copied(), &view.shape, &view.strides, axis)
}

fn resolve_line_size<R: Runtime>(
    requested: u8,
    supported: &[usize],
    lhs: &BufferView,
    rhs: &BufferView,
    out: &BufferView,
) -> usize {
    let max_line_size = max_line_size_for_view::<R>(supported, lhs)
        .min(max_line_size_for_view::<R>(supported, rhs))
        .min(max_line_size_for_view::<R>(supported, out));
    let limit = if requested == 0 {
        max_line_size
    } else {
        (requested as usize).min(max_line_size)
    };

    supported
        .iter()
        .copied()
        .filter(|&line_size| line_size <= limit)
        .max()
        .unwrap_or(1)
}

fn handle_with_offset<E: CubePrimitive>(handle: &Handle, offset_elements: usize) -> Handle {
    if offset_elements == 0 {
        handle.clone()
    } else {
        let offset_bytes = offset_elements * mem::size_of::<E>();
        handle.clone().offset_start(offset_bytes as u64)
    }
}

macro_rules! define_binary_kernel {
    ($name:ident, $op:tt) => {
        #[cube(launch_unchecked)]
        pub fn $name<E: Numeric>(
            lhs: &Tensor<Line<E>>,
            rhs: &Tensor<Line<E>>,
            output: &mut Tensor<Line<E>>,
        ) {
            if ABSOLUTE_POS < output.len() {
                output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] $op rhs[ABSOLUTE_POS];
            }
        }
    };
}

define_binary_kernel!(add_kernel, +);
define_binary_kernel!(sub_kernel, -);
define_binary_kernel!(mul_kernel, *);
define_binary_kernel!(div_kernel, /);

macro_rules! define_binary_launch {
    ($name:ident, $kernel:ident) => {
        pub fn $name<R: Runtime, E: CubeElement + Numeric>(
            client: &ComputeClient<R>,
            lhs: &Buffer<R, E>,
            rhs: &Buffer<R, E>,
            output: &mut Buffer<R, E>,
            requested_line_size: u8,
        ) {
            let expected_shape = BufferView::broadcast_shape(lhs.shape(), rhs.shape())
                .expect("binary kernel requires broadcastable inputs");
            assert_eq!(
                expected_shape.as_slice(),
                output.shape(),
                "output shape must match broadcasted inputs"
            );

            let output_shape = output.shape().to_vec();
            let lhs_view = broadcast_view_or_clone(lhs, &output_shape);
            let rhs_view = broadcast_view_or_clone(rhs, &output_shape);
            let out_view = output.view();

            let supported = R::supported_line_sizes();
            let line_size = resolve_line_size::<R>(
                requested_line_size,
                supported,
                &lhs_view,
                &rhs_view,
                out_view,
            );

            let num_elements = output.len();
            if num_elements == 0 {
                return;
            }

            let work_units = num_elements / line_size.max(1);
            if work_units == 0 {
                return;
            }

            let cube_dim = CubeDim::new_1d(256);
            let cube_count = CubeCount::new_1d((work_units as u32 + 255) / 256);

            let lhs_handle = handle_with_offset::<E>(lhs.handle(), lhs_view.offset);
            let rhs_handle = handle_with_offset::<E>(rhs.handle(), rhs_view.offset);
            let out_handle = handle_with_offset::<E>(output.handle(), out_view.offset);

            let lhs_arg = unsafe {
                TensorArg::from_raw_parts::<E>(
                    &lhs_handle,
                    &lhs_view.strides,
                    &lhs_view.shape,
                    line_size,
                )
            };
            let rhs_arg = unsafe {
                TensorArg::from_raw_parts::<E>(
                    &rhs_handle,
                    &rhs_view.strides,
                    &rhs_view.shape,
                    line_size,
                )
            };
            let out_arg = unsafe {
                TensorArg::from_raw_parts::<E>(
                    &out_handle,
                    &out_view.strides,
                    &out_view.shape,
                    line_size,
                )
            };

            unsafe {
                $kernel::launch_unchecked::<E, R>(
                    client, cube_count, cube_dim, lhs_arg, rhs_arg, out_arg,
                )
                .expect("binary kernel launch failed")
            };
        }
    };
}

define_binary_launch!(launch_add, add_kernel);
define_binary_launch!(launch_sub, sub_kernel);
define_binary_launch!(launch_mul, mul_kernel);
define_binary_launch!(launch_div, div_kernel);
