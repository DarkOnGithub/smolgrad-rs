use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn contiguous_kernel<E: CubePrimitive>(input: &Tensor<E>, output: &mut Tensor<E>, offset: u32) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS + offset as usize];
    }
}

/// Kernel to copy data between buffers (GPU-to-GPU)
#[cube(launch_unchecked)]
pub fn copy_kernel<E: CubePrimitive>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    input_offset: u32,
    output_offset: u32,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS + output_offset as usize] = input[ABSOLUTE_POS + input_offset as usize];
    }
}
