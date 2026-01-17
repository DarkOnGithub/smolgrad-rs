use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use smolgrad_rs::core::backend::executor::{Executor, ops::OpType};
use smolgrad_rs::core::backend::Buffer;

fn main() {
    // 1. Initialize the WGPU client
    // WGPU is the default backend for cross-platform GPU execution
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    // 2. Create the Executor
    // The executor manages the computation graph, buffer caching, and kernel fusion
    let executor = Executor::new(client.clone());

    // 3. Create some input data for our tensors
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let data_b = vec![5.0f32, 6.0, 7.0, 8.0];
    let shape = vec![2, 2];

    println!("Input A ({}): {:?}", "2x2", data_a);
    println!("Input B ({}): {:?}", "2x2", data_b);

    // 4. Create buffers and register them as sources in the computation graph
    // Source nodes represent inputs that already have data on the GPU
    let buffer_a = Buffer::from_data(&client, &data_a, shape.clone());
    let buffer_b = Buffer::from_data(&client, &data_b, shape.clone());

    let node_a = executor.register_source(buffer_a);
    let node_b = executor.register_source(buffer_b);

    // 5. Record operations (Computation Graph construction)
    // We are building a lazy execution graph. No kernels are launched yet.
    // Expression: result = (a + b) * a
    println!("\nRecording operations: (A + B) * A");
    
    let node_sum = executor.record_op(OpType::Add, vec![node_a, node_b], shape.clone());
    let node_result = executor.record_op(OpType::Mul, vec![node_sum, node_a], shape.clone());

    // 6. Execute the graph
    // This will analyze the graph, potentially fuse operations (Add + Mul), 
    // and launch the necessary kernels.
    println!("Executing computation graph...");
    let result_buffer = executor.execute_to::<f32>(node_result)
        .expect("Execution failed");

    // 7. Read back the data to the host
    let result_data = result_buffer.to_data(&client);

    println!("\nExecution result:");
    println!("Result: {:?}", result_data);

    // Verification
    // (1+5)*1=6, (2+6)*2=16, (3+7)*3=30, (4+8)*4=48
    let expected = vec![6.0, 16.0, 30.0, 48.0];
    assert_eq!(result_data, expected);
    println!("\nVerification successful!");
}
