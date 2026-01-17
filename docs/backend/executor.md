# Executor

This section documents the executor subsystem in `src/core/executor/`. It builds a
compute graph, validates and dispatches elementwise ops, caches results, and can
optionally fuse chains of elementwise operations into JIT-compiled kernels.

## High-level model

- `ComputeGraph` stores `GraphNode` entries with inputs, shapes, dtypes, and state.
- `Executor` owns the graph, a `BufferCache`, and a `ComputeClient` to run kernels.
- `BufferCache` stores type-erased buffers keyed by `NodeId`, with shape/dtype checks.
- `Dispatcher` maps `OpType` to backend kernels and validates input compatibility.
- `FusionCompiler` identifies fusible chains and emits a `FusionPlan` IR.
- `FusionRegistry` tracks compiled groups and their output nodes.

## Files and responsibilities

### `mod.rs`

Defines the `Executor` orchestrator and configuration.

- `ExecutorConfig` controls fusion (`enable_fusion`, `min_fusion_ops`, `max_fusion_chain`).
- `Executor` owns the graph, cache, fusion registry, and execution lock.
- `register_source` binds an input buffer to a graph node and caches it.
- `record_op`/`record_op_with_dtype` append nodes to the graph.
- `optimize` compiles fusion candidates into the registry.
- `execute_to_result`, `execute_targets`, and `execute_all_result` drive execution.

### `graph.rs`

Implements graph structure and scheduling.

- `NodeId` encodes a graph id and local id.
- `GraphNode` stores `OpType`, inputs, `NodeMetadata`, and `NodeState`.
- `ComputeGraph` maintains node maps, user lists, and topological order.
- `get_execution_order_for_targets` returns a pruned order for subgraphs.

### `ops.rs`

Defines the supported op set.

- `OpType` currently supports `Input`, `Add`, `Sub`, `Mul`, and `Div`.
- Helper methods (`num_inputs`, `is_elementwise`, `is_commutative`) guide validation.

### `dispatcher.rs`

Routes ops to backend kernels and enforces constraints.

- `execute_op_result` launches the correct kernel and returns a new buffer.
- `validate_inputs` checks input count and broadcast-compatible shapes.
- Shape compatibility allows trailing-dimension broadcasting.

### `cache.rs`

Type-erased buffer cache with validation helpers.

- `AnyBuffer` abstracts buffer access while preserving shape and dtype.
- `BufferCache` stores `CacheEntry` values with shape/dtype metadata.
- `insert_validated`/`get_validated` enforce expected shapes and dtypes.

### `error.rs`

Executor-specific error and dtype modeling.

- `DType` maps Rust types to logical executor dtypes.
- `ExecutorError` captures graph, validation, kernel, and fusion failures.

### `fusion/mod.rs`

Registry for fusion groups.

- `FusionGroup` pairs a `FusionPlan` with a `FusedKernel` instance.
- `FusionRegistry` maps fused nodes to output nodes and stored groups.

### `fusion/ir.rs`

Fusion compiler and intermediate representation.

- `FusedInstruction` encodes load, binary op, and store steps.
- `FusionPlan` captures inputs, output node, dtype, and shape.
- `FusionCompiler` discovers fusible chains and emits plans.

### `fusion/jit.rs`

Runtime kernel generation and launch.

- `FusedKernel` builds cubecl IR and assigns a stable kernel id.
- `launch_fused_kernel_jit` handles broadcast views and launches the kernel.

## Execution lifecycle

1. Register inputs with `register_source`, which adds `OpType::Input` nodes.
2. Record new ops with `record_op` or `record_op_with_dtype`.
3. Call `optimize` to detect fusion opportunities.
4. Execute targets with `execute_to_result`, `execute_targets`, or `execute_all_result`.
5. Results are cached; repeated execution reads directly from the `BufferCache`.
6. Use `clear_cache`, `clear_executed`, or `reset` to reset execution state.

The executor serializes execution via `execution_lock` to avoid running the same
node concurrently across threads.

## Fusion flow

- The compiler walks the graph in reverse execution order to build chains of
  elementwise ops with a shared dtype.
- Nodes are only fused if their outputs are not reused by non-fused users.
- A `FusionPlan` is translated into IR instructions, then `FusedKernel` builds
  cubecl IR and is launched by `launch_fused_kernel_jit`.
- Inputs are broadcast to the output shape before launch when needed.

## Validation and errors

- `validate_inputs` enforces input counts and broadcast-compatible shapes.
- `ComputeGraph::validate_node` checks node input counts and dtype consistency.
- Errors return `ExecutorError`, including `ShapeMismatch`, `DTypeMismatch`,
  `KernelExecutionFailed`, and `ExecutionInProgress`.

## Common usage patterns

### Build and execute a small graph

```rust
use crate::core::backend::Buffer;
use crate::core::backend::executor::{Executor, OpType};

let executor = Executor::new(client.clone());
let a = executor.register_source(Buffer::<R, f32>::empty(&client, vec![2, 2]));
let b = executor.register_source(Buffer::<R, f32>::empty(&client, vec![2, 2]));

let add = executor.record_op(OpType::Add, vec![a, b], vec![2, 2]);
executor.optimize();

let out = executor.execute_to_result::<f32>(add)?;
```

### Execute multiple targets

```rust
let outputs = executor.execute_targets::<f32>(&[node_a, node_b])?;
```

### Clear cache between runs

```rust
executor.clear_cache();
executor.clear_executed();
```
