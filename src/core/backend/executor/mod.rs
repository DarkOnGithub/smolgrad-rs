use crate::core::backend::{Buffer, BufferId};
use crate::core::backend::executor::cache::BufferCache;
use crate::core::backend::executor::dispatcher::{execute_op_result, validate_inputs};
use crate::core::backend::executor::error::{DType, ExecutorError, ExecutorResult};
use crate::core::backend::executor::fusion::{FusionCompiler, FusionRegistry, launch_fused_kernel_jit};
use crate::core::backend::executor::graph::{ComputeGraph, NodeId, NodeState};
use crate::core::backend::executor::ops::OpType;
use cubecl::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::{Mutex, RwLock};

pub mod cache;
pub mod dispatcher;
pub mod error;
pub mod fusion;
pub mod graph;
pub mod ops;

pub struct ExecutorConfig {
    pub enable_fusion: bool,
    pub min_fusion_ops: usize,
    pub max_fusion_chain: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            enable_fusion: true,
            min_fusion_ops: 2,
            max_fusion_chain: 16,
        }
    }
}

pub struct Executor<R: Runtime> {
    graph: RwLock<ComputeGraph>,
    client: ComputeClient<R>,
    cache: BufferCache,
    sources: RwLock<FxHashMap<BufferId, NodeId>>,
    execution_lock: Mutex<()>,
    fusion_registry: FusionRegistry,
    config: ExecutorConfig,
}

impl<R: Runtime> Executor<R> {
    pub fn new(client: ComputeClient<R>) -> Self {
        Self::with_config(client, ExecutorConfig::default())
    }

    pub fn with_config(client: ComputeClient<R>, config: ExecutorConfig) -> Self {
        Self {
            graph: RwLock::new(ComputeGraph::new()),
            client,
            cache: BufferCache::new(),
            sources: RwLock::new(FxHashMap::default()),
            execution_lock: Mutex::new(()),
            fusion_registry: FusionRegistry::new(),
            config,
        }
    }

    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    pub fn register_source<E: CubeElement + CubePrimitive + 'static>(
        &self,
        buffer: Buffer<R, E>,
    ) -> NodeId {
        let node_id = {
            let mut graph = self.graph.write().unwrap();
            graph.add_source_with_dtype(buffer.shape().to_vec(), DType::of::<E>())
        };
        let buffer_id = buffer.storage_id();
        self.cache.insert(node_id, buffer);
        self.sources.write().unwrap().insert(buffer_id, node_id);
        node_id
    }

    pub fn record_op(&self, op: OpType, inputs: Vec<NodeId>, shape: Vec<usize>) -> NodeId {
        self.graph.write().unwrap().add_node(op, inputs, shape)
    }

    pub fn record_op_with_dtype(
        &self,
        op: OpType,
        inputs: Vec<NodeId>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> NodeId {
        self.graph
            .write()
            .unwrap()
            .add_node_with_dtype(op, inputs, shape, dtype)
    }

    pub fn get_cached<E: CubeElement + CubePrimitive + 'static>(
        &self,
        node_id: NodeId,
    ) -> Option<Buffer<R, E>> {
        self.cache.get::<R, E>(node_id)
    }

    pub fn optimize(&self) {
        self.fusion_registry.clear();

        if self.config.enable_fusion {
            let graph = self.graph.read().unwrap();
            let compiler = FusionCompiler::new()
                .with_min_ops(self.config.min_fusion_ops)
                .with_max_chain(self.config.max_fusion_chain);

            let candidates = compiler.find_fusion_candidates(&graph);

            for plan in candidates {
                self.fusion_registry.register(plan);
            }
        }
        // Note: execution order is computed lazily in get_execution_order()
    }

    pub fn execute_to<E: CubeElement + CubePrimitive + Numeric + 'static>(
        &self,
        target: NodeId,
    ) -> Option<Buffer<R, E>> {
        self.execute_to_result::<E>(target).ok()
    }

    pub fn execute_to_result<E: CubeElement + CubePrimitive + Numeric + 'static>(
        &self,
        target: NodeId,
    ) -> ExecutorResult<Buffer<R, E>> {
        if let Some(buffer) = self.cache.get::<R, E>(target) {
            return Ok(buffer);
        }

        let _guard = self.execution_lock.lock().unwrap();

        if let Some(buffer) = self.cache.get::<R, E>(target) {
            return Ok(buffer);
        }

        let execution_order = {
            let mut graph = self.graph.write().unwrap();
            graph.get_execution_order_for_targets(&[target])
        };

        for &node_id in &execution_order {
            self.execute_node_with_fusion::<E>(node_id)?;
        }

        self.cache
            .get::<R, E>(target)
            .ok_or(ExecutorError::CacheMiss { node_id: target })
    }

    pub fn execute_targets<E: CubeElement + CubePrimitive + Numeric + 'static>(
        &self,
        targets: &[NodeId],
    ) -> ExecutorResult<Vec<Buffer<R, E>>> {
        let _guard = self.execution_lock.lock().unwrap();

        let execution_order = {
            let mut graph = self.graph.write().unwrap();
            graph.get_execution_order_for_targets(targets)
        };

        for &node_id in &execution_order {
            self.execute_node_with_fusion::<E>(node_id)?;
        }

        let mut results = Vec::with_capacity(targets.len());
        for &target in targets {
            let buffer = self
                .cache
                .get::<R, E>(target)
                .ok_or(ExecutorError::CacheMiss { node_id: target })?;
            results.push(buffer);
        }

        Ok(results)
    }

    fn execute_node_with_fusion<E: CubeElement + CubePrimitive + Numeric + 'static>(
        &self,
        node_id: NodeId,
    ) -> ExecutorResult<()> {
        if self.cache.contains(node_id) {
            return Ok(());
        }

        if self.fusion_registry.is_internal_fused_node(node_id) {
            return Ok(());
        }

        if self.fusion_registry.is_fusion_output(node_id) {
            return self.execute_fused_group::<E>(node_id);
        }

        self.execute_node::<E>(node_id)
    }

    fn execute_fused_group<E: CubeElement + CubePrimitive + Numeric + 'static>(
        &self,
        output_node: NodeId,
    ) -> ExecutorResult<()> {
        let group =
            self.fusion_registry
                .get_group(output_node)
                .ok_or(ExecutorError::NodeNotFound {
                    node_id: output_node,
                })?;

        for &fused_id in &group.plan.fused_nodes {
            let should_mark = {
                let mut graph = self.graph.write().unwrap();
                graph.try_mark_running(fused_id)?
            };
            if !should_mark {
                if fused_id == output_node {
                    return Ok(());
                }
            }
        }

        let (shape, dtype) = {
            let graph = self.graph.read().unwrap();
            let node = graph
                .get_node(output_node)
                .ok_or(ExecutorError::NodeNotFound {
                    node_id: output_node,
                })?;
            (node.metadata.shape.clone(), node.metadata.dtype.clone())
        };

        let mut input_buffers: Vec<Buffer<R, E>> = Vec::with_capacity(group.plan.input_count);
        for &input_id in &group.plan.input_nodes {
            let buffer = self
                .cache
                .get::<R, E>(input_id)
                .ok_or(ExecutorError::InputNotReady {
                    node_id: output_node,
                    input_id,
                })?;
            input_buffers.push(buffer);
        }

        let mut output = Buffer::empty(&self.client, shape.clone());

        launch_fused_kernel_jit(&self.client, &input_buffers, &mut output, &group.plan).map_err(
            |reason| ExecutorError::KernelExecutionFailed {
                node_id: output_node,
                op: "FusedKernel".to_string(),
                reason,
            },
        )?;

        self.cache
            .insert_validated(output_node, output, &shape, &dtype)?;

        {
            let mut graph = self.graph.write().unwrap();
            for &fused_id in &group.plan.fused_nodes {
                let _ = graph.mark_done(fused_id);
            }
        }

        Ok(())
    }

    fn execute_node<E: CubeElement + CubePrimitive + Numeric + 'static>(
        &self,
        node_id: NodeId,
    ) -> ExecutorResult<()> {
        if self.cache.contains(node_id) {
            return Ok(());
        }

        let should_execute = {
            let mut graph = self.graph.write().unwrap();
            graph.try_mark_running(node_id)?
        };

        if !should_execute {
            return Ok(());
        }

        let (op, inputs, shape, dtype) = {
            let graph = self.graph.read().unwrap();
            let node = graph
                .get_node(node_id)
                .ok_or(ExecutorError::NodeNotFound { node_id })?;
            (
                node.op.clone(),
                node.inputs.clone(),
                node.metadata.shape.clone(),
                node.metadata.dtype.clone(),
            )
        };

        if op == OpType::Input {
            let mut graph = self.graph.write().unwrap();
            graph.mark_done(node_id)?;
            return Ok(());
        }

        let mut buffers = Vec::with_capacity(inputs.len());
        for &input_id in &inputs {
            let buffer = self
                .cache
                .get::<R, E>(input_id)
                .ok_or(ExecutorError::InputNotReady { node_id, input_id })?;
            buffers.push(buffer);
        }

        validate_inputs(&op, &buffers, &shape, node_id)?;

        let output = execute_op_result(&self.client, &op, &buffers, &shape, node_id)?;

        self.cache
            .insert_validated(node_id, output, &shape, &dtype)?;

        {
            let mut graph = self.graph.write().unwrap();
            graph.mark_done(node_id)?;
        }

        Ok(())
    }

    pub fn execute_all<E: CubeElement + CubePrimitive + Numeric + 'static>(&self) {
        let _ = self.execute_all_result::<E>();
    }

    pub fn execute_all_result<E: CubeElement + CubePrimitive + Numeric + 'static>(
        &self,
    ) -> ExecutorResult<()> {
        let _guard = self.execution_lock.lock().unwrap();

        let execution_order = {
            let mut graph = self.graph.write().unwrap();
            graph.get_execution_order().to_vec()
        };

        for node_id in execution_order {
            if let Err(e) = self.execute_node_with_fusion::<E>(node_id) {
                match &e {
                    ExecutorError::InputNotReady { .. } => continue,
                    _ => return Err(e),
                }
            }
        }

        Ok(())
    }

    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    pub fn clear_executed(&self) {
        self.graph.write().unwrap().clear_executed();
    }

    pub fn reset(&self) {
        self.graph.write().unwrap().reset();
        self.fusion_registry.clear();
    }

    pub fn sync(&self) {
        let _ = pollster::block_on(self.client.sync());
    }

    pub fn validate_graph(&self) -> ExecutorResult<()> {
        let graph = self.graph.read().unwrap();
        for (&node_id, _) in &graph.nodes {
            graph.validate_node(node_id)?;
        }
        Ok(())
    }

    pub fn get_node_state(&self, node_id: NodeId) -> Option<NodeState> {
        let graph = self.graph.read().unwrap();
        graph.get_node(node_id).map(|n| n.state)
    }

    pub fn is_node_ready(&self, node_id: NodeId) -> bool {
        let graph = self.graph.read().unwrap();
        if let Some(node) = graph.get_node(node_id) {
            node.inputs.iter().all(|&id| {
                graph
                    .get_node(id)
                    .map(|n| n.state == NodeState::Done)
                    .unwrap_or(false)
            })
        } else {
            false
        }
    }

    pub fn fusion_count(&self) -> usize {
        self.fusion_registry.len()
    }

    pub fn is_fused(&self, node_id: NodeId) -> bool {
        self.fusion_registry.is_fused(node_id)
    }
}
