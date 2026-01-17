use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::backend::executor::error::{DType, ExecutorError, ExecutorResult};
use crate::core::backend::executor::ops::OpType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        NodeId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub fn with_graph(graph_id: u64, local_id: u64) -> Self {
        NodeId((graph_id << 32) | (local_id & 0xFFFFFFFF))
    }

    pub fn graph_id(&self) -> u64 {
        self.0 >> 32
    }

    pub fn local_id(&self) -> u64 {
        self.0 & 0xFFFFFFFF
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Pending,
    Running,
    Done,
    Failed,
}

impl Default for NodeState {
    fn default() -> Self {
        NodeState::Pending
    }
}

#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl NodeMetadata {
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        Self { shape, dtype }
    }

    pub fn with_shape(shape: Vec<usize>) -> Self {
        Self {
            shape,
            dtype: DType::F32,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_compatible(&self, other: &NodeMetadata) -> bool {
        self.dtype == other.dtype && self.shapes_broadcastable(&other.shape)
    }

    fn shapes_broadcastable(&self, other: &[usize]) -> bool {
        let max_ndim = self.shape.len().max(other.len());
        for i in 0..max_ndim {
            let dim_a = if i < self.shape.len() {
                self.shape[self.shape.len() - 1 - i]
            } else {
                1
            };
            let dim_b = if i < other.len() {
                other[other.len() - 1 - i]
            } else {
                1
            };
            if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub op: OpType,
    pub inputs: Vec<NodeId>,
    pub metadata: NodeMetadata,
    pub state: NodeState,
    #[deprecated(note = "use state instead")]
    pub executed: bool,
}

impl GraphNode {
    pub fn new(id: NodeId, op: OpType, inputs: Vec<NodeId>, shape: Vec<usize>) -> Self {
        Self {
            id,
            op,
            inputs,
            metadata: NodeMetadata::with_shape(shape),
            state: NodeState::Pending,
            #[allow(deprecated)]
            executed: false,
        }
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.metadata.dtype = dtype;
        self
    }

    pub fn shape(&self) -> &[usize] {
        &self.metadata.shape
    }

    pub fn dtype(&self) -> &DType {
        &self.metadata.dtype
    }

    pub fn is_done(&self) -> bool {
        self.state == NodeState::Done
    }

    pub fn is_pending(&self) -> bool {
        self.state == NodeState::Pending
    }

    pub fn is_running(&self) -> bool {
        self.state == NodeState::Running
    }

    pub fn mark_running(&mut self) -> bool {
        if self.state == NodeState::Pending {
            self.state = NodeState::Running;
            true
        } else {
            false
        }
    }

    pub fn mark_done(&mut self) {
        self.state = NodeState::Done;
        #[allow(deprecated)]
        {
            self.executed = true;
        }
    }

    pub fn mark_failed(&mut self) {
        self.state = NodeState::Failed;
    }

    pub fn reset(&mut self) {
        self.state = NodeState::Pending;
        #[allow(deprecated)]
        {
            self.executed = false;
        }
    }
}

#[derive(Debug)]
pub struct ComputeGraph {
    graph_id: u64,
    next_local_id: u64,
    pub nodes: FxHashMap<NodeId, GraphNode>,
    execution_order: Vec<NodeId>,
    users: FxHashMap<NodeId, Vec<NodeId>>,
    dirty: bool,
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeGraph {
    pub fn new() -> Self {
        static GRAPH_COUNTER: AtomicU64 = AtomicU64::new(0);
        Self {
            graph_id: GRAPH_COUNTER.fetch_add(1, Ordering::Relaxed),
            next_local_id: 0,
            nodes: FxHashMap::default(),
            execution_order: Vec::new(),
            users: FxHashMap::default(),
            dirty: false,
        }
    }

    fn next_node_id(&mut self) -> NodeId {
        let id = NodeId::with_graph(self.graph_id, self.next_local_id);
        self.next_local_id += 1;
        id
    }

    pub fn add_node(&mut self, op: OpType, inputs: Vec<NodeId>, shape: Vec<usize>) -> NodeId {
        let id = self.next_node_id();
        let node = GraphNode::new(id, op, inputs.clone(), shape);
        self.nodes.insert(id, node);
        for input in inputs {
            self.users.entry(input).or_default().push(id);
        }
        self.dirty = true;
        id
    }

    pub fn add_node_with_dtype(
        &mut self,
        op: OpType,
        inputs: Vec<NodeId>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> NodeId {
        let id = self.next_node_id();
        let node = GraphNode::new(id, op, inputs.clone(), shape).with_dtype(dtype);
        self.nodes.insert(id, node);
        for input in inputs {
            self.users.entry(input).or_default().push(id);
        }
        self.dirty = true;
        id
    }

    pub fn add_source(&mut self, shape: Vec<usize>) -> NodeId {
        let id = self.next_node_id();
        let mut node = GraphNode::new(id, OpType::Input, Vec::new(), shape);
        node.mark_done();
        self.nodes.insert(id, node);
        self.dirty = true;
        id
    }

    pub fn add_source_with_dtype(&mut self, shape: Vec<usize>, dtype: DType) -> NodeId {
        let id = self.next_node_id();
        let mut node = GraphNode::new(id, OpType::Input, Vec::new(), shape).with_dtype(dtype);
        node.mark_done();
        self.nodes.insert(id, node);
        self.dirty = true;
        id
    }

    pub fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&id)
    }

    pub fn is_unique_user(&self, node_id: NodeId, user_id: NodeId) -> bool {
        match self.users.get(&node_id) {
            Some(users) => users.len() == 1 && users[0] == user_id,
            None => false,
        }
    }

    pub fn get_users(&self, node_id: NodeId) -> Option<&[NodeId]> {
        self.users.get(&node_id).map(|v| v.as_slice())
    }

    pub fn get_execution_order(&mut self) -> &[NodeId] {
        self.compute_execution_order();
        &self.execution_order
    }

    pub fn get_ancestors(&self, targets: &[NodeId]) -> FxHashSet<NodeId> {
        let mut ancestors = FxHashSet::default();
        let mut queue: VecDeque<NodeId> = targets.iter().copied().collect();

        while let Some(node_id) = queue.pop_front() {
            if ancestors.contains(&node_id) {
                continue;
            }
            ancestors.insert(node_id);

            if let Some(node) = self.nodes.get(&node_id) {
                for &input_id in &node.inputs {
                    if !ancestors.contains(&input_id) {
                        queue.push_back(input_id);
                    }
                }
            }
        }

        ancestors
    }

    pub fn get_execution_order_for_targets(&mut self, targets: &[NodeId]) -> Vec<NodeId> {
        self.compute_execution_order();
        let ancestors = self.get_ancestors(targets);
        self.execution_order
            .iter()
            .filter(|id| ancestors.contains(id))
            .copied()
            .collect()
    }

    pub fn try_mark_running(&mut self, node_id: NodeId) -> ExecutorResult<bool> {
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(ExecutorError::NodeNotFound { node_id })?;

        match node.state {
            NodeState::Pending => {
                node.state = NodeState::Running;
                Ok(true)
            }
            NodeState::Running => Err(ExecutorError::ExecutionInProgress { node_id }),
            NodeState::Done => Ok(false),
            NodeState::Failed => Ok(false),
        }
    }

    pub fn mark_done(&mut self, node_id: NodeId) -> ExecutorResult<()> {
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(ExecutorError::NodeNotFound { node_id })?;
        node.mark_done();
        Ok(())
    }

    pub fn mark_failed(&mut self, node_id: NodeId) -> ExecutorResult<()> {
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(ExecutorError::NodeNotFound { node_id })?;
        node.mark_failed();
        Ok(())
    }

    pub fn validate_node(&self, node_id: NodeId) -> ExecutorResult<()> {
        let node = self
            .nodes
            .get(&node_id)
            .ok_or(ExecutorError::NodeNotFound { node_id })?;

        let expected_inputs = node.op.num_inputs();
        if node.inputs.len() != expected_inputs {
            return Err(ExecutorError::InvalidInputCount {
                node_id,
                op: format!("{:?}", node.op),
                expected: expected_inputs,
                actual: node.inputs.len(),
            });
        }

        for &input_id in &node.inputs {
            if !self.nodes.contains_key(&input_id) {
                return Err(ExecutorError::NodeNotFound { node_id: input_id });
            }
        }

        if node.op.is_elementwise() {
            let node_dtype = &node.metadata.dtype;
            for &input_id in &node.inputs {
                if let Some(input_node) = self.nodes.get(&input_id) {
                    if &input_node.metadata.dtype != node_dtype {
                        return Err(ExecutorError::DTypeMismatch {
                            node_id,
                            expected: node_dtype.clone(),
                            actual: input_node.metadata.dtype.clone(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    #[deprecated(note = "Use fusion::FusionCompiler::find_fusion_candidates instead")]
    pub fn optimize(&mut self) {
        self.compute_execution_order();
    }

    pub fn clear_executed(&mut self) {
        self.nodes.retain(|_, node| !node.is_done());
        self.users.retain(|_, users| {
            users.retain(|id| self.nodes.contains_key(id));
            !users.is_empty()
        });
        self.dirty = true;
    }

    pub fn reset(&mut self) {
        for node in self.nodes.values_mut() {
            node.reset();
        }
    }

    fn compute_execution_order(&mut self) {
        if !self.dirty {
            return;
        }

        let mut in_degree: FxHashMap<NodeId, usize> = FxHashMap::default();
        let mut dependents: FxHashMap<NodeId, Vec<NodeId>> = FxHashMap::default();

        for (&id, node) in &self.nodes {
            in_degree.entry(id).or_insert(0);
            for &input_id in &node.inputs {
                *in_degree.entry(id).or_insert(0) += 1;
                dependents.entry(input_id).or_default().push(id);
            }
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(&id, _)| id)
            .collect();

        self.execution_order.clear();

        while let Some(node_id) = queue.pop_front() {
            self.execution_order.push(node_id);
            if let Some(deps) = dependents.get(&node_id) {
                for &dep_id in deps {
                    if let Some(degree) = in_degree.get_mut(&dep_id) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(dep_id);
                        }
                    }
                }
            }
        }

        self.dirty = false;
    }
}
