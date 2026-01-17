use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::hash::Hash;

use crate::core::backend::executor::error::DType;
use crate::core::backend::executor::graph::{ComputeGraph, NodeId};
use crate::core::backend::executor::ops::OpType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl FusedBinaryOp {
    pub fn from_op_type(op: &OpType) -> Option<Self> {
        match op {
            OpType::Add => Some(FusedBinaryOp::Add),
            OpType::Sub => Some(FusedBinaryOp::Sub),
            OpType::Mul => Some(FusedBinaryOp::Mul),
            OpType::Div => Some(FusedBinaryOp::Div),
            _ => None,
        }
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            FusedBinaryOp::Add => "+",
            FusedBinaryOp::Sub => "-",
            FusedBinaryOp::Mul => "*",
            FusedBinaryOp::Div => "/",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusedInstruction {
    LoadInput(usize),
    BinaryOp {
        op: FusedBinaryOp,
        lhs: usize,
        rhs: usize,
    },
    Store(usize),
}

#[derive(Debug, Clone)]
pub struct FusionPlan {
    pub instructions: Vec<FusedInstruction>,
    pub input_count: usize,
    pub input_nodes: Vec<NodeId>,
    pub output_node: NodeId,
    pub fused_nodes: Vec<NodeId>,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl FusionPlan {
    pub fn signature(&self) -> FusionSignature {
        FusionSignature {
            instructions: self.instructions.clone(),
            input_count: self.input_count,
            dtype: self.dtype.clone(),
        }
    }

    pub fn temp_count(&self) -> usize {
        let mut max_reg = self.input_count;
        for inst in &self.instructions {
            if let FusedInstruction::BinaryOp { lhs, rhs, .. } = inst {
                max_reg = max_reg.max(*lhs + 1).max(*rhs + 1);
            }
            if let FusedInstruction::Store(reg) = inst {
                max_reg = max_reg.max(*reg + 1);
            }
        }
        max_reg.saturating_sub(self.input_count)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusionSignature {
    pub instructions: Vec<FusedInstruction>,
    pub input_count: usize,
    pub dtype: DType,
}

pub struct FusionCompiler {
    min_ops_to_fuse: usize,
    max_chain_length: usize,
}

impl Default for FusionCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionCompiler {
    pub fn new() -> Self {
        Self {
            min_ops_to_fuse: 2,
            max_chain_length: 16,
        }
    }

    pub fn with_min_ops(mut self, min: usize) -> Self {
        self.min_ops_to_fuse = min;
        self
    }

    pub fn with_max_chain(mut self, max: usize) -> Self {
        self.max_chain_length = max;
        self
    }

    pub fn find_fusion_candidates(&self, graph: &ComputeGraph) -> Vec<FusionPlan> {
        let mut candidates = Vec::new();
        let mut fused_nodes: FxHashSet<NodeId> = FxHashSet::default();

        let execution_order: Vec<NodeId> = {
            let mut in_degree: FxHashMap<NodeId, usize> = FxHashMap::default();
            let mut dependents: FxHashMap<NodeId, Vec<NodeId>> = FxHashMap::default();

            for (&id, node) in &graph.nodes {
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

            let mut order = Vec::new();
            while let Some(node_id) = queue.pop_front() {
                order.push(node_id);
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
            order
        };

        for &node_id in execution_order.iter().rev() {
            if fused_nodes.contains(&node_id) {
                continue;
            }

            let node = match graph.get_node(node_id) {
                Some(n) => n,
                None => continue,
            };

            if node.is_done() || !node.op.is_elementwise() {
                continue;
            }

            if let Some(plan) = self.build_fusion_plan(graph, node_id, &fused_nodes) {
                if plan.fused_nodes.len() >= self.min_ops_to_fuse {
                    for &fused_id in &plan.fused_nodes {
                        fused_nodes.insert(fused_id);
                    }
                    candidates.push(plan);
                }
            }
        }

        candidates
    }

    fn build_fusion_plan(
        &self,
        graph: &ComputeGraph,
        output_node: NodeId,
        already_fused: &FxHashSet<NodeId>,
    ) -> Option<FusionPlan> {
        let output = graph.get_node(output_node)?;
        if !output.op.is_elementwise() {
            return None;
        }

        let dtype = output.metadata.dtype.clone();
        let shape = output.metadata.shape.clone();

        let mut chain_nodes: Vec<NodeId> = Vec::new();
        let mut external_inputs: Vec<NodeId> = Vec::new();
        let mut visited: FxHashSet<NodeId> = FxHashSet::default();

        self.collect_fusible_subgraph(
            graph,
            output_node,
            output_node,
            &dtype,
            already_fused,
            &mut chain_nodes,
            &mut external_inputs,
            &mut visited,
        );

        if chain_nodes.is_empty() {
            return None;
        }

        let plan = self.compile_to_ir(
            graph,
            &chain_nodes,
            &external_inputs,
            output_node,
            dtype,
            shape,
        )?;
        Some(plan)
    }

    fn collect_fusible_subgraph(
        &self,
        graph: &ComputeGraph,
        node_id: NodeId,
        output_id: NodeId,
        dtype: &DType,
        already_fused: &FxHashSet<NodeId>,
        chain_nodes: &mut Vec<NodeId>,
        external_inputs: &mut Vec<NodeId>,
        visited: &mut FxHashSet<NodeId>,
    ) {
        if visited.contains(&node_id) {
            return;
        }
        visited.insert(node_id);

        if chain_nodes.len() >= self.max_chain_length {
            if !external_inputs.contains(&node_id) {
                external_inputs.push(node_id);
            }
            return;
        }

        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => return,
        };

        let can_fuse = node.op.is_elementwise()
            && !node.is_done()
            && !already_fused.contains(&node_id)
            && &node.metadata.dtype == dtype
            && (node_id == output_id || self.is_only_used_by(graph, node_id, visited));

        if can_fuse {
            chain_nodes.push(node_id);

            for &input_id in &node.inputs {
                self.collect_fusible_subgraph(
                    graph,
                    input_id,
                    output_id,
                    dtype,
                    already_fused,
                    chain_nodes,
                    external_inputs,
                    visited,
                );
            }
        } else {
            if !external_inputs.contains(&node_id) {
                external_inputs.push(node_id);
            }
        }
    }

    fn is_only_used_by(
        &self,
        graph: &ComputeGraph,
        node_id: NodeId,
        fused_set: &FxHashSet<NodeId>,
    ) -> bool {
        if let Some(users) = graph.get_users(node_id) {
            users.iter().all(|user| fused_set.contains(user))
        } else {
            true
        }
    }

    fn compile_to_ir(
        &self,
        graph: &ComputeGraph,
        chain_nodes: &[NodeId],
        external_inputs: &[NodeId],
        output_node: NodeId,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Option<FusionPlan> {
        let input_count = external_inputs.len();

        let mut node_to_reg: FxHashMap<NodeId, usize> = FxHashMap::default();
        for (i, &input_id) in external_inputs.iter().enumerate() {
            node_to_reg.insert(input_id, i);
        }

        let mut instructions = Vec::new();

        for i in 0..input_count {
            instructions.push(FusedInstruction::LoadInput(i));
        }

        let mut next_reg = input_count;

        let sorted_chain = self.topological_sort_chain(graph, chain_nodes, external_inputs);

        for &node_id in &sorted_chain {
            let node = graph.get_node(node_id)?;
            let op = FusedBinaryOp::from_op_type(&node.op)?;

            if node.inputs.len() != 2 {
                return None;
            }

            let lhs_reg = *node_to_reg.get(&node.inputs[0])?;
            let rhs_reg = *node_to_reg.get(&node.inputs[1])?;

            let result_reg = next_reg;
            next_reg += 1;

            instructions.push(FusedInstruction::BinaryOp {
                op,
                lhs: lhs_reg,
                rhs: rhs_reg,
            });

            node_to_reg.insert(node_id, result_reg);
        }

        let output_reg = *node_to_reg.get(&output_node)?;
        instructions.push(FusedInstruction::Store(output_reg));

        Some(FusionPlan {
            instructions,
            input_count,
            input_nodes: external_inputs.to_vec(),
            output_node,
            fused_nodes: chain_nodes.to_vec(),
            dtype,
            shape,
        })
    }

    fn topological_sort_chain(
        &self,
        graph: &ComputeGraph,
        chain_nodes: &[NodeId],
        external_inputs: &[NodeId],
    ) -> Vec<NodeId> {
        let chain_set: FxHashSet<NodeId> = chain_nodes.iter().copied().collect();
        let _external_set: FxHashSet<NodeId> = external_inputs.iter().copied().collect();

        let mut in_degree: FxHashMap<NodeId, usize> = FxHashMap::default();
        let mut dependents: FxHashMap<NodeId, Vec<NodeId>> = FxHashMap::default();

        for &node_id in chain_nodes {
            in_degree.entry(node_id).or_insert(0);
            if let Some(node) = graph.get_node(node_id) {
                for &input_id in &node.inputs {
                    if chain_set.contains(&input_id) {
                        *in_degree.entry(node_id).or_insert(0) += 1;
                        dependents.entry(input_id).or_default().push(node_id);
                    }
                }
            }
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(id, degree)| **degree == 0 && chain_set.contains(id))
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::new();
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);
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

        result
    }
}
