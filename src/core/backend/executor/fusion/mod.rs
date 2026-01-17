pub mod ir;
pub mod jit;


pub use jit::{FusedKernel, launch_fused_kernel_jit};

pub use ir::{FusedBinaryOp, FusedInstruction, FusionCompiler, FusionPlan, FusionSignature};

use rustc_hash::FxHashMap;
use std::sync::RwLock;

use crate::core::backend::executor::graph::NodeId;

/// A fusion group contains all information needed to execute a fused kernel
#[derive(Debug, Clone)]
pub struct FusionGroup {
    pub plan: FusionPlan,
    /// The JIT-compiled kernel for this fusion group
    pub kernel: FusedKernel,
}

impl FusionGroup {
    pub fn new(plan: FusionPlan) -> Self {
        let kernel = FusedKernel::new(plan.clone());
        Self { plan, kernel }
    }
}

pub struct FusionRegistry {
    groups: RwLock<FxHashMap<NodeId, FusionGroup>>,
    node_to_group: RwLock<FxHashMap<NodeId, NodeId>>,
}

impl Default for FusionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionRegistry {
    pub fn new() -> Self {
        Self {
            groups: RwLock::new(FxHashMap::default()),
            node_to_group: RwLock::new(FxHashMap::default()),
        }
    }

    pub fn register(&self, plan: FusionPlan) {
        let output_node = plan.output_node;
        let fused_nodes = plan.fused_nodes.clone();
        let group = FusionGroup::new(plan);

        {
            let mut node_to_group = self.node_to_group.write().unwrap();
            for &node_id in &fused_nodes {
                node_to_group.insert(node_id, output_node);
            }
        }

        {
            let mut groups = self.groups.write().unwrap();
            groups.insert(output_node, group);
        }
    }

    pub fn get_group(&self, output_node: NodeId) -> Option<FusionGroup> {
        let groups = self.groups.read().unwrap();
        groups.get(&output_node).cloned()
    }

    pub fn get_group_for_node(&self, node_id: NodeId) -> Option<FusionGroup> {
        let output_node = {
            let node_to_group = self.node_to_group.read().unwrap();
            *node_to_group.get(&node_id)?
        };
        self.get_group(output_node)
    }

    pub fn is_fused(&self, node_id: NodeId) -> bool {
        let node_to_group = self.node_to_group.read().unwrap();
        node_to_group.contains_key(&node_id)
    }

    pub fn is_fusion_output(&self, node_id: NodeId) -> bool {
        let groups = self.groups.read().unwrap();
        groups.contains_key(&node_id)
    }

    pub fn is_internal_fused_node(&self, node_id: NodeId) -> bool {
        if !self.is_fused(node_id) {
            return false;
        }
        !self.is_fusion_output(node_id)
    }

    pub fn clear(&self) {
        self.groups.write().unwrap().clear();
        self.node_to_group.write().unwrap().clear();
    }

    pub fn len(&self) -> usize {
        self.groups.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
