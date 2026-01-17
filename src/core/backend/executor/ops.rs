use crate::core::backend::executor::graph::{ComputeGraph, GraphNode, NodeId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpType {
    Input,
    Add,
    Sub,
    Mul,
    Div,
}

impl OpType {
    pub fn num_inputs(&self) -> usize {
        match self {
            OpType::Input => 0,
            OpType::Add | OpType::Sub | OpType::Mul | OpType::Div => 2,
        }
    }

    pub fn is_elementwise(&self) -> bool {
        matches!(self, OpType::Add | OpType::Sub | OpType::Mul | OpType::Div)
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, OpType::Add | OpType::Sub | OpType::Mul | OpType::Div)
    }

    pub fn is_commutative(&self) -> bool {
        matches!(self, OpType::Add | OpType::Mul)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[deprecated(note = "Use fusion::FusionCompiler instead for automatic fusion")]
pub enum FusionPattern {
    MulAdd { mul_id: NodeId, addend_id: NodeId },
}

#[allow(deprecated)]
impl FusionPattern {
    pub fn try_match(_node: &GraphNode, _graph: &ComputeGraph) -> Option<Self> {
        None
    }

    pub fn involved_nodes(&self) -> Vec<NodeId> {
        match self {
            FusionPattern::MulAdd { mul_id, .. } => vec![*mul_id],
        }
    }
}
