use std::any::TypeId;
use std::fmt;

use super::graph::NodeId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    Other(&'static str),
}

impl DType {
    pub fn from_type_id(type_id: TypeId) -> Self {
        if type_id == TypeId::of::<f32>() {
            DType::F32
        } else if type_id == TypeId::of::<f64>() {
            DType::F64
        } else if type_id == TypeId::of::<i8>() {
            DType::I8
        } else if type_id == TypeId::of::<i16>() {
            DType::I16
        } else if type_id == TypeId::of::<i32>() {
            DType::I32
        } else if type_id == TypeId::of::<i64>() {
            DType::I64
        } else if type_id == TypeId::of::<u8>() {
            DType::U8
        } else if type_id == TypeId::of::<u16>() {
            DType::U16
        } else if type_id == TypeId::of::<u32>() {
            DType::U32
        } else if type_id == TypeId::of::<u64>() {
            DType::U64
        } else if type_id == TypeId::of::<bool>() {
            DType::Bool
        } else {
            DType::Other("unknown")
        }
    }

    pub fn of<T: 'static>() -> Self {
        Self::from_type_id(TypeId::of::<T>())
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::I8 => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::U16 => write!(f, "u16"),
            DType::U32 => write!(f, "u32"),
            DType::U64 => write!(f, "u64"),
            DType::Bool => write!(f, "bool"),
            DType::Other(name) => write!(f, "{}", name),
        }
    }
}

#[derive(Debug)]
pub enum ExecutorError {
    NodeNotFound {
        node_id: NodeId,
    },
    InputNotReady {
        node_id: NodeId,
        input_id: NodeId,
    },
    CacheMiss {
        node_id: NodeId,
    },
    ShapeMismatch {
        node_id: NodeId,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    DTypeMismatch {
        node_id: NodeId,
        expected: DType,
        actual: DType,
    },
    KernelExecutionFailed {
        node_id: NodeId,
        op: String,
        reason: String,
    },
    InvalidInputCount {
        node_id: NodeId,
        op: String,
        expected: usize,
        actual: usize,
    },
    ExecutionInProgress {
        node_id: NodeId,
    },
    GraphCycle {
        involved_nodes: Vec<NodeId>,
    },
    ValidationFailed {
        node_id: NodeId,
        reason: String,
    },
    FusionFailed {
        node_id: NodeId,
        reason: String,
    },
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutorError::NodeNotFound { node_id } => {
                write!(f, "node {:?} not found in graph", node_id)
            }
            ExecutorError::InputNotReady { node_id, input_id } => {
                write!(f, "input {:?} not ready for node {:?}", input_id, node_id)
            }
            ExecutorError::CacheMiss { node_id } => {
                write!(f, "buffer not found in cache for node {:?}", node_id)
            }
            ExecutorError::ShapeMismatch {
                node_id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "shape mismatch at node {:?}: expected {:?}, got {:?}",
                    node_id, expected, actual
                )
            }
            ExecutorError::DTypeMismatch {
                node_id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "dtype mismatch at node {:?}: expected {}, got {}",
                    node_id, expected, actual
                )
            }
            ExecutorError::KernelExecutionFailed {
                node_id,
                op,
                reason,
            } => {
                write!(
                    f,
                    "kernel execution failed for op {} at node {:?}: {}",
                    op, node_id, reason
                )
            }
            ExecutorError::InvalidInputCount {
                node_id,
                op,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "invalid input count for op {} at node {:?}: expected {}, got {}",
                    op, node_id, expected, actual
                )
            }
            ExecutorError::ExecutionInProgress { node_id } => {
                write!(f, "node {:?} is already being executed", node_id)
            }
            ExecutorError::GraphCycle { involved_nodes } => {
                write!(
                    f,
                    "cycle detected in graph involving nodes {:?}",
                    involved_nodes
                )
            }
            ExecutorError::ValidationFailed { node_id, reason } => {
                write!(f, "validation failed for node {:?}: {}", node_id, reason)
            }
            ExecutorError::FusionFailed { node_id, reason } => {
                write!(f, "fusion failed for node {:?}: {}", node_id, reason)
            }
        }
    }
}

impl std::error::Error for ExecutorError {}

pub type ExecutorResult<T> = Result<T, ExecutorError>;
