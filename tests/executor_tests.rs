use smolgrad_rs::core::backend::executor::error::{DType, ExecutorError};
use smolgrad_rs::core::backend::executor::fusion::{
    FusedBinaryOp, FusedInstruction, FusedKernel, FusionCompiler, FusionGroup, FusionPlan,
    FusionRegistry,
};
use smolgrad_rs::core::backend::executor::graph::{ComputeGraph, NodeId, NodeState};
use smolgrad_rs::core::backend::executor::ops::OpType;
use cubecl::prelude::CubeKernel;

#[test]
fn test_target_subgraph_isolation() {
    let graph = ComputeGraph::new();
    let mut g = graph;

    let a = g.add_source(vec![2, 2]);
    let b = g.add_source(vec![2, 2]);
    let c = g.add_source(vec![2, 2]);

    let d = g.add_node(OpType::Add, vec![a, b], vec![2, 2]);
    let e = g.add_node(OpType::Mul, vec![b, c], vec![2, 2]);
    let _f = g.add_node(OpType::Add, vec![d, e], vec![2, 2]);

    let order_d = g.get_execution_order_for_targets(&[d]);

    assert!(order_d.contains(&a));
    assert!(order_d.contains(&b));
    assert!(order_d.contains(&d));
    assert!(
        !order_d.contains(&c),
        "c should not be in execution order for d"
    );
    assert!(
        !order_d.contains(&e),
        "e should not be in execution order for d"
    );
}

#[test]
fn test_validation_catches_input_count_mismatch() {
    let mut graph = ComputeGraph::new();

    let a = graph.add_source(vec![2, 2]);
    let b = graph.add_node(OpType::Add, vec![a], vec![2, 2]);

    let result = graph.validate_node(b);
    assert!(result.is_err());
    if let Err(ExecutorError::InvalidInputCount {
        expected, actual, ..
    }) = result
    {
        assert_eq!(expected, 2);
        assert_eq!(actual, 1);
    }
}

#[test]
fn test_fusion_detection() {
    let mut graph = ComputeGraph::new();

    let a = graph.add_source(vec![4, 4]);
    let b = graph.add_source(vec![4, 4]);
    let c = graph.add_source(vec![4, 4]);

    let mul = graph.add_node(OpType::Mul, vec![a, b], vec![4, 4]);
    let add = graph.add_node(OpType::Add, vec![mul, c], vec![4, 4]);

    let compiler = FusionCompiler::new();
    let candidates = compiler.find_fusion_candidates(&graph);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].output_node, add);
    assert!(candidates[0].fused_nodes.contains(&mul));
    assert!(candidates[0].fused_nodes.contains(&add));
}

#[test]
fn test_ancestor_computation() {
    let mut graph = ComputeGraph::new();

    let a = graph.add_source(vec![2, 2]);
    let b = graph.add_source(vec![2, 2]);
    let c = graph.add_source(vec![2, 2]);

    let d = graph.add_node(OpType::Add, vec![a, b], vec![2, 2]);
    let e = graph.add_node(OpType::Mul, vec![b, c], vec![2, 2]);
    let f = graph.add_node(OpType::Add, vec![d, e], vec![2, 2]);

    let ancestors_f = graph.get_ancestors(&[f]);
    assert!(ancestors_f.contains(&a));
    assert!(ancestors_f.contains(&b));
    assert!(ancestors_f.contains(&c));
    assert!(ancestors_f.contains(&d));
    assert!(ancestors_f.contains(&e));
    assert!(ancestors_f.contains(&f));

    let ancestors_d = graph.get_ancestors(&[d]);
    assert!(ancestors_d.contains(&a));
    assert!(ancestors_d.contains(&b));
    assert!(ancestors_d.contains(&d));
    assert!(!ancestors_d.contains(&c));
    assert!(!ancestors_d.contains(&e));
    assert!(!ancestors_d.contains(&f));
}

#[test]
fn test_target_execution_order() {
    let mut graph = ComputeGraph::new();

    let a = graph.add_source(vec![2, 2]);
    let b = graph.add_source(vec![2, 2]);
    let c = graph.add_source(vec![2, 2]);

    let d = graph.add_node(OpType::Add, vec![a, b], vec![2, 2]);
    let _e = graph.add_node(OpType::Mul, vec![b, c], vec![2, 2]);

    let order = graph.get_execution_order_for_targets(&[d]);
    assert!(order.contains(&a));
    assert!(order.contains(&b));
    assert!(order.contains(&d));
    assert!(!order.contains(&c));
}

#[test]
fn test_node_state_transitions() {
    let mut graph = ComputeGraph::new();
    let a = graph.add_source(vec![2, 2]);
    let b = graph.add_source(vec![2, 2]);
    let c = graph.add_node(OpType::Add, vec![a, b], vec![2, 2]);

    assert!(graph.get_node(c).unwrap().state == NodeState::Pending);

    assert!(graph.try_mark_running(c).unwrap());
    assert!(graph.get_node(c).unwrap().state == NodeState::Running);

    assert!(graph.try_mark_running(c).is_err());

    graph.mark_done(c).unwrap();
    assert!(graph.get_node(c).unwrap().state == NodeState::Done);

    assert!(!graph.try_mark_running(c).unwrap());
}

#[test]
fn test_op_properties() {
    assert!(OpType::Add.is_elementwise());
    assert!(OpType::Add.is_binary());
    assert!(OpType::Add.is_commutative());

    assert!(OpType::Sub.is_elementwise());
    assert!(OpType::Sub.is_binary());
    assert!(!OpType::Sub.is_commutative());

    assert!(OpType::Mul.is_elementwise());
    assert!(OpType::Mul.is_binary());
    assert!(OpType::Mul.is_commutative());

    assert!(OpType::Div.is_elementwise());
    assert!(OpType::Div.is_binary());
    assert!(!OpType::Div.is_commutative());

    assert!(!OpType::Input.is_elementwise());
    assert!(!OpType::Input.is_binary());
}

#[test]
fn test_num_inputs() {
    assert_eq!(OpType::Input.num_inputs(), 0);
    assert_eq!(OpType::Add.num_inputs(), 2);
    assert_eq!(OpType::Sub.num_inputs(), 2);
    assert_eq!(OpType::Mul.num_inputs(), 2);
    assert_eq!(OpType::Div.num_inputs(), 2);
}

#[test]
fn test_fusion_registry() {
    let mut graph = ComputeGraph::new();

    let a = graph.add_source(vec![4, 4]);
    let b = graph.add_source(vec![4, 4]);
    let c = graph.add_source(vec![4, 4]);

    let mul = graph.add_node(OpType::Mul, vec![a, b], vec![4, 4]);
    let add = graph.add_node(OpType::Add, vec![mul, c], vec![4, 4]);

    let compiler = FusionCompiler::new();
    let candidates = compiler.find_fusion_candidates(&graph);

    let registry = FusionRegistry::new();
    for plan in candidates {
        registry.register(plan);
    }

    assert!(registry.is_fused(mul));
    assert!(registry.is_fused(add));
    assert!(registry.is_fusion_output(add));
    assert!(!registry.is_fusion_output(mul));
    assert!(registry.is_internal_fused_node(mul));
}

#[test]
fn test_fusion_group_kernel() {
    let mut graph = ComputeGraph::new();

    let a = graph.add_source(vec![4, 4]);
    let b = graph.add_source(vec![4, 4]);
    let c = graph.add_source(vec![4, 4]);

    let mul = graph.add_node(OpType::Mul, vec![a, b], vec![4, 4]);
    let _add = graph.add_node(OpType::Add, vec![mul, c], vec![4, 4]);

    let compiler = FusionCompiler::new();
    let candidates = compiler.find_fusion_candidates(&graph);

    assert!(!candidates.is_empty());
    let group = FusionGroup::new(candidates[0].clone());

    // Verify the JIT kernel was created
    assert_eq!(group.plan.input_count, 3);

    // The kernel definition should have the right number of buffers
    let def = group.kernel.define();
    assert_eq!(def.buffers.len(), 4); // 3 inputs + 1 output
}

#[test]
fn test_fusion_plan_for_fma() {
    let mut graph = ComputeGraph::new();

    let a = graph.add_source(vec![4, 4]);
    let b = graph.add_source(vec![4, 4]);
    let c = graph.add_source(vec![4, 4]);

    let mul = graph.add_node(OpType::Mul, vec![a, b], vec![4, 4]);
    let add = graph.add_node(OpType::Add, vec![mul, c], vec![4, 4]);

    let compiler = FusionCompiler::new();
    let candidates = compiler.find_fusion_candidates(&graph);

    assert_eq!(candidates.len(), 1);
    let plan = &candidates[0];
    assert_eq!(plan.fused_nodes.len(), 2);
    assert_eq!(plan.input_count, 3);
    assert_eq!(plan.output_node, add);
}

#[test]
fn test_longer_chain_fusion() {
    let mut graph = ComputeGraph::new();

    let a = graph.add_source(vec![4, 4]);
    let b = graph.add_source(vec![4, 4]);

    let add1 = graph.add_node(OpType::Add, vec![a, b], vec![4, 4]);
    let mul = graph.add_node(OpType::Mul, vec![add1, b], vec![4, 4]);
    let sub = graph.add_node(OpType::Sub, vec![mul, a], vec![4, 4]);
    let _div = graph.add_node(OpType::Div, vec![sub, b], vec![4, 4]);

    let compiler = FusionCompiler::new();
    let candidates = compiler.find_fusion_candidates(&graph);

    assert_eq!(candidates.len(), 1);
    let plan = &candidates[0];
    assert_eq!(plan.fused_nodes.len(), 4);
    assert_eq!(plan.input_count, 2);
}

#[test]
fn test_no_fusion_for_multi_use() {
    let mut graph = ComputeGraph::new();

    let a = graph.add_source(vec![4, 4]);
    let b = graph.add_source(vec![4, 4]);
    let c = graph.add_source(vec![4, 4]);

    let mul = graph.add_node(OpType::Mul, vec![a, b], vec![4, 4]);
    let _add1 = graph.add_node(OpType::Add, vec![mul, c], vec![4, 4]);
    let _add2 = graph.add_node(OpType::Add, vec![mul, a], vec![4, 4]);

    let compiler = FusionCompiler::new();
    let candidates = compiler.find_fusion_candidates(&graph);

    assert!(candidates.is_empty() || candidates.iter().all(|p| p.fused_nodes.len() < 2));
}

#[test]
fn test_fusion_signature() {
    let mut graph = ComputeGraph::new();

    let a = graph.add_source(vec![4, 4]);
    let b = graph.add_source(vec![4, 4]);
    let c = graph.add_source(vec![4, 4]);

    let mul = graph.add_node(OpType::Mul, vec![a, b], vec![4, 4]);
    let _add = graph.add_node(OpType::Add, vec![mul, c], vec![4, 4]);

    let compiler = FusionCompiler::new();
    let candidates = compiler.find_fusion_candidates(&graph);

    assert!(!candidates.is_empty());
    let sig = candidates[0].signature();
    assert_eq!(sig.input_count, 3);
}

fn make_test_node_ids(count: usize) -> Vec<NodeId> {
    (0..count).map(|_| NodeId::new()).collect()
}

#[test]
fn test_fused_kernel_id_stability() {
    let node_ids = make_test_node_ids(5);
    let plan = FusionPlan {
        instructions: vec![
            FusedInstruction::LoadInput(0),
            FusedInstruction::LoadInput(1),
            FusedInstruction::BinaryOp {
                op: FusedBinaryOp::Mul,
                lhs: 0,
                rhs: 1,
            },
            FusedInstruction::LoadInput(2),
            FusedInstruction::BinaryOp {
                op: FusedBinaryOp::Add,
                lhs: 2,
                rhs: 3,
            },
            FusedInstruction::Store(4),
        ],
        input_count: 3,
        input_nodes: vec![node_ids[0], node_ids[1], node_ids[2]],
        output_node: node_ids[4],
        fused_nodes: vec![node_ids[3], node_ids[4]],
        dtype: DType::F32,
        shape: vec![4, 4],
    };

    let kernel1 = FusedKernel::new(plan.clone());
    let kernel2 = FusedKernel::new(plan);

    assert_eq!(kernel1.id, kernel2.id);
}

#[test]
fn test_kernel_definition_builds() {
    let node_ids = make_test_node_ids(3);
    let plan = FusionPlan {
        instructions: vec![
            FusedInstruction::LoadInput(0),
            FusedInstruction::LoadInput(1),
            FusedInstruction::BinaryOp {
                op: FusedBinaryOp::Add,
                lhs: 0,
                rhs: 1,
            },
            FusedInstruction::Store(2),
        ],
        input_count: 2,
        input_nodes: vec![node_ids[0], node_ids[1]],
        output_node: node_ids[2],
        fused_nodes: vec![node_ids[2]],
        dtype: DType::F32,
        shape: vec![4, 4],
    };

    let kernel = FusedKernel::new(plan);
    let def = kernel.define();

    // Should have 2 input + 1 output buffers
    assert_eq!(def.buffers.len(), 3);
    // No scalars
    assert!(def.scalars.is_empty());
}
