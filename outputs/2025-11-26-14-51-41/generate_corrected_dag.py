#!/usr/bin/env python3

import os

def generate_corrected_dag():
    """Generate corrected DAG addressing all feedback issues"""
    
    dag_content = """// Large-Scale Cross-Node Expert Parallelism DAG - Corrected Version
digraph {
    fontname=Arial rankdir=TB size="30,40"
    
    // Input and Output nodes
    input [label="INPUT\\nGPU: N/A\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightcyan shape=ellipse style=filled]
    output [label="OUTPUT\\nGPU: N/A\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightcyan shape=ellipse style=filled]
    
    // Multi-Head Attention Components (Tensor Parallel across GPUs 0-7)
    ln1 [label="LayerNorm\\nGPU: All\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightblue style=filled]
    
    q_proj [label="Q Projection (Column Parallel)\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]" fillcolor=lightblue style=filled]
    k_proj [label="K Projection (Column Parallel)\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]" fillcolor=lightblue style=filled]
    v_proj [label="V Projection (Column Parallel)\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]" fillcolor=lightblue style=filled]
    
    // Communication nodes for tensor parallelism
    comm_q [label="All-Gather Q\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]\\nOutput: [batch_size=4, seq_len=2048, num_heads=128, head_dim=128]" fillcolor=lightyellow shape=ellipse style=filled]
    comm_k [label="All-Gather K\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]\\nOutput: [batch_size=4, seq_len=2048, num_heads=128, head_dim=128]" fillcolor=lightyellow shape=ellipse style=filled]
    comm_v [label="All-Gather V\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]\\nOutput: [batch_size=4, seq_len=2048, num_heads=128, head_dim=128]" fillcolor=lightyellow shape=ellipse style=filled]
    
    // Attention computation
    attn_score [label="Attention Score\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, seq_len=2048, num_heads=128]\\nOutput: [batch_size=4, seq_len=2048, seq_len=2048, num_heads=128]" fillcolor=lightblue style=filled]
    attn_softmax [label="Softmax\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, seq_len=2048, num_heads=128]\\nOutput: [batch_size=4, seq_len=2048, seq_len=2048, num_heads=128]" fillcolor=lightblue style=filled]
    attn_weight [label="Weighted Values\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, seq_len=2048, num_heads=128]\\nOutput: [batch_size=4, seq_len=2048, num_heads=128, head_dim=128]" fillcolor=lightblue style=filled]
    
    o_proj [label="O Projection (Row Parallel)\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, num_heads=128, head_dim=128]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightblue style=filled]
    
    // Residual connection
    res1 [label="Residual Add 1\\nGPU: All\\nInput1: [batch_size=4, seq_len=2048, token_dim=7168]\\nInput2: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightblue style=filled]
    
    // MoE Components - Expert Parallelism with 16 experts across 16 GPUs
    gate [label="Gate Network\\nGPU: Routing Node\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, top_k=2]" fillcolor=lightgreen shape=parallelogram style=filled]
    
    // Token routing - single split node with parameterized GPU mapping
    token_split [label="Token Split\\nGPU: 0_0,0_1,0_2,0_3,1_0,1_1,1_2,1_3,2_0,2_1,2_2,2_3,3_0,3_1,3_2,3_3\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    
    // Expert computation subgraph - parameterized for all 16 experts
    expert_gate [label="Expert Gate (Parameterized)\\nGPU: GPU_i for expert_i\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up [label="Expert Up Projection\\nGPU: GPU_i for expert_i\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act [label="Expert Activation (SiLU)\\nGPU: GPU_i for expert_i\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down [label="Expert Down Projection\\nGPU: GPU_i for expert_i\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul [label="Expert Gate Multiply\\nGPU: GPU_i for expert_i\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Token aggregation - single aggregate node with parameterized GPU mapping
    token_aggregate [label="Token Aggregate\\nGPU: 0_0,0_1,0_2,0_3,1_0,1_1,1_2,1_3,2_0,2_1,2_2,2_3,3_0,3_1,3_2,3_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    
    final_agg [label="Final Expert Aggregation\\nGPU: All\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    
    // Final residual and output
    res2 [label="Residual Add 2\\nGPU: All\\nInput1: [batch_size=4, seq_len=2048, token_dim=7168]\\nInput2: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightblue style=filled]
    
    // Connections
    input -> ln1
    ln1 -> q_proj
    ln1 -> k_proj  
    ln1 -> v_proj
    
    q_proj -> comm_q
    k_proj -> comm_k
    v_proj -> comm_v
    
    comm_q -> attn_score
    comm_k -> attn_score
    attn_score -> attn_softmax
    attn_softmax -> attn_weight
    comm_v -> attn_weight
    attn_weight -> o_proj
    o_proj -> res1
    input -> res1
    res1 -> gate
    
    // MoE routing and computation
    gate -> token_split [style=dashed]
    token_split -> expert_gate
    token_split -> expert_up
    
    expert_gate -> expert_act
    expert_up -> expert_mul
    expert_act -> expert_mul
    expert_mul -> expert_down
    expert_down -> token_aggregate
    
    token_aggregate -> final_agg
    final_agg -> res2
    res1 -> res2
    res2 -> output
}"""

    # Write the corrected DAG file
    dag_file_path = "../outputs/2025-11-26-14-51-41/corrected_large_scale_ep_dag.dot"
    with open(dag_file_path, 'w') as f:
        f.write(dag_content)
    
    # Also create a detailed version showing all 16 experts explicitly but with correct formatting
    detailed_dag_content = """// Large-Scale Cross-Node Expert Parallelism DAG - Detailed Corrected Version
digraph {
    fontname=Arial rankdir=TB size="40,60"
    
    // Input and Output nodes
    input [label="INPUT\\nGPU: N/A\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightcyan shape=ellipse style=filled]
    output [label="OUTPUT\\nGPU: N/A\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightcyan shape=ellipse style=filled]
    
    // Multi-Head Attention Components (Tensor Parallel across GPUs 0-7)
    ln1 [label="LayerNorm\\nGPU: All\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightblue style=filled]
    
    q_proj [label="Q Projection (Column Parallel)\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]" fillcolor=lightblue style=filled]
    k_proj [label="K Projection (Column Parallel)\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]" fillcolor=lightblue style=filled]
    v_proj [label="V Projection (Column Parallel)\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]" fillcolor=lightblue style=filled]
    
    // Communication nodes for tensor parallelism
    comm_q [label="All-Gather Q\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]\\nOutput: [batch_size=4, seq_len=2048, num_heads=128, head_dim=128]" fillcolor=lightyellow shape=ellipse style=filled]
    comm_k [label="All-Gather K\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]\\nOutput: [batch_size=4, seq_len=2048, num_heads=128, head_dim=128]" fillcolor=lightyellow shape=ellipse style=filled]
    comm_v [label="All-Gather V\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, num_heads=16, head_dim=128]\\nOutput: [batch_size=4, seq_len=2048, num_heads=128, head_dim=128]" fillcolor=lightyellow shape=ellipse style=filled]
    
    // Attention computation
    attn_score [label="Attention Score\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, seq_len=2048, num_heads=128]\\nOutput: [batch_size=4, seq_len=2048, seq_len=2048, num_heads=128]" fillcolor=lightblue style=filled]
    attn_softmax [label="Softmax\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, seq_len=2048, num_heads=128]\\nOutput: [batch_size=4, seq_len=2048, seq_len=2048, num_heads=128]" fillcolor=lightblue style=filled]
    attn_weight [label="Weighted Values\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, seq_len=2048, num_heads=128]\\nOutput: [batch_size=4, seq_len=2048, num_heads=128, head_dim=128]" fillcolor=lightblue style=filled]
    
    o_proj [label="O Projection (Row Parallel)\\nGPU: 0-7\\nInput: [batch_size=4, seq_len=2048, num_heads=128, head_dim=128]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightblue style=filled]
    
    // Residual connection
    res1 [label="Residual Add 1\\nGPU: All\\nInput1: [batch_size=4, seq_len=2048, token_dim=7168]\\nInput2: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightblue style=filled]
    
    // MoE Components - Expert Parallelism with 16 experts across 16 GPUs
    gate [label="Gate Network\\nGPU: Routing Node\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, top_k=2]" fillcolor=lightgreen shape=parallelogram style=filled]
    
    // Token routing to experts
    split_0 [label="Token Split\\nGPU: 0_0\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_1 [label="Token Split\\nGPU: 0_1\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_2 [label="Token Split\\nGPU: 0_2\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_3 [label="Token Split\\nGPU: 0_3\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_4 [label="Token Split\\nGPU: 1_0\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_5 [label="Token Split\\nGPU: 1_1\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_6 [label="Token Split\\nGPU: 1_2\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_7 [label="Token Split\\nGPU: 1_3\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_8 [label="Token Split\\nGPU: 2_0\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_9 [label="Token Split\\nGPU: 2_1\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_10 [label="Token Split\\nGPU: 2_2\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_11 [label="Token Split\\nGPU: 2_3\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_12 [label="Token Split\\nGPU: 3_0\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_13 [label="Token Split\\nGPU: 3_1\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_14 [label="Token Split\\nGPU: 3_2\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    split_15 [label="Token Split\\nGPU: 3_3\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    
    // Expert 0 computations
    expert_gate_0 [label="Expert 0 Gate\\nGPU: 0_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_0 [label="Expert 0 Up\\nGPU: 0_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_0 [label="Expert 0 Activation\\nGPU: 0_0\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_0 [label="Expert 0 Down\\nGPU: 0_0\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_0 [label="Expert 0 Gate Multiply\\nGPU: 0_0\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 1 computations
    expert_gate_1 [label="Expert 1 Gate\\nGPU: 0_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_1 [label="Expert 1 Up\\nGPU: 0_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_1 [label="Expert 1 Activation\\nGPU: 0_1\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_1 [label="Expert 1 Down\\nGPU: 0_1\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_1 [label="Expert 1 Gate Multiply\\nGPU: 0_1\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 2 computations
    expert_gate_2 [label="Expert 2 Gate\\nGPU: 0_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_2 [label="Expert 2 Up\\nGPU: 0_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_2 [label="Expert 2 Activation\\nGPU: 0_2\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_2 [label="Expert 2 Down\\nGPU: 0_2\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_2 [label="Expert 2 Gate Multiply\\nGPU: 0_2\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 3 computations
    expert_gate_3 [label="Expert 3 Gate\\nGPU: 0_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_3 [label="Expert 3 Up\\nGPU: 0_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_3 [label="Expert 3 Activation\\nGPU: 0_3\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_3 [label="Expert 3 Down\\nGPU: 0_3\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_3 [label="Expert 3 Gate Multiply\\nGPU: 0_3\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 4 computations
    expert_gate_4 [label="Expert 4 Gate\\nGPU: 1_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_4 [label="Expert 4 Up\\nGPU: 1_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_4 [label="Expert 4 Activation\\nGPU: 1_0\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_4 [label="Expert 4 Down\\nGPU: 1_0\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_4 [label="Expert 4 Gate Multiply\\nGPU: 1_0\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 5 computations
    expert_gate_5 [label="Expert 5 Gate\\nGPU: 1_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_5 [label="Expert 5 Up\\nGPU: 1_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_5 [label="Expert 5 Activation\\nGPU: 1_1\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_5 [label="Expert 5 Down\\nGPU: 1_1\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_5 [label="Expert 5 Gate Multiply\\nGPU: 1_1\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 6 computations - FIXED LABEL
    expert_gate_6 [label="Expert 6 Gate\\nGPU: 1_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_6 [label="Expert 6 Up\\nGPU: 1_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_6 [label="Expert 6 Activation\\nGPU: 1_2\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_6 [label="Expert 6 Down\\nGPU: 1_2\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_6 [label="Expert 6 Gate Multiply\\nGPU: 1_2\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 7 computations - FIXED LABEL
    expert_gate_7 [label="Expert 7 Gate\\nGPU: 1_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_7 [label="Expert 7 Up\\nGPU: 1_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_7 [label="Expert 7 Activation\\nGPU: 1_3\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_7 [label="Expert 7 Down\\nGPU: 1_3\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_7 [label="Expert 7 Gate Multiply\\nGPU: 1_3\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 8 computations
    expert_gate_8 [label="Expert 8 Gate\\nGPU: 2_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_8 [label="Expert 8 Up\\nGPU: 2_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_8 [label="Expert 8 Activation\\nGPU: 2_0\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_8 [label="Expert 8 Down\\nGPU: 2_0\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_8 [label="Expert 8 Gate Multiply\\nGPU: 2_0\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 9 computations
    expert_gate_9 [label="Expert 9 Gate\\nGPU: 2_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_9 [label="Expert 9 Up\\nGPU: 2_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_9 [label="Expert 9 Activation\\nGPU: 2_1\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_9 [label="Expert 9 Down\\nGPU: 2_1\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_9 [label="Expert 9 Gate Multiply\\nGPU: 2_1\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 10 computations
    expert_gate_10 [label="Expert 10 Gate\\nGPU: 2_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_10 [label="Expert 10 Up\\nGPU: 2_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_10 [label="Expert 10 Activation\\nGPU: 2_2\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_10 [label="Expert 10 Down\\nGPU: 2_2\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_10 [label="Expert 10 Gate Multiply\\nGPU: 2_2\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 11 computations
    expert_gate_11 [label="Expert 11 Gate\\nGPU: 2_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_11 [label="Expert 11 Up\\nGPU: 2_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_11 [label="Expert 11 Activation\\nGPU: 2_3\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_11 [label="Expert 11 Down\\nGPU: 2_3\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_11 [label="Expert 11 Gate Multiply\\nGPU: 2_3\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 12 computations
    expert_gate_12 [label="Expert 12 Gate\\nGPU: 3_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_12 [label="Expert 12 Up\\nGPU: 3_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_12 [label="Expert 12 Activation\\nGPU: 3_0\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_12 [label="Expert 12 Down\\nGPU: 3_0\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_12 [label="Expert 12 Gate Multiply\\nGPU: 3_0\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 13 computations
    expert_gate_13 [label="Expert 13 Gate\\nGPU: 3_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_13 [label="Expert 13 Up\\nGPU: 3_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_13 [label="Expert 13 Activation\\nGPU: 3_1\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_13 [label="Expert 13 Down\\nGPU: 3_1\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_13 [label="Expert 13 Gate Multiply\\nGPU: 3_1\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 14 computations
    expert_gate_14 [label="Expert 14 Gate\\nGPU: 3_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_14 [label="Expert 14 Up\\nGPU: 3_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_14 [label="Expert 14 Activation\\nGPU: 3_2\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_14 [label="Expert 14 Down\\nGPU: 3_2\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_14 [label="Expert 14 Gate Multiply\\nGPU: 3_2\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Expert 15 computations
    expert_gate_15 [label="Expert 15 Gate\\nGPU: 3_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_up_15 [label="Expert 15 Up\\nGPU: 3_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_act_15 [label="Expert 15 Activation\\nGPU: 3_3\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    expert_down_15 [label="Expert 15 Down\\nGPU: 3_3\\nInput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, token_dim=7168]" fillcolor=lightblue style=filled]
    expert_mul_15 [label="Expert 15 Gate Multiply\\nGPU: 3_3\\nInput1: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nInput2: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]\\nOutput: [batch_size=4, dynamic_seq_len, mlp_hidden=2048]" fillcolor=lightblue style=filled]
    
    // Token aggregation nodes
    agg_0 [label="Token Aggregate\\nGPU: 0_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_1 [label="Token Aggregate\\nGPU: 0_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_2 [label="Token Aggregate\\nGPU: 0_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_3 [label="Token Aggregate\\nGPU: 0_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_4 [label="Token Aggregate\\nGPU: 1_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_5 [label="Token Aggregate\\nGPU: 1_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_6 [label="Token Aggregate\\nGPU: 1_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_7 [label="Token Aggregate\\nGPU: 1_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_8 [label="Token Aggregate\\nGPU: 2_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_9 [label="Token Aggregate\\nGPU: 2_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_10 [label="Token Aggregate\\nGPU: 2_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_11 [label="Token Aggregate\\nGPU: 2_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_12 [label="Token Aggregate\\nGPU: 3_0\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_13 [label="Token Aggregate\\nGPU: 3_1\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_14 [label="Token Aggregate\\nGPU: 3_2\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    agg_15 [label="Token Aggregate\\nGPU: 3_3\\nInput: [batch_size=4, dynamic_seq_len, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    
    final_agg [label="Final Expert Aggregation\\nGPU: All\\nInput: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightgreen shape=parallelogram style=filled]
    
    // Final residual and output
    res2 [label="Residual Add 2\\nGPU: All\\nInput1: [batch_size=4, seq_len=2048, token_dim=7168]\\nInput2: [batch_size=4, seq_len=2048, token_dim=7168]\\nOutput: [batch_size=4, seq_len=2048, token_dim=7168]" fillcolor=lightblue style=filled]
    
    // Connections
    input -> ln1
    ln1 -> q_proj
    ln1 -> k_proj  
    ln1 -> v_proj
    
    q_proj -> comm_q
    k_proj -> comm_k
    v_proj -> comm_v
    
    comm_q -> attn_score
    comm_k -> attn_score
    attn_score -> attn_softmax
    attn_softmax -> attn_weight
    comm_v -> attn_weight
    attn_weight -> o_proj
    o_proj -> res1
    input -> res1
    res1 -> gate
    
    // MoE routing - dashed lines for expert selection
    gate -> split_0 [style=dashed]
    gate -> split_1 [style=dashed]
    gate -> split_2 [style=dashed]
    gate -> split_3 [style=dashed]
    gate -> split_4 [style=dashed]
    gate -> split_5 [style=dashed]
    gate -> split_6 [style=dashed]
    gate -> split_7 [style=dashed]
    gate -> split_8 [style=dashed]
    gate -> split_9 [style=dashed]
    gate -> split_10 [style=dashed]
    gate -> split_11 [style=dashed]
    gate -> split_12 [style=dashed]
    gate -> split_13 [style=dashed]
    gate -> split_14 [style=dashed]
    gate -> split_15 [style=dashed]
    
    // Expert 0 pipeline
    split_0 -> expert_gate_0
    split_0 -> expert_up_0
    expert_gate_0 -> expert_act_0
    expert_up_0 -> expert_mul_0
    expert_act_0 -> expert_mul_0
    expert_mul_0 -> expert_down_0
    expert_down_0 -> agg_0
    
    // Expert 1 pipeline
    split_1 -> expert_gate_1
    split_1 -> expert_up_1
    expert_gate_1 -> expert_act_1
    expert_up_1 -> expert_mul_1
    expert_act_1 -> expert_mul_1
    expert_mul_1 -> expert_down_1
    expert_down_1 -> agg_1
    
    // Expert 2 pipeline
    split_2 -> expert_gate_2
    split_2 -> expert_up_2
    expert_gate_2 -> expert_act_2
    expert_up_2 -> expert_mul_2
    expert_act_2 -> expert_mul_2
    expert_mul_2 -> expert_down_2
    expert_down_2 -> agg_2
    
    // Expert 3 pipeline
    split_3 -> expert_gate_3
    split_3 -> expert_up_3
    expert_gate_3 -> expert_act_3
    expert_up_3 -> expert_mul_3
    expert_act_3 -> expert_mul_3
    expert_mul_3 -> expert_down_3
    expert_down_3 -> agg_3
    
    // Expert 4 pipeline
    split_4 -> expert_gate_4
    split_4 -> expert_up_4
    expert_gate_4 -> expert_act_4
    expert_up_4 -> expert_mul_4
    expert_act_4 -> expert_mul_4
    expert_mul_4 -> expert_down_4
    expert_down_4 -> agg_4
    
    // Expert 5 pipeline
    split_5 -> expert_gate_5
    split_5 -> expert_up_5
    expert_gate_5 -> expert_act_5
    expert_up_5 -> expert_mul_5
    expert_act_5 -> expert_mul_5
    expert_mul_5 -> expert_down_5
    expert_down_5 -> agg_5
    
    // Expert 6 pipeline
    split_6 -> expert_gate_6
    split_6 -> expert_up_6
    expert_gate_6 -> expert_act_6
    expert_up_6 -> expert_mul_6
    expert_act_6 -> expert_mul_6
    expert_mul_6 -> expert_down_6
    expert_down_6 -> agg_6
    
    // Expert 7 pipeline
    split_7 -> expert_gate_7
    split_7 -> expert_up_7
    expert_gate_7 -> expert_act_7
    expert_up_7 -> expert_mul_7
    expert_act_7 -> expert_mul_7
    expert_mul_7 -> expert_down_7
    expert_down_7 -> agg_7
    
    // Expert 8 pipeline
    split_8 -> expert_gate_8
    split_8 -> expert_up_8
    expert_gate_8 -> expert_act_8
    expert_up_8 -> expert_mul_8
    expert_act_8 -> expert_mul_8
    expert_mul_8 -> expert_down_8
    expert_down_8 -> agg_8
    
    // Expert 9 pipeline
    split_9 -> expert_gate_9
    split_9 -> expert_up_9
    expert_gate_9 -> expert_act_9
    expert_up_9 -> expert_mul_9
    expert_act_9 -> expert_mul_9
    expert_mul_9 -> expert_down_9
    expert_down_9 -> agg_9
    
    // Expert 10 pipeline
    split_10 -> expert_gate_10
    split_10 -> expert_up_10
    expert_gate_10 -> expert_act_10
    expert_up_10 -> expert_mul_10
    expert_act_10 -> expert_mul_10
    expert_mul_10 -> expert_down_10
    expert_down_10 -> agg_10
    
    // Expert 11 pipeline
    split_11 -> expert_gate_11
    split_11 -> expert_up_11
    expert_gate_11 -> expert_act_11
    expert_up_11 -> expert_mul_11
    expert_act_11 -> expert_mul_11
    expert_mul_11 -> expert_down_11
    expert_down_11 -> agg_11
    
    // Expert 12 pipeline
    split_12 -> expert_gate_12
    split_12 -> expert_up_12
    expert_gate_12 -> expert_act_12
    expert_up_12 -> expert_mul_12
    expert_act_12 -> expert_mul_12
    expert_mul_12 -> expert_down_12
    expert_down_12 -> agg_12
    
    // Expert 13 pipeline
    split_13 -> expert_gate_13
    split_13 -> expert_up_13
    expert_gate_13 -> expert_act_13
    expert_up_13 -> expert_mul_13
    expert_act_13 -> expert_mul_13
    expert_mul_13 -> expert_down_13
    expert_down_13 -> agg_13
    
    // Expert 14 pipeline
    split_14 -> expert_gate_14
    split_14 -> expert_up_14
    expert_gate_14 -> expert_act_14
    expert_up_14 -> expert_mul_14
    expert_act_14 -> expert_mul_14
    expert_mul_14 -> expert_down_14
    expert_down_14 -> agg_14
    
    // Expert 15 pipeline
    split_15 -> expert_gate_15
    split_15 -> expert_up_15
    expert_gate_15 -> expert_act_15
    expert_up_15 -> expert_mul_15
    expert_act_15 -> expert_mul_15
    expert_mul_15 -> expert_down_15
    expert_down_15 -> agg_15
    
    // Final aggregation
    agg_0 -> final_agg
    agg_1 -> final_agg
    agg_2 -> final_agg
    agg_3 -> final_agg
    agg_4 -> final_agg
    agg_5 -> final_agg
    agg_6 -> final_agg
    agg_7 -> final_agg
    agg_8 -> final_agg
    agg_9 -> final_agg
    agg_10 -> final_agg
    agg_11 -> final_agg
    agg_12 -> final_agg
    agg_13 -> final_agg
    agg_14 -> final_agg
    agg_15 -> final_agg
    
    final_agg -> res2
    res1 -> res2
    res2 -> output
}"""

    # Write the detailed corrected DAG file
    detailed_dag_file_path = "../outputs/2025-11-26-14-51-41/detailed_corrected_large_scale_ep_dag.dot"
    with open(detailed_dag_file_path, 'w') as f:
        f.write(detailed_dag_content)
    
    print(f"Generated corrected DAG files:")
    print(f"1. Simplified version: {dag_file_path}")
    print(f"2. Detailed version: {detailed_dag_file_path}")
    
    return dag_file_path, detailed_dag_file_path

if __name__ == "__main__":
    generate_corrected_dag()