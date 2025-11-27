#!/usr/bin/env python3
"""
Engineering-level DAG generator for Large-Scale Cross-Node Expert Parallelism
Integrates DP, TP, and PP with single-expert-per-GPU deployment
"""

import os
import json

def generate_dag():
    """Generate the complete deployment DAG"""
    
    # Configuration based on paper
    BATCH_SIZE = 32
    SEQ_LEN = 2048
    HIDDEN_DIM = 7168
    NUM_HEADS = 128
    HEAD_DIM = 56
    FFN_HIDDEN = 18432
    NUM_EXPERTS_PER_LAYER = 128
    EP_DEGREE = 16
    TP_DEGREE = 8
    DP_DEGREE = 4
    
    # Calculate dimensions
    KV_HEADS = NUM_HEADS  # MLA uses same number
    Q_HEADS_TENSOR_PARALLEL = NUM_HEADS // TP_DEGREE
    KV_HEADS_TENSOR_PARALLEL = KV_HEADS // TP_DEGREE
    HIDDEN_TENSOR_PARALLEL = HIDDEN_DIM // TP_DEGREE
    FFN_HIDDEN_TENSOR_PARALLEL = FFN_HIDDEN // TP_DEGREE
    
    dot_content = '''digraph LargeScaleMoEDeployment {
    rankdir=TB;
    compound=true;
    node [fontname="Courier New", fontsize=10];
    
    // Graph styling
    graph [bgcolor=white, margin=0.2];
    
    // Input node
    input_node [shape=rectangle, style=filled, fillcolor=lightblue,
                label="GPU: INPUT\nInput Layer\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, hidden=7168]"];
    
    // Layer 1 - Dense Layer (First 3 layers are dense)
    subgraph cluster_layer1_dense {
        label="Layer 1 - Dense (DP=4, TP=8)";
        style=dashed;
        color=blue;
        
        // MLA attention - Tensor Parallel across 8 GPUs
        subgraph cluster_mla_dense1 {
            label="MLA Attention - Tensor Parallel";
            style=dotted;
            color=green;
            
            // Q projection - TP split
            mla_q_dense1_0 [shape=ellipse, style=filled, fillcolor=lightyellow,
                           label="GPU: 0\nQ Proj (1/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            mla_q_dense1_1 [shape=ellipse, style=filled, fillcolor=lightyellow,
                           label="GPU: 1\nQ Proj (2/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            mla_q_dense1_2 [shape=ellipse, style=filled, fillcolor=lightyellow,
                           label="GPU: 2\nQ Proj (3/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            mla_q_dense1_3 [shape=ellipse, style=filled, fillcolor=lightyellow,
                           label="GPU: 3\nQ Proj (4/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            mla_q_dense1_4 [shape=ellipse, style=filled, fillcolor=lightyellow,
                           label="GPU: 4\nQ Proj (5/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            mla_q_dense1_5 [shape=ellipse, style=filled, fillcolor=lightyellow,
                           label="GPU: 5\nQ Proj (6/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            mla_q_dense1_6 [shape=ellipse, style=filled, fillcolor=lightyellow,
                           label="GPU: 6\nQ Proj (7/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            mla_q_dense1_7 [shape=ellipse, style=filled, fillcolor=lightyellow,
                           label="GPU: 7\nQ Proj (8/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            
            // K/V projection - TP split (MLA uses compressed latent)
            mla_kv_dense1_0 [shape=ellipse, style=filled, fillcolor=lightcoral,
                            label="GPU: 0\nK/V Proj (1/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56, kv_compressed=512]"];
            mla_kv_dense1_1 [shape=ellipse, style=filled, fillcolor=lightcoral,
                            label="GPU: 1\nK/V Proj (2/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56, kv_compressed=512]"];
            mla_kv_dense1_2 [shape=ellipse, style=filled, fillcolor=lightcoral,
                            label="GPU: 2\nK/V Proj (3/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56, kv_compressed=512]"];
            mla_kv_dense1_3 [shape=ellipse, style=filled, fillcolor=lightcoral,
                            label="GPU: 3\nK/V Proj (4/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56, kv_compressed=512]"];
            mla_kv_dense1_4 [shape=ellipse, style=filled, fillcolor=lightcoral,
                            label="GPU: 4\nK/V Proj (5/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56, kv_compressed=512]"];
            mla_kv_dense1_5 [shape=ellipse, style=filled, fillcolor=lightcoral,
                            label="GPU: 5\nK/V Proj (6/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56, kv_compressed=512]"];
            mla_kv_dense1_6 [shape=ellipse, style=filled, fillcolor=lightcoral,
                            label="GPU: 6\nK/V Proj (7/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56, kv_compressed=512]"];
            mla_kv_dense1_7 [shape=ellipse, style=filled, fillcolor=lightcoral,
                            label="GPU: 7\nK/V Proj (8/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56, kv_compressed=512]"];
            
            // Attention computation - each GPU computes partial attention
            attn_dense1_0 [shape=rectangle, style=filled, fillcolor=lightgreen,
                          label="GPU: 0\nAttention Compute\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            attn_dense1_1 [shape=rectangle, style=filled, fillcolor=lightgreen,
                          label="GPU: 1\nAttention Compute\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            attn_dense1_2 [shape=rectangle, style=filled, fillcolor=lightgreen,
                          label="GPU: 2\nAttention Compute\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            attn_dense1_3 [shape=rectangle, style=filled, fillcolor=lightgreen,
                          label="GPU: 3\nAttention Compute\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            attn_dense1_4 [shape=rectangle, style=filled, fillcolor=lightgreen,
                          label="GPU: 4\nAttention Compute\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            attn_dense1_5 [shape=rectangle, style=filled, fillcolor=lightgreen,
                          label="GPU: 5\nAttention Compute\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            attn_dense1_6 [shape=rectangle, style=filled, fillcolor=lightgreen,
                          label="GPU: 6\nAttention Compute\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            attn_dense1_7 [shape=rectangle, style=filled, fillcolor=lightgreen,
                          label="GPU: 7\nAttention Compute\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, heads=16, d_k=56]"];
            
            // Output projection - TP reduction
            out_proj_dense1_0 [shape=parallelogram, style=filled, fillcolor=lightblue,
                              label="GPU: 0\nOutput Proj (1/8)\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            out_proj_dense1_1 [shape=parallelogram, style=filled, fillcolor=lightblue,
                              label="GPU: 1\nOutput Proj (2/8)\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            out_proj_dense1_2 [shape=parallelogram, style=filled, fillcolor=lightblue,
                              label="GPU: 2\nOutput Proj (3/8)\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            out_proj_dense1_3 [shape=parallelogram, style=filled, fillcolor=lightblue,
                              label="GPU: 3\nOutput Proj (4/8)\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            out_proj_dense1_4 [shape=parallelogram, style=filled, fillcolor=lightblue,
                              label="GPU: 4\nOutput Proj (5/8)\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            out_proj_dense1_5 [shape=parallelogram, style=filled, fillcolor=lightblue,
                              label="GPU: 5\nOutput Proj (6/8)\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            out_proj_dense1_6 [shape=parallelogram, style=filled, fillcolor=lightblue,
                              label="GPU: 6\nOutput Proj (7/8)\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            out_proj_dense1_7 [shape=parallelogram, style=filled, fillcolor=lightblue,
                              label="GPU: 7\nOutput Proj (8/8)\nInput: [batch_size=32, seq_len=2048, heads=16, d_k=56]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            
            // All-reduce for attention output
            attn_allreduce_dense1 [shape=parallelogram, style=filled, fillcolor=orange,
                                  label="GPU: ALL\nAll-Reduce Attention\nInput: [batch_size=32, seq_len=2048, hidden=896]\nOutput: [batch_size=32, seq_len=2048, hidden=7168]"];
        }
        
        // MLP - Tensor Parallel
        subgraph cluster_mlp_dense1 {
            label="MLP - Tensor Parallel";
            style=dotted;
            color=purple;
            
            // First linear (column parallel)
            mlp_linear1_dense1_0 [shape=ellipse, style=filled, fillcolor=lightyellow,
                               label="GPU: 0\nMLP Linear1 (1/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_linear1_dense1_1 [shape=ellipse, style=filled, fillcolor=lightyellow,
                               label="GPU: 1\nMLP Linear1 (2/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_linear1_dense1_2 [shape=ellipse, style=filled, fillcolor=lightyellow,
                               label="GPU: 2\nMLP Linear1 (3/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_linear1_dense1_3 [shape=ellipse, style=filled, fillcolor=lightyellow,
                               label="GPU: 3\nMLP Linear1 (4/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_linear1_dense1_4 [shape=ellipse, style=filled, fillcolor=lightyellow,
                               label="GPU: 4\nMLP Linear1 (5/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_linear1_dense1_5 [shape=ellipse, style=filled, fillcolor=lightyellow,
                               label="GPU: 5\nMLP Linear1 (6/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_linear1_dense1_6 [shape=ellipse, style=filled, fillcolor=lightyellow,
                               label="GPU: 6\nMLP Linear1 (7/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_linear1_dense1_7 [shape=ellipse, style=filled, fillcolor=lightyellow,
                               label="GPU: 7\nMLP Linear1 (8/8)\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            
            // GELU activation
            mlp_gelu_dense1_0 [shape=rectangle, style=filled, fillcolor=lightgreen,
                            label="GPU: 0\nGELU\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_gelu_dense1_1 [shape=rectangle, style=filled, fillcolor=lightgreen,
                            label="GPU: 1\nGELU\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_gelu_dense1_2 [shape=rectangle, style=filled, fillcolor=lightgreen,
                            label="GPU: 2\nGELU\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_gelu_dense1_3 [shape=rectangle, style=filled, fillcolor=lightgreen,
                            label="GPU: 3\nGELU\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_gelu_dense1_4 [shape=rectangle, style=filled, fillcolor=lightgreen,
                            label="GPU: 4\nGELU\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_gelu_dense1_5 [shape=rectangle, style=filled, fillcolor=lightgreen,
                            label="GPU: 5\nGELU\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_gelu_dense1_6 [shape=rectangle, style=filled, fillcolor=lightgreen,
                            label="GPU: 6\nGELU\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            mlp_gelu_dense1_7 [shape=rectangle, style=filled, fillcolor=lightgreen,
                            label="GPU: 7\nGELU\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, ffn_hidden=2304]"];
            
            // Second linear (row parallel)
            mlp_linear2_dense1_0 [shape=parallelogram, style=filled, fillcolor=lightblue,
                               label="GPU: 0\nMLP Linear2 (1/8)\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            mlp_linear2_dense1_1 [shape=parallelogram, style=filled, fillcolor=lightblue,
                               label="GPU: 1\nMLP Linear2 (2/8)\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            mlp_linear2_dense1_2 [shape=parallelogram, style=filled, fillcolor=lightblue,
                               label="GPU: 2\nMLP Linear2 (3/8)\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            mlp_linear2_dense1_3 [shape=parallelogram, style=filled, fillcolor=lightblue,
                               label="GPU: 3\nMLP Linear2 (4/8)\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            mlp_linear2_dense1_4 [shape=parallelogram, style=filled, fillcolor=lightblue,
                               label="GPU: 4\nMLP Linear2 (5/8)\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            mlp_linear2_dense1_5 [shape=parallelogram, style=filled, fillcolor=lightblue,
                               label="GPU: 5\nMLP Linear2 (6/8)\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            mlp_linear2_dense1_6 [shape=parallelogram, style=filled, fillcolor=lightblue,
                               label="GPU: 6\nMLP Linear2 (7/8)\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            mlp_linear2_dense1_7 [shape=parallelogram, style=filled, fillcolor=lightblue,
                               label="GPU: 7\nMLP Linear2 (8/8)\nInput: [batch_size=32, seq_len=2048, ffn_hidden=2304]\nOutput: [batch_size=32, seq_len=2048, hidden=896]"];
            
            // All-reduce for MLP output
            mlp_allreduce_dense1 [shape=parallelogram, style=filled, fillcolor=orange,
                               label="GPU: ALL\nAll-Reduce MLP\nInput: [batch_size=32, seq_len=2048, hidden=896]\nOutput: [batch_size=32, seq_len=2048, hidden=7168]"];
        }
    }
    
    // Layer 4 - MoE Layer (First representative MoE layer)
    subgraph cluster_layer4_moe {
        label="Layer 4 - MoE Layer (EP=16, TP=8, DP=4)";
        style=dashed;
        color=red;
        
        // MLA attention - same as dense but across different GPUs due to EP
        subgraph cluster_mla_moe4 {
            label="MLA Attention - Expert Parallel";
            style=dotted;
            color=green;
            
            // Attention all-reduce within EP group
            attn_allreduce_moe4_ep0 [shape=parallelogram, style=filled, fillcolor=orange,
                                   label="GPU: EP0 Group\nAll-Reduce Attention\nInput: [batch_size=8, seq_len=2048, hidden=896]\nOutput: [batch_size=8, seq_len=2048, hidden=7168]"];
        }
        
        // Gating mechanism - decides which expert to route tokens to
        gating_moe4 [shape=parallelogram, style=filled, fillcolor=gold,
                    label="GPU: GATING\nExpert Gating\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [expert_id=128, tokens_per_expert=varies]"];
        
        // Token routing based on gating decisions
        token_routing_moe4 [shape=parallelogram, style=filled, fillcolor=lightcyan,
                           label="GPU: ROUTER\nToken Router\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [distributed to 128 experts]"];
        
        // Expert computation - each GPU hosts one expert
        subgraph cluster_experts_moe4 {
            label="Expert Computation - One Expert per GPU";
            style=dotted;
            color=brown;
            
            // Expert 0
            expert0_moe4_gate [shape=ellipse, style=filled, fillcolor=pink,
                             label="GPU: 0\nExpert 0 Gate\nInput: [tokens=varies, hidden=7168]\nOutput: [tokens=varies, hidden=7168, gate_weights=varies]"];
            expert0_moe4_up_proj [shape=ellipse, style=filled, fillcolor=pink,
                                label="GPU: 0\nExpert 0 Up-Proj\nInput: [tokens=varies, hidden=7168]\nOutput: [tokens=varies, ffn_hidden=18432]"];
            expert0_moe4_gate_proj [shape=ellipse, style=filled, fillcolor=pink,
                                  label="GPU: 0\nExpert 0 Gate-Proj\nInput: [tokens=varies, hidden=7168]\nOutput: [tokens=varies, ffn_hidden=18432]"];
            expert0_moe4_silu [shape=rectangle, style=filled, fillcolor=pink,
                             label="GPU: 0\nExpert 0 SiLU\nInput: [tokens=varies, ffn_hidden=18432]\nOutput: [tokens=varies, ffn_hidden=18432]"];
            expert0_moe4_mul [shape=rectangle, style=filled, fillcolor=pink,
                            label="GPU: 0\nExpert 0 Element-wise Mul\nInput: [tokens=varies, ffn_hidden=18432]\nOutput: [tokens=varies, ffn_hidden=18432]"];
            expert0_moe4_down_proj [shape=parallelogram, style=filled, fillcolor=pink,
                                  label="GPU: 0\nExpert 0 Down-Proj\nInput: [tokens=varies, ffn_hidden=18432]\nOutput: [tokens=varies, hidden=7168]"];
            
            // Expert 1
            expert1_moe4_gate [shape=ellipse, style=filled, fillcolor=lightpink,
                             label="GPU: 1\nExpert 1 Gate\nInput: [tokens=varies, hidden=7168]\nOutput: [tokens=varies, hidden=7168, gate_weights=varies]"];
            expert1_moe4_up_proj [shape=ellipse, style=filled, fillcolor=lightpink,
                                label="GPU: 1\nExpert 1 Up-Proj\nInput: [tokens=varies, hidden=7168]\nOutput: [tokens=varies, ffn_hidden=18432]"];
            expert1_moe4_gate_proj [shape=ellipse, style=filled, fillcolor=lightpink,
                                  label="GPU: 1\nExpert 1 Gate-Proj\nInput: [tokens=varies, hidden=7168]\nOutput: [tokens=varies, ffn_hidden=18432]"];
            expert1_moe4_silu [shape=rectangle, style=filled, fillcolor=lightpink,
                             label="GPU: 1\nExpert 1 SiLU\nInput: [tokens=varies, ffn_hidden=18432]\nOutput: [tokens=varies, ffn_hidden=18432]"];
            expert1_moe4_mul [shape=rectangle, style=filled, fillcolor=lightpink,
                            label="GPU: 1\nExpert 1 Element-wise Mul\nInput: [tokens=varies, ffn_hidden=18432]\nOutput: [tokens=varies, ffn_hidden=18432]"];
            expert1_moe4_down_proj [shape=parallelogram, style=filled, fillcolor=lightpink,
                                  label="GPU: 1\nExpert 1 Down-Proj\nInput: [tokens=varies, ffn_hidden=18432]\nOutput: [tokens=varies, hidden=7168]"];
        }
        
        // Expert aggregation - collect results from all experts
        expert_aggregation_moe4 [shape=parallelogram, style=filled, fillcolor=gold,
                               label="GPU: AGGREGATOR\nExpert Aggregation\nInput: [from 128 experts, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, hidden=7168]"];
    }
    
    // Simplified Layer 5 and 6 representations
    layer5_moe [shape=rectangle, style=filled, fillcolor=lightgray,
               label="Layer 5 - MoE Layer\nGPU: 0-127\nParallel execution across all GPUs\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, hidden=7168]"];
    
    layer6_moe [shape=rectangle, style=filled, fillcolor=lightgray,
               label="Layer 6 - MoE Layer\nGPU: 0-127\nParallel execution across all GPUs\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, hidden=7168]"];
    
    // Output node
    output_node [shape=rectangle, style=filled, fillcolor=lightcoral,
                label="GPU: OUTPUT\nOutput Layer\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, vocab_size]"];
    
    // Edges - connecting the computation flow
    // Input to Layer 1
    input_node -> mla_q_dense1_0 [style=solid];
    input_node -> mla_kv_dense1_0 [style=solid];
    
    // MLA connections
    mla_q_dense1_0 -> attn_dense1_0 [style=solid];
    mla_kv_dense1_0 -> attn_dense1_0 [style=solid];
    attn_dense1_0 -> out_proj_dense1_0 [style=solid];
    
    // All-reduce for attention
    out_proj_dense1_0 -> attn_allreduce_dense1 [style=solid];
    
    // MLP connections
    attn_allreduce_dense1 -> mlp_linear1_dense1_0 [style=solid];
    mlp_linear1_dense1_0 -> mlp_gelu_dense1_0 [style=solid];
    mlp_gelu_dense1_0 -> mlp_linear2_dense1_0 [style=solid];
    mlp_linear2_dense1_0 -> mlp_allreduce_dense1 [style=solid];
    
    // Layer 1 to Layer 4 (MoE)
    mlp_allreduce_dense1 -> gating_moe4 [style=solid];
    
    // Gating to routing (dashed for communication)
    gating_moe4 -> token_routing_moe4 [style=dashed];
    
    // Routing to experts (dashed for communication)
    token_routing_moe4 -> expert0_moe4_gate [style=dashed, label="Token routing"];
    token_routing_moe4 -> expert1_moe4_gate [style=dashed, label="Token routing"];
    
    // Expert computation flow
    expert0_moe4_gate -> expert0_moe4_up_proj [style=solid];
    expert0_moe4_gate -> expert0_moe4_gate_proj [style=solid];
    expert0_moe4_up_proj -> expert0_moe4_mul [style=solid];
    expert0_moe4_gate_proj -> expert0_moe4_silu [style=solid];
    expert0_moe4_silu -> expert0_moe4_mul [style=solid];
    expert0_moe4_mul -> expert0_moe4_down_proj [style=solid];
    
    expert1_moe4_gate -> expert1_moe4_up_proj [style=solid];
    expert1_moe4_up_proj -> expert1_moe4_mul [style=solid];
    expert1_moe4_gate_proj -> expert1_moe4_silu [style=solid];
    expert1_moe4_silu -> expert1_moe4_mul [style=solid];
    expert1_moe4_mul -> expert1_moe4_down_proj [style=solid];
    
    // Experts to aggregation (dashed for communication)
    expert0_moe4_down_proj -> expert_aggregation_moe4 [style=dashed, label="Expert result"];
    expert1_moe4_down_proj -> expert_aggregation_moe4 [style=dashed, label="Expert result"];
    
    // Layer connections
    expert_aggregation_moe4 -> layer5_moe [style=solid];
    layer5_moe -> layer6_moe [style=solid];
    layer6_moe -> output_node [style=solid];
    
    // Residual connections
    attn_allreduce_dense1 -> mlp_linear1_dense1_0 [style=dashed, label="Residual", constraint=false];
    expert_aggregation_moe4 -> layer5_moe [style=dashed, label="Residual", constraint=false];
}
'''
    
    return dot_content

def save_dag_files():
    """Save the DAG to both .dot and .svg files"""
    dag_content = generate_dag()
    
    # Save .dot file
    dot_path = "../outputs/2025-11-27-14-40-21/large_scale_moe_deployment.dot"
    with open(dot_path, 'w') as f:
        f.write(dag_content)
    
    # Generate SVG using graphviz
    import subprocess
    svg_path = "../outputs/2025-11-27-14-40-21/large_scale_moe_deployment.svg"
    try:
        subprocess.run(['dot', '-Tsvg', dot_path, '-o', svg_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not generate SVG file: {e}")
        svg_path = None
    
    return {
        "dot_file": dot_path,
        "svg_file": svg_path
    }

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs("../outputs/2025-11-27-14-40-21", exist_ok=True)
    
    # Generate and save DAG
    paths = save_dag_files()
    
    # Verify DAG is acyclic using the Extract Info From DAG tool
    from ExtractInfoFromDAG import extract_dag_info
    
    dag_info = extract_dag_info("../outputs/2025-11-27-14-40-21/large_scale_moe_deployment.dot")
    
    if dag_info["has_cycle"]:
        raise ValueError("Generated DAG contains cycles!")
    
    # Print paths for JSON output
    print(json.dumps(paths, indent=2))