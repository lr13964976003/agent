#!/usr/bin/env python3
"""
Generate DAGs for Ring Attention with Sequence Parallelism paper
Creates two DAGs:
1. Baseline: Tensor Parallel + Pipeline Parallel
2. Proposed: Ring Attention + Sequence Parallelism
"""

import os

# Create output directory
output_dir = "../outputs/2025-11-14-09-19-11"
os.makedirs(output_dir, exist_ok=True)

# Model configuration
BATCH_SIZE = 128
SEQ_LEN = 100000
HEADS = 32
HEAD_DIM = 128
HIDDEN_SIZE = 4096
MLP_HIDDEN_SIZE = 32768
NUM_LAYERS = 4
PRECISION = "bf16"

# GPU configuration for baseline
TP_DEGREE = 8
PP_DEGREE = 2
TOTAL_GPUS = 16

# GPU configuration for proposed
RING_DEGREE = 16
SEQ_PER_DEVICE = SEQ_LEN // RING_DEGREE

def generate_baseline_dag():
    """Generate baseline DAG: Tensor Parallel + Pipeline Parallel"""
    
    dot_content = '''digraph baseline_tensor_pipeline_parallel {
    rankdir=TB;
    node [shape=box, fontname="Arial"];
    
    // Global input
    input [label="Input\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: all GPUs", shape=ellipse];
    
    // Pipeline Stage 0: Devices 0-7, Layers 0-1
    subgraph cluster_pipeline_stage_0 {
        label="Pipeline Stage 0\\nDevices: 0-7\\nLayers: 0,1";
        style=dashed;
        
        // Layer 0
        subgraph cluster_layer_0 {
            label="Layer 0 (Pipeline Stage 0)";
            style=dashed;
            
            // Device 0-7 operations for Layer 0
            layer0_norm [label="LayerNorm\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: 0-7", shape=box];
            
            // QKV Projection (Column Parallel)
            layer0_q_proj [label="Q Projection\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, heads=32, head_dim=128]\\nGPU: 0-7 (TP: column split)", shape=box];
            layer0_k_proj [label="K Projection\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, heads=32, head_dim=128]\\nGPU: 0-7 (TP: column split)", shape=box];
            layer0_v_proj [label="V Projection\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, heads=32, head_dim=128]\\nGPU: 0-7 (TP: column split)", shape=box];
            
            // All-gather for KV
            layer0_allgather_k [label="AllGather K\\nInput: [batch_size=128, seq_len=100000, heads=4, head_dim=128]\\nOutput: [batch_size=128, seq_len=100000, heads=32, head_dim=128]\\nGPU: 0-7 (TP communication)", shape=ellipse];
            layer0_allgather_v [label="AllGather V\\nInput: [batch_size=128, seq_len=100000, heads=4, head_dim=128]\\nOutput: [batch_size=128, seq_len=100000, heads=32, head_dim=128]\\nGPU: 0-7 (TP communication)", shape=ellipse];
            
            // Attention computation
            layer0_attention [label="Multi-Head Attention\\nInput: [batch_size=128, seq_len=100000, heads=32, head_dim=128]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: 0-7", shape=box];
            
            // Output projection (Row Parallel)
            layer0_out_proj [label="Output Projection\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: 0-7 (TP: row split)", shape=box];
            
            // All-reduce for output
            layer0_allreduce [label="AllReduce\\nInput: [batch_size=128, seq_len=100000, hidden=512]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: 0-7 (TP communication)", shape=ellipse];
            
            // Residual connection
            layer0_residual [label="Residual Add\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: 0-7", shape=box];
            
            // LayerNorm 2
            layer0_norm2 [label="LayerNorm\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: 0-7", shape=box];
            
            // MLP (Column/Row Parallel)
            layer0_mlp_fc1 [label="MLP FC1\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, mlp_hidden=4096]\\nGPU: 0-7 (TP: column split)", shape=box];
            layer0_mlp_fc2 [label="MLP FC2\\nInput: [batch_size=128, seq_len=100000, mlp_hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=512]\\nGPU: 0-7 (TP: row split)", shape=box];
            layer0_mlp_allreduce [label="AllReduce\\nInput: [batch_size=128, seq_len=100000, hidden=512]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: 0-7 (TP communication)", shape=ellipse];
            layer0_mlp_residual [label="MLP Residual Add\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: 0-7", shape=box];
        }
        
        // Layer 1 (similar structure to Layer 0)
        subgraph cluster_layer_1 {
            label="Layer 1 (Pipeline Stage 0)";
            style=dashed;
            
            layer1_norm [label="LayerNorm\\nGPU: 0-7", shape=box];
            layer1_q_proj [label="Q Projection\\nGPU: 0-7", shape=box];
            layer1_k_proj [label="K Projection\\nGPU: 0-7", shape=box];
            layer1_v_proj [label="V Projection\\nGPU: 0-7", shape=box];
            layer1_allgather_k [label="AllGather K\\nGPU: 0-7", shape=ellipse];
            layer1_allgather_v [label="AllGather V\\nGPU: 0-7", shape=ellipse];
            layer1_attention [label="Multi-Head Attention\\nGPU: 0-7", shape=box];
            layer1_out_proj [label="Output Projection\\nGPU: 0-7", shape=box];
            layer1_allreduce [label="AllReduce\\nGPU: 0-7", shape=ellipse];
            layer1_residual [label="Residual Add\\nGPU: 0-7", shape=box];
            layer1_norm2 [label="LayerNorm\\nGPU: 0-7", shape=box];
            layer1_mlp_fc1 [label="MLP FC1\\nGPU: 0-7", shape=box];
            layer1_mlp_fc2 [label="MLP FC2\\nGPU: 0-7", shape=box];
            layer1_mlp_allreduce [label="AllReduce\\nGPU: 0-7", shape=ellipse];
            layer1_mlp_residual [label="MLP Residual Add\\nGPU: 0-7", shape=box];
        }
    }
    
    // Pipeline Stage 1: Devices 8-15, Layers 2-3
    subgraph cluster_pipeline_stage_1 {
        label="Pipeline Stage 1\\nDevices: 8-15\\nLayers: 2,3";
        style=dashed;
        
        // Pipeline communication
        pipeline_send_0_1 [label="Send\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: 0-7 → 8-15", shape=parallelogram];
        pipeline_recv_1_0 [label="Receive\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: 8-15", shape=parallelogram];
        
        // Layer 2
        subgraph cluster_layer_2 {
            label="Layer 2 (Pipeline Stage 1)";
            style=dashed;
            
            layer2_norm [label="LayerNorm\\nGPU: 8-15", shape=box];
            layer2_q_proj [label="Q Projection\\nGPU: 8-15", shape=box];
            layer2_k_proj [label="K Projection\\nGPU: 8-15", shape=box];
            layer2_v_proj [label="V Projection\\nGPU: 8-15", shape=box];
            layer2_allgather_k [label="AllGather K\\nGPU: 8-15", shape=ellipse];
            layer2_allgather_v [label="AllGather V\\nGPU: 8-15", shape=ellipse];
            layer2_attention [label="Multi-Head Attention\\nGPU: 8-15", shape=box];
            layer2_out_proj [label="Output Projection\\nGPU: 8-15", shape=box];
            layer2_allreduce [label="AllReduce\\nGPU: 8-15", shape=ellipse];
            layer2_residual [label="Residual Add\\nGPU: 8-15", shape=box];
            layer2_norm2 [label="LayerNorm\\nGPU: 8-15", shape=box];
            layer2_mlp_fc1 [label="MLP FC1\\nGPU: 8-15", shape=box];
            layer2_mlp_fc2 [label="MLP FC2\\nGPU: 8-15", shape=box];
            layer2_mlp_allreduce [label="AllReduce\\nGPU: 8-15", shape=ellipse];
            layer2_mlp_residual [label="MLP Residual Add\\nGPU: 8-15", shape=box];
        }
        
        // Layer 3
        subgraph cluster_layer_3 {
            label="Layer 3 (Pipeline Stage 1)";
            style=dashed;
            
            layer3_norm [label="LayerNorm\\nGPU: 8-15", shape=box];
            layer3_q_proj [label="Q Projection\\nGPU: 8-15", shape=box];
            layer3_k_proj [label="K Projection\\nGPU: 8-15", shape=box];
            layer3_v_proj [label="V Projection\\nGPU: 8-15", shape=box];
            layer3_allgather_k [label="AllGather K\\nGPU: 8-15", shape=ellipse];
            layer3_allgather_v [label="AllGather V\\nGPU: 8-15", shape=ellipse];
            layer3_attention [label="Multi-Head Attention\\nGPU: 8-15", shape=box];
            layer3_out_proj [label="Output Projection\\nGPU: 8-15", shape=box];
            layer3_allreduce [label="AllReduce\\nGPU: 8-15", shape=ellipse];
            layer3_residual [label="Residual Add\\nGPU: 8-15", shape=box];
            layer3_norm2 [label="LayerNorm\\nGPU: 8-15", shape=box];
            layer3_mlp_fc1 [label="MLP FC1\\nGPU: 8-15", shape=box];
            layer3_mlp_fc2 [label="MLP FC2\\nGPU: 8-15", shape=box];
            layer3_mlp_allreduce [label="AllReduce\\nGPU: 8-15", shape=ellipse];
            layer3_mlp_residual [label="MLP Residual Add\\nGPU: 8-15", shape=box];
        }
    }
    
    // Global output
    output [label="Output\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: all GPUs", shape=ellipse];
    
    // Connections
    input -> layer0_norm;
    
    // Layer 0 connections
    layer0_norm -> layer0_q_proj;
    layer0_norm -> layer0_k_proj;
    layer0_norm -> layer0_v_proj;
    layer0_k_proj -> layer0_allgather_k;
    layer0_v_proj -> layer0_allgather_v;
    layer0_q_proj -> layer0_attention;
    layer0_allgather_k -> layer0_attention;
    layer0_allgather_v -> layer0_attention;
    layer0_attention -> layer0_out_proj;
    layer0_out_proj -> layer0_allreduce;
    layer0_allreduce -> layer0_residual;
    input -> layer0_residual [style=dashed];
    layer0_residual -> layer0_norm2;
    layer0_norm2 -> layer0_mlp_fc1;
    layer0_mlp_fc1 -> layer0_mlp_fc2;
    layer0_mlp_fc2 -> layer0_mlp_allreduce;
    layer0_mlp_allreduce -> layer0_mlp_residual;
    layer0_residual -> layer0_mlp_residual [style=dashed];
    
    // Layer 1 connections (simplified)
    layer0_mlp_residual -> layer1_norm;
    layer1_norm -> layer1_q_proj;
    layer1_norm -> layer1_k_proj;
    layer1_norm -> layer1_v_proj;
    layer1_k_proj -> layer1_allgather_k;
    layer1_v_proj -> layer1_allgather_v;
    layer1_q_proj -> layer1_attention;
    layer1_allgather_k -> layer1_attention;
    layer1_allgather_v -> layer1_attention;
    layer1_attention -> layer1_out_proj;
    layer1_out_proj -> layer1_allreduce;
    layer1_allreduce -> layer1_residual;
    layer0_mlp_residual -> layer1_residual [style=dashed];
    layer1_residual -> layer1_norm2;
    layer1_norm2 -> layer1_mlp_fc1;
    layer1_mlp_fc1 -> layer1_mlp_fc2;
    layer1_mlp_fc2 -> layer1_mlp_allreduce;
    layer1_mlp_allreduce -> layer1_mlp_residual;
    layer1_residual -> layer1_mlp_residual [style=dashed];
    
    // Pipeline communication
    layer1_mlp_residual -> pipeline_send_0_1;
    pipeline_send_0_1 -> pipeline_recv_1_0;
    
    // Layer 2 connections (simplified)
    pipeline_recv_1_0 -> layer2_norm;
    layer2_norm -> layer2_q_proj;
    layer2_norm -> layer2_k_proj;
    layer2_norm -> layer2_v_proj;
    layer2_k_proj -> layer2_allgather_k;
    layer2_v_proj -> layer2_allgather_v;
    layer2_q_proj -> layer2_attention;
    layer2_allgather_k -> layer2_attention;
    layer2_allgather_v -> layer2_attention;
    layer2_attention -> layer2_out_proj;
    layer2_out_proj -> layer2_allreduce;
    layer2_allreduce -> layer2_residual;
    pipeline_recv_1_0 -> layer2_residual [style=dashed];
    layer2_residual -> layer2_norm2;
    layer2_norm2 -> layer2_mlp_fc1;
    layer2_mlp_fc1 -> layer2_mlp_fc2;
    layer2_mlp_fc2 -> layer2_mlp_allreduce;
    layer2_mlp_allreduce -> layer2_mlp_residual;
    layer2_residual -> layer2_mlp_residual [style=dashed];
    
    // Layer 3 connections (simplified)
    layer2_mlp_residual -> layer3_norm;
    layer3_norm -> layer3_q_proj;
    layer3_norm -> layer3_k_proj;
    layer3_norm -> layer3_v_proj;
    layer3_k_proj -> layer3_allgather_k;
    layer3_v_proj -> layer3_allgather_v;
    layer3_q_proj -> layer3_attention;
    layer3_allgather_k -> layer3_attention;
    layer3_allgather_v -> layer3_attention;
    layer3_attention -> layer3_out_proj;
    layer3_out_proj -> layer3_allreduce;
    layer3_allreduce -> layer3_residual;
    layer2_mlp_residual -> layer3_residual [style=dashed];
    layer3_residual -> layer3_norm2;
    layer3_norm2 -> layer3_mlp_fc1;
    layer3_mlp_fc1 -> layer3_mlp_fc2;
    layer3_mlp_fc2 -> layer3_mlp_allreduce;
    layer3_mlp_allreduce -> layer3_mlp_residual;
    layer3_residual -> layer3_mlp_residual [style=dashed];
    
    layer3_mlp_residual -> output;
}
'''
    
    with open(f"{output_dir}/baseline_tensor_pipeline_parallel.dot", "w") as f:
        f.write(dot_content)
    
    return f"{output_dir}/baseline_tensor_pipeline_parallel.dot"

def generate_proposed_dag():
    """Generate proposed DAG: Ring Attention + Sequence Parallelism"""
    
    dot_content = '''digraph proposed_ring_attention_sequence_parallel {
    rankdir=TB;
    node [shape=box, fontname="Arial"];
    
    // Global input
    input [label="Input\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: all GPUs", shape=ellipse];
    
    // Input split for sequence parallelism
    split_input [label="Split Sequence\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=6250, hidden=4096]\\nGPU: all GPUs (seq parallel)", shape=parallelogram];
    
    // Repeat for each GPU device
    devices = [f"device_{i}" for i in range(16)]
    
    // Device 0 as representative
    subgraph cluster_device_0 {
        label="Device 0 (Ring Stage 0)\\nSequence: 0-6250";
        style=dashed;
        
        // Input for device 0
        input_0 [label="Local Input\\nInput: [batch_size=128, seq_len=6250, hidden=4096]\\nOutput: [batch_size=128, seq_len=6250, hidden=4096]\\nGPU: 0", shape=ellipse];
        
        // Layer 0 for device 0
        subgraph cluster_device0_layer0 {
            label="Layer 0 (Device 0)";
            style=dashed;
            
            layer0_norm_0 [label="LayerNorm\\nGPU: 0", shape=box];
            layer0_q_proj_0 [label="Q Projection\\nInput: [batch_size=128, seq_len=6250, hidden=4096]\\nOutput: [batch_size=128, seq_len=6250, heads=32, head_dim=128]\\nGPU: 0", shape=box];
            layer0_k_proj_0 [label="K Projection\\nInput: [batch_size=128, seq_len=6250, hidden=4096]\\nOutput: [batch_size=128, seq_len=6250, heads=32, head_dim=128]\\nGPU: 0", shape=box];
            layer0_v_proj_0 [label="V Projection\\nInput: [batch_size=128, seq_len=6250, hidden=4096]\\nOutput: [batch_size=128, seq_len=6250, heads=32, head_dim=128]\\nGPU: 0", shape=box];
            
            // Ring attention stages
            stage0_recv_kv [label="Recv KV\\nInput: [batch_size=128, seq_len=6250, heads=32, head_dim=128]\\nOutput: [batch_size=128, seq_len=6250, heads=32, head_dim=128]\\nGPU: 0 (from GPU 15)", shape=ellipse];
            stage0_compute [label="Attention Step\\nInput: [batch_size=128, seq_len=6250, heads=32, head_dim=128]\\nOutput: [batch_size=128, seq_len=6250, hidden=4096]\\nGPU: 0", shape=box];
            stage0_send_kv [label="Send KV\\nInput: [batch_size=128, seq_len=6250, heads=32, head_dim=128]\\nOutput: [batch_size=128, seq_len=6250, heads=32, head_dim=128]\\nGPU: 0 (to GPU 1)", shape=ellipse];
            
            // Accumulation
            stage0_accumulate [label="Accumulate\\nInput: [batch_size=128, seq_len=6250, hidden=4096]\\nOutput: [batch_size=128, seq_len=6250, hidden=4096]\\nGPU: 0", shape=box];
            
            // Output projection
            layer0_out_proj_0 [label="Output Projection\\nGPU: 0", shape=box];
            layer0_residual_0 [label="Residual Add\\nGPU: 0", shape=box];
            layer0_norm2_0 [label="LayerNorm\\nGPU: 0", shape=box];
            layer0_mlp_fc1_0 [label="MLP FC1\\nGPU: 0", shape=box];
            layer0_mlp_fc2_0 [label="MLP FC2\\nGPU: 0", shape=box];
            layer0_mlp_residual_0 [label="MLP Residual Add\\nGPU: 0", shape=box];
        }
        
        // Layer 1 for device 0
        subgraph cluster_device0_layer1 {
            label="Layer 1 (Device 0)";
            style=dashed;
            
            layer1_norm_0 [label="LayerNorm\\nGPU: 0", shape=box];
            layer1_q_proj_0 [label="Q Projection\\nGPU: 0", shape=box];
            layer1_k_proj_0 [label="K Projection\\nGPU: 0", shape=box];
            layer1_v_proj_0 [label="V Projection\\nGPU: 0", shape=box];
            
            layer1_ring_stage0 [label="Ring Attention\\nGPU: 0", shape=box];
            layer1_accumulate_0 [label="Accumulate\\nGPU: 0", shape=box];
            layer1_out_proj_0 [label="Output Projection\\nGPU: 0", shape=box];
            layer1_residual_0 [label="Residual Add\\nGPU: 0", shape=box];
            layer1_norm2_0 [label="LayerNorm\\nGPU: 0", shape=box];
            layer1_mlp_fc1_0 [label="MLP FC1\\nGPU: 0", shape=box];
            layer1_mlp_fc2_0 [label="MLP FC2\\nGPU: 0", shape=box];
            layer1_mlp_residual_0 [label="MLP Residual Add\\nGPU: 0", shape=box];
        }
        
        // Layer 2 for device 0
        subgraph cluster_device0_layer2 {
            label="Layer 2 (Device 0)";
            style=dashed;
            
            layer2_norm_0 [label="LayerNorm\\nGPU: 0", shape=box];
            layer2_q_proj_0 [label="Q Projection\\nGPU: 0", shape=box];
            layer2_k_proj_0 [label="K Projection\\nGPU: 0", shape=box];
            layer2_v_proj_0 [label="V Projection\\nGPU: 0", shape=box];
            
            layer2_ring_stage0 [label="Ring Attention\\nGPU: 0", shape=box];
            layer2_accumulate_0 [label="Accumulate\\nGPU: 0", shape=box];
            layer2_out_proj_0 [label="Output Projection\\nGPU: 0", shape=box];
            layer2_residual_0 [label="Residual Add\\nGPU: 0", shape=box];
            layer2_norm2_0 [label="LayerNorm\\nGPU: 0", shape=box];
            layer2_mlp_fc1_0 [label="MLP FC1\\nGPU: 0", shape=box];
            layer2_mlp_fc2_0 [label="MLP FC2\\nGPU: 0", shape=box];
            layer2_mlp_residual_0 [label="MLP Residual Add\\nGPU: 0", shape=box];
        }
        
        // Layer 3 for device 0
        subgraph cluster_device0_layer3 {
            label="Layer 3 (Device 0)";
            style=dashed;
            
            layer3_norm_0 [label="LayerNorm\\nGPU: 0", shape=box];
            layer3_q_proj_0 [label="Q Projection\\nGPU: 0", shape=box];
            layer3_k_proj_0 [label="K Projection\\nGPU: 0", shape=box];
            layer3_v_proj_0 [label="V Projection\\nGPU: 0", shape=box];
            
            layer3_ring_stage0 [label="Ring Attention\\nGPU: 0", shape=box];
            layer3_accumulate_0 [label="Accumulate\\nGPU: 0", shape=box];
            layer3_out_proj_0 [label="Output Projection\\nGPU: 0", shape=box];
            layer3_residual_0 [label="Residual Add\\nGPU: 0", shape=box];
            layer3_norm2_0 [label="LayerNorm\\nGPU: 0", shape=box];
            layer3_mlp_fc1_0 [label="MLP FC1\\nGPU: 0", shape=box];
            layer3_mlp_fc2_0 [label="MLP FC2\\nGPU: 0", shape=box];
            layer3_mlp_residual_0 [label="MLP Residual Add\\nGPU: 0", shape=box];
        }
        
        output_0 [label="Local Output\\nInput: [batch_size=128, seq_len=6250, hidden=4096]\\nOutput: [batch_size=128, seq_len=6250, hidden=4096]\\nGPU: 0", shape=ellipse];
    }
    
    // Aggregation
    aggregate_output [label="Aggregate Sequence\\nInput: [batch_size=128, seq_len=6250, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: all GPUs (seq parallel)", shape=parallelogram];
    
    // Global output
    final_output [label="Output\\nInput: [batch_size=128, seq_len=100000, hidden=4096]\\nOutput: [batch_size=128, seq_len=100000, hidden=4096]\\nGPU: all GPUs", shape=ellipse];
    
    // Connections for device 0
    input -> split_input;
    split_input -> input_0;
    
    input_0 -> layer0_norm_0;
    layer0_norm_0 -> layer0_q_proj_0;
    layer0_norm_0 -> layer0_k_proj_0;
    layer0_norm_0 -> layer0_v_proj_0;
    
    layer0_k_proj_0 -> stage0_compute;
    layer0_v_proj_0 -> stage0_compute;
    layer0_q_proj_0 -> stage0_compute;
    stage0_recv_kv -> stage0_compute;
    stage0_compute -> stage0_accumulate;
    stage0_compute -> stage0_send_kv;
    
    stage0_accumulate -> layer0_out_proj_0;
    layer0_out_proj_0 -> layer0_residual_0;
    input_0 -> layer0_residual_0 [style=dashed];
    layer0_residual_0 -> layer0_norm2_0;
    layer0_norm2_0 -> layer0_mlp_fc1_0;
    layer0_mlp_fc1_0 -> layer0_mlp_fc2_0;
    layer0_mlp_fc2_0 -> layer0_mlp_residual_0;
    layer0_residual_0 -> layer0_mlp_residual_0 [style=dashed];
    
    // Continue with remaining layers for device 0
    layer0_mlp_residual_0 -> layer1_norm_0;
    layer1_norm_0 -> layer1_q_proj_0;
    layer1_norm_0 -> layer1_k_proj_0;
    layer1_norm_0 -> layer1_v_proj_0;
    layer1_q_proj_0 -> layer1_ring_stage0;
    layer1_k_proj_0 -> layer1_ring_stage0;
    layer1_v_proj_0 -> layer1_ring_stage0;
    layer1_ring_stage0 -> layer1_accumulate_0;
    layer1_accumulate_0 -> layer1_out_proj_0;
    layer1_out_proj_0 -> layer1_residual_0;
    layer0_mlp_residual_0 -> layer1_residual_0 [style=dashed];
    layer1_residual_0 -> layer1_norm2_0;
    layer1_norm2_0 -> layer1_mlp_fc1_0;
    layer1_mlp_fc1_0 -> layer1_mlp_fc2_0;
    layer1_mlp_fc2_0 -> layer1_mlp_residual_0;
    layer1_residual_0 -> layer1_mlp_residual_0 [style=dashed];
    
    layer1_mlp_residual_0 -> layer2_norm_0;
    layer2_norm_0 -> layer2_q_proj_0;
    layer2_norm_0 -> layer2_k_proj_0;
    layer2_norm_0 -> layer2_v_proj_0;
    layer2_q_proj_0 -> layer2_ring_stage0;
    layer2_k_proj_0 -> layer2_ring_stage0;
    layer2_v_proj_0 -> layer2_ring_stage0;
    layer2_ring_stage0 -> layer2_accumulate_0;
    layer2_accumulate_0 -> layer2_out_proj_0;
    layer2_out_proj_0 -> layer2_residual_0;
    layer1_mlp_residual_0 -> layer2_residual_0 [style=dashed];
    layer2_residual_0 -> layer2_norm2_0;
    layer2_norm2_0 -> layer2_mlp_fc1_0;
    layer2_mlp_fc1_0 -> layer2_mlp_fc2_0;
    layer2_mlp_fc2_0 -> layer2_mlp_residual_0;
    layer2_residual_0 -> layer2_mlp_residual_0 [style=dashed];
    
    layer2_mlp_residual_0 -> layer3_norm_0;
    layer3_norm_0 -> layer3_q_proj_0;
    layer3_norm_0 -> layer3_k_proj_0;
    layer3_norm_0 -> layer3_v_proj_0;
    layer3_q_proj_0 -> layer3_ring_stage0;
    layer3_k_proj_0 -> layer3_ring_stage0;
    layer3_v_proj_0 -> layer3_ring_stage0;
    layer3_ring_stage0 -> layer3_accumulate_0;
    layer3_accumulate_0 -> layer3_out_proj_0;
    layer3_out_proj_0 -> layer3_residual_0;
    layer2_mlp_residual_0 -> layer3_residual_0 [style=dashed];
    layer3_residual_0 -> layer3_norm2_0;
    layer3_norm2_0 -> layer3_mlp_fc1_0;
    layer3_mlp_fc1_0 -> layer3_mlp_fc2_0;
    layer3_mlp_fc2_0 -> layer3_mlp_residual_0;
    layer3_residual_0 -> layer3_mlp_residual_0 [style=dashed];
    
    layer3_mlp_residual_0 -> output_0;
    output_0 -> aggregate_output;
    
    // Ring connections (conceptual)
    edge [style=dashed, color=red];
    stage0_send_kv -> stage0_recv_kv [label="ring", constraint=false];
    
    aggregate_output -> final_output;
}
'''
    
    with open(f"{output_dir}/proposed_ring_attention_sequence_parallel.dot", "w") as f:
        f.write(dot_content)
    
    return f"{output_dir}/proposed_ring_attention_sequence_parallel.dot"

if __name__ == "__main__":
    baseline_path = generate_baseline_dag()
    proposed_path = generate_proposed_dag()
    
    print(f"Generated baseline DAG: {baseline_path}")
    print(f"Generated proposed DAG: {proposed_path}")
    
    # Generate SVG files using Graphviz
    import subprocess
    
    for dot_file in [baseline_path, proposed_path]:
        svg_file = dot_file.replace('.dot', '.svg')
        try:
            subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], check=True)
            print(f"Generated SVG: {svg_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating SVG for {dot_file}: {e}")
        except FileNotFoundError:
            print("Graphviz 'dot' not found. Install graphviz to generate SVG files.")