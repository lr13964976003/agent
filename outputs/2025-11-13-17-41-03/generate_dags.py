#!/usr/bin/env python3
"""
Generate DAGs for baseline and proposed transformer models
- Baseline: Tensor Parallel + Pipeline Parallel (16 GPUs)
- Proposed: Ring Attention + Sequence Parallel (16 GPUs)
"""

import os
from typing import List, Dict, Tuple

class DAGGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def tensor_split_description(self, full_dim: int, num_splits: int, split_type: str) -> str:
        """Generate tensor dimension description after splitting"""
        if split_type == "column":
            return f"{full_dim // num_splits}"
        elif split_type == "row":
            return f"{full_dim}"
        return str(full_dim)
    
    def create_baseline_dag(self) -> str:
        """Create DAG for baseline: Tensor Parallel + Pipeline Parallel"""
        dot_content = """digraph G {
    rankdir=TB;
    node [shape=rectangle, style=filled];
    
    // Title
    label="Baseline: Tensor Parallel (TP=8) + Pipeline Parallel (PP=2)";
    labelloc="t";
    fontsize=24;
    
    // Input node
    input [shape=ellipse, label="Input\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: Host"];
    
    // ===== Pipeline Stage 0 (GPUs 0-7) =====
    subgraph cluster_stage0 {
        label="Pipeline Stage 0\nGPUs 0-7";
        style=dashed;
        
        // Embedding on GPU 0
        embedding [label="Embedding\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0"];
        
        // Layer 0
        subgraph cluster_layer0 {
            label="Layer 0 (Stage 0)";
            style=dotted;
            
            // Layer Norm 0
            ln0 [label="LayerNorm\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0-7"];
            
            // QKV projections (column parallel)
            q_proj_0 [label="Q Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 0-7 (column split)"];
            k_proj_0 [label="K Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 0-7 (column split)"];
            v_proj_0 [label="V Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 0-7 (column split)"];
            
            // Multi-head attention
            mha_0 [label="Multi-Head Attention\nInput QKV: [batch=1024, seq=10000, hidden=1024]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 0-7"];
            
            // Output projection (row parallel)
            out_proj_0 [label="Output Projection\nInput: [batch=1024, seq=10000, hidden=1024]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0-7 (row split)"];
            
            // Residual
            residual0 [shape=parallelogram, label="Add\nInput: [batch=1024, seq=10000, hidden=8192] x2\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0-7"];
            
            // MLP
            gate_proj_0 [label="Gate Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 0-7 (column split)"];
            up_proj_0 [label="Up Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 0-7 (column split)"];
            silu_0 [label="SiLU\nInput: [batch=1024, seq=10000, hidden=4096]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 0-7"];
            mul_0 [label="Elementwise Mul\nInput: [batch=1024, seq=10000, hidden=4096] x2\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 0-7"];
            down_proj_0 [label="Down Projection\nInput: [batch=1024, seq=10000, hidden=4096]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0-7 (row split)"];
            
            // MLP residual
            residual_mlp0 [shape=parallelogram, label="Add\nInput: [batch=1024, seq=10000, hidden=8192] x2\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0-7"];
        }
        
        // Layer 1 (similar to layer 0)
        subgraph cluster_layer1 {
            label="Layer 1 (Stage 0)";
            style=dotted;
            
            ln1 [label="LayerNorm\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0-7"];
            
            q_proj_1 [label="Q Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 0-7 (column split)"];
            k_proj_1 [label="K Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 0-7 (column split)"];
            v_proj_1 [label="V Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 0-7 (column split)"];
            
            mha_1 [label="Multi-Head Attention\nInput QKV: [batch=1024, seq=10000, hidden=1024]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 0-7"];
            out_proj_1 [label="Output Projection\nInput: [batch=1024, seq=10000, hidden=1024]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0-7 (row split)"];
            residual1 [shape=parallelogram, label="Add\nInput: [batch=1024, seq=10000, hidden=8192] x2\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0-7"];
            
            gate_proj_1 [label="Gate Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 0-7 (column split)"];
            up_proj_1 [label="Up Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 0-7 (column split)"];
            silu_1 [label="SiLU\nInput: [batch=1024, seq=10000, hidden=4096]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 0-7"];
            mul_1 [label="Elementwise Mul\nInput: [batch=1024, seq=10000, hidden=4096] x2\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 0-7"];
            down_proj_1 [label="Down Projection\nInput: [batch=1024, seq=10000, hidden=4096]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0-7 (row split)"];
            residual_mlp1 [shape=parallelogram, label="Add\nInput: [batch=1024, seq=10000, hidden=8192] x2\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 0-7"];
        }
    }
    
    // ===== Pipeline Stage 1 (GPUs 8-15) =====
    subgraph cluster_stage1 {
        label="Pipeline Stage 1\nGPUs 8-15";
        style=dashed;
        
        // Pipeline communication
        pipeline_comm [shape=ellipse, label="Pipeline Communication\nSend/Receive activations\nGPU 7 -> GPU 8"];
        
        // Layer 2
        subgraph cluster_layer2 {
            label="Layer 2 (Stage 1)";
            style=dotted;
            
            ln2 [label="LayerNorm\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 8-15"];
            
            q_proj_2 [label="Q Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 8-15 (column split)"];
            k_proj_2 [label="K Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 8-15 (column split)"];
            v_proj_2 [label="V Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 8-15 (column split)"];
            
            mha_2 [label="Multi-Head Attention\nInput QKV: [batch=1024, seq=10000, hidden=1024]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 8-15"];
            out_proj_2 [label="Output Projection\nInput: [batch=1024, seq=10000, hidden=1024]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 8-15 (row split)"];
            residual2 [shape=parallelogram, label="Add\nInput: [batch=1024, seq=10000, hidden=8192] x2\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 8-15"];
            
            gate_proj_2 [label="Gate Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 8-15 (column split)"];
            up_proj_2 [label="Up Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 8-15 (column split)"];
            silu_2 [label="SiLU\nInput: [batch=1024, seq=10000, hidden=4096]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 8-15"];
            mul_2 [label="Elementwise Mul\nInput: [batch=1024, seq=10000, hidden=4096] x2\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 8-15"];
            down_proj_2 [label="Down Projection\nInput: [batch=1024, seq=10000, hidden=4096]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 8-15 (row split)"];
            residual_mlp2 [shape=parallelogram, label="Add\nInput: [batch=1024, seq=10000, hidden=8192] x2\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 8-15"];
        }
        
        // Layer 3
        subgraph cluster_layer3 {
            label="Layer 3 (Stage 1)";
            style=dotted;
            
            ln3 [label="LayerNorm\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 8-15"];
            
            q_proj_3 [label="Q Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 8-15 (column split)"];
            k_proj_3 [label="K Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 8-15 (column split)"];
            v_proj_3 [label="V Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 8-15 (column split)"];
            
            mha_3 [label="Multi-Head Attention\nInput QKV: [batch=1024, seq=10000, hidden=1024]\nOutput: [batch=1024, seq=10000, hidden=1024]\nGPU: 8-15"];
            out_proj_3 [label="Output Projection\nInput: [batch=1024, seq=10000, hidden=1024]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 8-15 (row split)"];
            residual3 [shape=parallelogram, label="Add\nInput: [batch=1024, seq=10000, hidden=8192] x2\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 8-15"];
            
            gate_proj_3 [label="Gate Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 8-15 (column split)"];
            up_proj_3 [label="Up Projection\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 8-15 (column split)"];
            silu_3 [label="SiLU\nInput: [batch=1024, seq=10000, hidden=4096]\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 8-15"];
            mul_3 [label="Elementwise Mul\nInput: [batch=1024, seq=10000, hidden=4096] x2\nOutput: [batch=1024, seq=10000, hidden=4096]\nGPU: 8-15"];
            down_proj_3 [label="Down Projection\nInput: [batch=1024, seq=10000, hidden=4096]\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 8-15 (row split)"];
            residual_mlp3 [shape=parallelogram, label="Add\nInput: [batch=1024, seq=10000, hidden=8192] x2\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: 8-15"];
        }
    }
    
    // Final output
    output [shape=ellipse, label="Output\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, vocab_size]\nGPU: 15"];
    
    // Connections
    input -> embedding;
    embedding -> ln0;
    ln0 -> q_proj_0;
    ln0 -> k_proj_0;
    ln0 -> v_proj_0;
    q_proj_0 -> mha_0;
    k_proj_0 -> mha_0;
    v_proj_0 -> mha_0;
    mha_0 -> out_proj_0;
    out_proj_0 -> residual0;
    embedding -> residual0;
    residual0 -> gate_proj_0;
    residual0 -> up_proj_0;
    gate_proj_0 -> silu_0;
    up_proj_0 -> mul_0;
    silu_0 -> mul_0;
    mul_0 -> down_proj_0;
    down_proj_0 -> residual_mlp0;
    residual0 -> residual_mlp0;
    
    residual_mlp0 -> ln1;
    ln1 -> q_proj_1;
    ln1 -> k_proj_1;
    ln1 -> v_proj_1;
    q_proj_1 -> mha_1;
    k_proj_1 -> mha_1;
    v_proj_1 -> mha_1;
    mha_1 -> out_proj_1;
    out_proj_1 -> residual1;
    residual_mlp0 -> residual1;
    residual1 -> gate_proj_1;
    residual1 -> up_proj_1;
    gate_proj_1 -> silu_1;
    up_proj_1 -> mul_1;
    silu_1 -> mul_1;
    mul_1 -> down_proj_1;
    down_proj_1 -> residual_mlp1;
    residual1 -> residual_mlp1;
    
    residual_mlp1 -> pipeline_comm;
    pipeline_comm -> ln2;
    
    ln2 -> q_proj_2;
    ln2 -> k_proj_2;
    ln2 -> v_proj_2;
    q_proj_2 -> mha_2;
    k_proj_2 -> mha_2;
    v_proj_2 -> mha_2;
    mha_2 -> out_proj_2;
    out_proj_2 -> residual2;
    pipeline_comm -> residual2;
    residual2 -> gate_proj_2;
    residual2 -> up_proj_2;
    gate_proj_2 -> silu_2;
    up_proj_2 -> mul_2;
    silu_2 -> mul_2;
    mul_2 -> down_proj_2;
    down_proj_2 -> residual_mlp2;
    residual2 -> residual_mlp2;
    
    residual_mlp2 -> ln3;
    ln3 -> q_proj_3;
    ln3 -> k_proj_3;
    ln3 -> v_proj_3;
    q_proj_3 -> mha_3;
    k_proj_3 -> mha_3;
    v_proj_3 -> mha_3;
    mha_3 -> out_proj_3;
    out_proj_3 -> residual3;
    residual_mlp2 -> residual3;
    residual3 -> gate_proj_3;
    residual3 -> up_proj_3;
    gate_proj_3 -> silu_3;
    up_proj_3 -> mul_3;
    silu_3 -> mul_3;
    mul_3 -> down_proj_3;
    down_proj_3 -> residual_mlp3;
    residual3 -> residual_mlp3;
    
    residual_mlp3 -> output;
}
"""
        return dot_content
    
    def create_proposed_dag(self) -> str:
        """Create DAG for proposed: Ring Attention + Sequence Parallelism"""
        dot_content = """digraph G {
    rankdir=TB;
    node [shape=rectangle, style=filled];
    
    // Title
    label="Proposed: Ring Attention + Sequence Parallelism (16 devices)";
    labelloc="t";
    fontsize=24;
    
    // Input node
    input [shape=ellipse, label="Input\nInput: [batch=1024, seq=10000, hidden=8192]\nGPU: Host"];
    
    // Sequence split across devices
    sequence_split [shape=parallelogram, label="Sequence Split\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=625, hidden=8192] x 16\nGPU: All 16"];
    
    // ===== Layer 0 =====
    subgraph cluster_layer0 {
        label="Layer 0 (All Devices)";
        style=dashed;
        
        // Loop through all devices
        subgraph cluster_device0 {
            label="GPU 0 (Seq 0-624)";
            style=dotted;
            
            // Embedding for device 0
            emb0 [label="Embedding\nInput: [batch=1024, seq=625, hidden=8192]\nOutput: [batch=1024, seq=625, hidden=8192]\nGPU: 0"];
            
            // Layer Norm
            ln0_0 [label="LayerNorm\nInput: [batch=1024, seq=625, hidden=8192]\nOutput: [batch=1024, seq=625, hidden=8192]\nGPU: 0"];
            
            // QKV projections (full parameters, local computation)
            q_proj0_0 [label="Q Projection\nInput: [batch=1024, seq=625, hidden=8192]\nOutput: [batch=1024, seq=625, hidden=8192]\nGPU: 0"];
            k_proj0_0 [label="K Projection\nInput: [batch=1024, seq=625, hidden=8192]\nOutput: [batch=1024, seq=625, hidden=8192]\nGPU: 0"];
            v_proj0_0 [label="V Projection\nInput: [batch=1024, seq=625, hidden=8192]\nOutput: [batch=1024, seq=625, hidden=8192]\nGPU: 0"];
            
            // Ring attention components
            ring_attn0_0 [label="Local Q Computation\nInput: [batch=1024, seq=625, hidden=8192]\nOutput: [batch=1024, seq=625, heads=16, dim=512]\nGPU: 0"];
            
            // KV ring exchange for 16 stages
            kv_send0_0 [shape=ellipse, label="Send KV\nInput: [batch=1024, seq=625, hidden=8192]\nOutput: → GPU 1\nGPU: 0"];
            kv_recv0_0 [shape=ellipse, label="Recv KV\nInput: ← GPU 15\nOutput: [batch=1024, seq=625, hidden=8192]\nGPU: 0"];
            
            // Attention computation
            attn0_0 [label="Attention\nInput: Q[batch=1024, seq=625, heads=16, dim=512]\n       K[batch=1024, seq=625, heads=16, dim=512]\n       V[batch=1024, seq=625, heads=16, dim=512]\nOutput: [batch=1024, seq=625, heads=16, dim=512]\nGPU: 0"];
            
            // Output projection
            out_proj0_0 [label="Output Projection\nInput: [batch=1024, seq=625, heads=16, dim=512]\nOutput: [batch=1024, seq=625, hidden=8192]\nGPU: 0"];
            
            // Residual
            residual0_0 [shape=parallelogram, label="Add\nInput: [batch=1024, seq=625, hidden=8192] x2\nOutput: [batch=1024, seq=625, hidden=8192]\nGPU: 0"];
            
            // MLP
            gate0_0 [label="Gate Projection\nInput: [batch=1024, seq=625, hidden=8192]\nOutput: [batch=1024, seq=625, hidden=32768]\nGPU: 0"];
            up0_0 [label="Up Projection\nInput: [batch=1024, seq=625, hidden=8192]\nOutput: [batch=1024, seq=625, hidden=32768]\nGPU: 0"];
            silu0_0 [label="SiLU\nInput: [batch=1024, seq=625, hidden=32768]\nOutput: [batch=1024, seq=625, hidden=32768]\nGPU: 0"];
            mul0_0 [label="Elementwise Mul\nInput: [batch=1024, seq=625, hidden=32768] x2\nOutput: [batch=1024, seq=625, hidden=32768]\nGPU: 0"];
            down0_0 [label="Down Projection\nInput: [batch=1024, seq=625, hidden=32768]\nOutput: [batch=1024, seq=625, hidden=8192]\nGPU: 0"];
            residual_mlp0_0 [shape=parallelogram, label="Add\nInput: [batch=1024, seq=625, hidden=8192] x2\nOutput: [batch=1024, seq=625, hidden=8192]\nGPU: 0"];
        }
        
        // Repeat for device 1 (simplified representation)
        device1_rep [label="...\nGPU 1-15\n(Same structure as GPU 0)"];
    }
    
    // Ring topology connections
    ring_topology [shape=ellipse, label="Ring Topology\nKV exchange in 16 stages\nGPU 0→1→2→...→15→0"];
    
    // ===== Layer 1 (repeated for all layers) =====
    subgraph cluster_layer1_all {
        label="Layer 1-3 (All Devices - Same Structure)";
        style=dashed;
        
        layer1_rep [label="Layers 1-3\nSame structure repeated\n16 devices, 625 tokens each"];
    }
    
    // Sequence gather
    sequence_gather [shape=parallelogram, label="Sequence Gather\nInput: [batch=1024, seq=625, hidden=8192] x 16\nOutput: [batch=1024, seq=10000, hidden=8192]\nGPU: All 16"];
    
    // Final output
    output [shape=ellipse, label="Output\nInput: [batch=1024, seq=10000, hidden=8192]\nOutput: [batch=1024, seq=10000, vocab_size]\nGPU: All 16"];
    
    // Connections
    input -> sequence_split;
    
    // Device 0 connections
    sequence_split -> emb0;
    emb0 -> ln0_0;
    ln0_0 -> q_proj0_0;
    ln0_0 -> k_proj0_0;
    ln0_0 -> v_proj0_0;
    
    q_proj0_0 -> ring_attn0_0;
    k_proj0_0 -> kv_send0_0;
    v_proj0_0 -> kv_send0_0;
    
    kv_recv0_0 -> attn0_0;
    ring_attn0_0 -> attn0_0;
    
    attn0_0 -> out_proj0_0;
    out_proj0_0 -> residual0_0;
    emb0 -> residual0_0;
    
    residual0_0 -> gate0_0;
    residual0_0 -> up0_0;
    gate0_0 -> silu0_0;
    up0_0 -> mul0_0;
    silu0_0 -> mul0_0;
    mul0_0 -> down0_0;
    down0_0 -> residual_mlp0_0;
    residual0_0 -> residual_mlp0_0;
    
    // Connect to layer representation
    residual_mlp0_0 -> layer1_rep;
    layer1_rep -> sequence_gather;
    sequence_gather -> output;
    
    // Ring topology visualization
    kv_send0_0 -> ring_topology [style=dashed];
    ring_topology -> kv_recv0_0 [style=dashed];
}
"""
        return dot_content
    
    def generate_dags(self):
        """Generate both baseline and proposed DAGs"""
        # Create baseline DAG
        baseline_content = self.create_baseline_dag()
        baseline_path = os.path.join(self.output_dir, "baseline_dag.dot")
        with open(baseline_path, 'w') as f:
            f.write(baseline_content)
        
        # Create proposed DAG
        proposed_content = self.create_proposed_dag()
        proposed_path = os.path.join(self.output_dir, "proposed_dag.dot")
        with open(proposed_path, 'w') as f:
            f.write(proposed_content)
            
        return [baseline_path, proposed_path]

if __name__ == "__main__":
    generator = DAGGenerator("../outputs/2025-11-13-17-41-03")
    paths = generator.generate_dags()
    print(f"Generated DAGs: {paths}")