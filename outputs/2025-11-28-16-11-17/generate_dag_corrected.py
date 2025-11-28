#!/usr/bin/env python3

import os

def generate_corrected_dag():
    """Generate a corrected DAG that fixes the structural issues identified in the feedback"""
    
    dot_content = '''// Large-Scale Cross-Node Expert Parallelism DAG - Corrected Version
digraph {
    rankdir=LR
    node [fontname=Arial fontsize=10]
    
    // Input Layer
    subgraph cluster_input {
        label="Input Distribution Layer"
        style=rounded
        
        input_tokens [label="Input Tokens\\n[batch_size=?, seq_len=?]\\nGPU: N/A" fillcolor=lightgray shape=egg style=filled]
        token_split [label="Token Split\\n[batch_size=?, seq_len=?] → [batch_size=?, seq_len=?/256]\\nGPU: 0-255" fillcolor=lightgreen shape=parallelogram style=filled]
        
        input_tokens -> token_split [label=broadcast]
    }
    
    // Layer 1: GPU 0 Processing Pipeline
    subgraph cluster_gpu0 {
        label="GPU 0: Complete Processing Pipeline"
        style=rounded
        
        mla_0 [label="MLA\\n[batch_size=?, seq_len=?, heads=128, d_k=56]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 0" fillcolor=lightblue shape=rectangle style=filled]
        gate_0 [label="Expert Gating\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, top_k=2]\\nGPU: 0" fillcolor=lightgreen shape=parallelogram style=filled]
        residual_add_0 [label="Residual Add\\n[batch_size=?, seq_len=?, dim=7168] + [batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 0" fillcolor=lightgreen shape=parallelogram style=filled]
        expert_0 [label="Expert MLP 0\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 0" fillcolor=lightblue shape=rectangle style=filled]
        
        token_split -> mla_0 [label="tokens for GPU 0"]
        mla_0 -> gate_0
        gate_0 -> residual_add_0
        token_split -> residual_add_0 [label=residual style=dashed]
        residual_add_0 -> expert_0
    }
    
    // Layer 2: GPU 64 Processing Pipeline
    subgraph cluster_gpu64 {
        label="GPU 64: Complete Processing Pipeline"
        style=rounded
        
        mla_64 [label="MLA\\n[batch_size=?, seq_len=?, heads=128, d_k=56]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 64" fillcolor=lightblue shape=rectangle style=filled]
        gate_64 [label="Expert Gating\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, top_k=2]\\nGPU: 64" fillcolor=lightgreen shape=parallelogram style=filled]
        residual_add_64 [label="Residual Add\\n[batch_size=?, seq_len=?, dim=7168] + [batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 64" fillcolor=lightgreen shape=parallelogram style=filled]
        expert_64 [label="Expert MLP 64\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 64" fillcolor=lightblue shape=rectangle style=filled]
        
        token_split -> mla_64 [label="tokens for GPU 64"]
        mla_64 -> gate_64
        gate_64 -> residual_add_64
        token_split -> residual_add_64 [label=residual style=dashed]
        residual_add_64 -> expert_64
    }
    
    // Layer 3: GPU 128 Processing Pipeline
    subgraph cluster_gpu128 {
        label="GPU 128: Complete Processing Pipeline"
        style=rounded
        
        mla_128 [label="MLA\\n[batch_size=?, seq_len=?, heads=128, d_k=56]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 128" fillcolor=lightblue shape=rectangle style=filled]
        gate_128 [label="Expert Gating\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, top_k=2]\\nGPU: 128" fillcolor=lightgreen shape=parallelogram style=filled]
        residual_add_128 [label="Residual Add\\n[batch_size=?, seq_len=?, dim=7168] + [batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 128" fillcolor=lightgreen shape=parallelogram style=filled]
        expert_128 [label="Expert MLP 128\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 128" fillcolor=lightblue shape=rectangle style=filled]
        
        token_split -> mla_128 [label="tokens for GPU 128"]
        mla_128 -> gate_128
        gate_128 -> residual_add_128
        token_split -> residual_add_128 [label=residual style=dashed]
        residual_add_128 -> expert_128
    }
    
    // Layer 4: GPU 192 Processing Pipeline
    subgraph cluster_gpu192 {
        label="GPU 192: Complete Processing Pipeline"
        style=rounded
        
        mla_192 [label="MLA\\n[batch_size=?, seq_len=?, heads=128, d_k=56]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 192" fillcolor=lightblue shape=rectangle style=filled]
        gate_192 [label="Expert Gating\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, top_k=2]\\nGPU: 192" fillcolor=lightgreen shape=parallelogram style=filled]
        residual_add_192 [label="Residual Add\\n[batch_size=?, seq_len=?, dim=7168] + [batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 192" fillcolor=lightgreen shape=parallelogram style=filled]
        expert_192 [label="Expert MLP 192\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 192" fillcolor=lightblue shape=rectangle style=filled]
        
        token_split -> mla_192 [label="tokens for GPU 192"]
        mla_192 -> gate_192
        gate_192 -> residual_add_192
        token_split -> residual_add_192 [label=residual style=dashed]
        residual_add_192 -> expert_192
    }
    
    // Layer 5: GPU 255 Processing Pipeline
    subgraph cluster_gpu255 {
        label="GPU 255: Complete Processing Pipeline"
        style=rounded
        
        mla_255 [label="MLA\\n[batch_size=?, seq_len=?, heads=128, d_k=56]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 255" fillcolor=lightblue shape=rectangle style=filled]
        gate_255 [label="Expert Gating\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, top_k=2]\\nGPU: 255" fillcolor=lightgreen shape=parallelogram style=filled]
        residual_add_255 [label="Residual Add\\n[batch_size=?, seq_len=?, dim=7168] + [batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 255" fillcolor=lightgreen shape=parallelogram style=filled]
        expert_255 [label="Expert MLP 255\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 255" fillcolor=lightblue shape=rectangle style=filled]
        
        token_split -> mla_255 [label="tokens for GPU 255"]
        mla_255 -> gate_255
        gate_255 -> residual_add_255
        token_split -> residual_add_255 [label=residual style=dashed]
        residual_add_255 -> expert_255
    }
    
    // Cross-Node Communication and Routing
    subgraph cluster_communication {
        label="Cross-Node Communication & Routing"
        style=rounded
        
        // Routing communications from gates to experts
        comm_0_to_64 [label="Token Routing\\n[batch_size=?, seq_len=?, dim=7168]\\n→ GPU: 64" fillcolor=lightyellow shape=ellipse style=filled]
        comm_0_to_128 [label="Token Routing\\n[batch_size=?, seq_len=?, dim=7168]\\n→ GPU: 128" fillcolor=lightyellow shape=ellipse style=filled]
        comm_0_to_192 [label="Token Routing\\n[batch_size=?, seq_len=?, dim=7168]\\n→ GPU: 192" fillcolor=lightyellow shape=ellipse style=filled]
        comm_0_to_255 [label="Token Routing\\n[batch_size=?, seq_len=?, dim=7168]\\n→ GPU: 255" fillcolor=lightyellow shape=ellipse style=filled]
        
        comm_64_to_0 [label="Token Routing\\n[batch_size=?, seq_len=?, dim=7168]\\n→ GPU: 0" fillcolor=lightyellow shape=ellipse style=filled]
        comm_64_to_128 [label="Token Routing\\n[batch_size=?, seq_len=?, dim=7168]\\n→ GPU: 128" fillcolor=lightyellow shape=ellipse style=filled]
        comm_64_to_192 [label="Token Routing\\n[batch_size=?, seq_len=?, dim=7168]\\n→ GPU: 192" fillcolor=lightyellow shape=ellipse style=filled]
        comm_64_to_255 [label="Token Routing\\n[batch_size=?, seq_len=?, dim=7168]\\n→ GPU: 255" fillcolor=lightyellow shape=ellipse style=filled]
        
        // Async transfers between nodes
        async_0_64 [label="Async Transfer\\nGPU: 0 ↔ GPU: 64" fillcolor=lightyellow shape=ellipse style=filled]
        async_64_128 [label="Async Transfer\\nGPU: 64 ↔ GPU: 128" fillcolor=lightyellow shape=ellipse style=filled]
        async_128_192 [label="Async Transfer\\nGPU: 128 ↔ GPU: 192" fillcolor=lightyellow shape=ellipse style=filled]
        async_192_255 [label="Async Transfer\\nGPU: 192 ↔ GPU: 255" fillcolor=lightyellow shape=ellipse style=filled]
        
        // Gate to routing connections (dashed lines)
        gate_0 -> comm_0_to_64 [label="route tokens" style=dashed]
        gate_0 -> comm_0_to_128 [label="route tokens" style=dashed]
        gate_0 -> comm_0_to_192 [label="route tokens" style=dashed]
        gate_0 -> comm_0_to_255 [label="route tokens" style=dashed]
        
        gate_64 -> comm_64_to_0 [label="route tokens" style=dashed]
        gate_64 -> comm_64_to_128 [label="route tokens" style=dashed]
        gate_64 -> comm_64_to_192 [label="route tokens" style=dashed]
        gate_64 -> comm_64_to_255 [label="route tokens" style=dashed]
        
        // Routing to expert connections
        comm_0_to_64 -> expert_64
        comm_0_to_128 -> expert_128
        comm_0_to_192 -> expert_192
        comm_0_to_255 -> expert_255
        
        comm_64_to_0 -> expert_0
        comm_64_to_128 -> expert_128
        comm_64_to_192 -> expert_192
        comm_64_to_255 -> expert_255
        
        // Async transfer connections
        gate_0 -> async_0_64 [label="coord" style=dashed]
        gate_64 -> async_0_64 [label="coord" style=dashed]
        gate_64 -> async_64_128 [label="coord" style=dashed]
        gate_128 -> async_64_128 [label="coord" style=dashed]
        gate_128 -> async_128_192 [label="coord" style=dashed]
        gate_192 -> async_128_192 [label="coord" style=dashed]
        gate_192 -> async_192_255 [label="coord" style=dashed]
        gate_255 -> async_192_255 [label="coord" style=dashed]
    }
    
    // Output Aggregation
    subgraph cluster_output {
        label="Output Aggregation"
        style=rounded
        
        collect_res [label="Gather Results\\n[batch_size=?, seq_len=?, dim=7168] from all GPUs\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 0-255" fillcolor=lightgreen shape=parallelogram style=filled]
        final_out [label="Final Output\\n[batch_size=?, seq_len=?, dim=7168]\\nGPU: 0" fillcolor=lightgray shape=egg style=filled]
        
        expert_0 -> collect_res [label="from GPU 0"]
        expert_64 -> collect_res [label="from GPU 64"]
        expert_128 -> collect_res [label="from GPU 128"]
        expert_192 -> collect_res [label="from GPU 192"]
        expert_255 -> collect_res [label="from GPU 255"]
        
        collect_res -> final_out
    }
}'''
    
    return dot_content

def main():
    # Generate the corrected DAG
    dot_content = generate_corrected_dag()
    
    # Save the DOT file
    dot_file_path = "../outputs/2025-11-28-16-11-17/large_ep_dag_corrected.dot"
    with open(dot_file_path, 'w') as f:
        f.write(dot_content)
    
    print(f"Generated corrected DAG file: {dot_file_path}")
    
    # Generate SVG image using graphviz
    try:
        svg_file_path = "../outputs/2025-11-28-16-11-17/large_ep_dag_corrected.svg"
        os.system(f"dot -Tsvg {dot_file_path} -o {svg_file_path}")
        print(f"Generated SVG image: {svg_file_path}")
    except Exception as e:
        print(f"Warning: Could not generate SVG image: {e}")
    
    return {
        "dot_file": dot_file_path,
        "svg_file": svg_file_path if 'svg_file_path' in locals() else None
    }

if __name__ == "__main__":
    result = main()
    print(f"Files generated: {result}")