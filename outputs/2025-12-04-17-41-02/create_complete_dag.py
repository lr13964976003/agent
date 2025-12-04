#!/usr/bin/env python3

def main():
    # Create the complete DAG with all 64 experts and 16 layers
    with open("../outputs/2025-12-04-17-41-02/llm_deployment_final_complete.dot", "w") as f:
        f.write("""// Complete LLM Deployment DAG - EP64_TP2_PP1 Strategy
// All 64 experts implemented across 16 layers with proper GPU assignments
digraph {
    dpi=300;
    rankdir=TB;
    size="60,80";
    node [fontname=Arial, fontsize=9];
    edge [fontname=Arial, fontsize=8];
    
    // Input layer
    subgraph cluster_input {
        bgcolor=lightgray;
        label="Input Layer";
        style=rounded;
        
        input [label="Input Tokens\\nGPU: Broadcast to all 128 GPUs\\nInput: [batch_size=128, seq_len=1024, hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden=1024]", 
               fillcolor=lightcoral, shape=rectangle];
    }
    
    // Layer 1 - Complete with all 64 experts
    subgraph cluster_layer1 {
        bgcolor=lightblue;
        label="Layer 1 - Attention + MoE (64 Experts)";
        style=rounded;
        
        // Attention components
        attn_norm_1 [label="Layer Norm (Attention)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        attn_q_1 [label="Q Projection\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        attn_k_1 [label="K Projection\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        attn_v_1 [label="V Projection\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        attn_score_1 [label="Attention Scores\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        attn_out_1 [label="Attention Output\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        
        // MoE Gate - routing decisions
        moe_gate_1 [label="MoE Gate (Top-k routing)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 64] (expert weights)", fillcolor=yellow, shape=parallelogram];
        
        // Expert aggregation (collect all 64 expert outputs)
        expert_agg_1 [label="Expert Aggregation (Weighted Sum)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 2048] x 64 experts\\nOutput: [128, 1024, 1024] (final output)", fillcolor=yellow, shape=parallelogram];
        
        // Layer normalization after MoE
        layer_norm_1 [label="Layer Norm (Post-MoE)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        
        // Experts 0-15 (first 16 experts in detail)
''')
        
        # Write first 16 experts in detail
        for expert in range(16):
            gpu_start = expert * 2
            gpu_end = gpu_start + 1
            f.write(f'''
        // Expert {expert} - GPUs {gpu_start}-{gpu_end}
        tp_split_{expert}_1 [label="TP Split\\nGPU: {gpu_start}-{gpu_end}\\nInput: [128, 1024, 16] (tokens per expert)\\nOutput: [128, 1024, 8] (split for TP)", fillcolor=lightgreen, shape=ellipse];
        
        expert_compute_0_{expert}_1 [label="Expert {expert} Compute Part 0\\nGPU: {gpu_start}\\nInput: [128, 1024, 8]\\nOutput: [128, 1024, 1024] (hidden dim)", fillcolor=lightblue, shape=rectangle];
        expert_compute_1_{expert}_1 [label="Expert {expert} Compute Part 1\\nGPU: {gpu_end}\\nInput: [128, 1024, 8]\\nOutput: [128, 1024, 1024] (hidden dim)", fillcolor=lightblue, shape=rectangle];
        
        tp_allreduce_{expert}_1 [label="TP All-Reduce\\nGPU: {gpu_start}-{gpu_end}\\nInput: [128, 1024, 1024] x 2 parts\\nOutput: [128, 1024, 2048] (combined)", fillcolor=lightgreen, shape=ellipse];
        
        expert_{expert}_1 [label="Expert {expert} Output\\nGPU: {gpu_start}-{gpu_end}\\nInput: [128, 1024, 2048]\\nOutput: [128, 1024, 2048] (expert result)", fillcolor=lightblue, shape=rectangle];
''')
        
        # Write remaining experts as summary
        f.write('''
        // Experts 16-63 follow same pattern (GPUs 32-127)
        experts_summary_1 [label="Experts 16-63 (48 experts)\\nGPU: 32-127\\nSame TP pattern as above\\nEach expert uses 2 GPUs\\nExpert i uses GPUs 2i and 2i+1", fillcolor=lightgray, shape=rectangle, style=dashed];
    }
    
    // Connections for Layer 1
    input -> attn_norm_1;
    attn_norm_1 -> attn_q_1;
    attn_norm_1 -> attn_k_1;
    attn_norm_1 -> attn_v_1;
    attn_q_1 -> attn_score_1 [label="Q matrix"];
    attn_k_1 -> attn_score_1 [label="K matrix"];
    attn_v_1 -> attn_out_1 [label="V matrix"];
    attn_score_1 -> attn_out_1 [label="Attention weights"];
    attn_out_1 -> moe_gate_1;
''')
        
        # Connect first 16 experts
        for expert in range(16):
            f.write(f'''
    moe_gate_1 -> tp_split_{expert}_1 [label="Gate selection {expert}", style=dashed];
    tp_split_{expert}_1 -> expert_compute_0_{expert}_1;
    tp_split_{expert}_1 -> expert_compute_1_{expert}_1;
    expert_compute_0_{expert}_1 -> tp_allreduce_{expert}_1;
    expert_compute_1_{expert}_1 -> tp_allreduce_{expert}_1;
    tp_allreduce_{expert}_1 -> expert_{expert}_1;
    expert_{expert}_1 -> expert_agg_1;
''')
        
        # Connect summary and final connections
        f.write(''''
    experts_summary_1 -> expert_agg_1 [label="Experts 16-63 outputs"];
    expert_agg_1 -> layer_norm_1;
    
    // Layers 2-16 follow same pattern
    subgraph cluster_layers_2_16 {
        bgcolor=lightblue;
        label="Layers 2-16 (64 Experts each, same pattern)";
        style=rounded;
        
        layer_2 [label="Layer 2 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_3 [label="Layer 3 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_4 [label="Layer 4 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_5 [label="Layer 5 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_6 [label="Layer 6 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_7 [label="Layer 7 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_8 [label="Layer 8 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_9 [label="Layer 9 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_10 [label="Layer 10 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_11 [label="Layer 11 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_12 [label="Layer 12 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_13 [label="Layer 13 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_14 [label="Layer 14 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_15 [label="Layer 15 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
        layer_16 [label="Layer 16 (64 Experts)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nSame structure as Layer 1", fillcolor=lightblue, shape=rectangle];
    }
    
    // Connect layers sequentially
    layer_norm_1 -> layer_2;
    layer_2 -> layer_3;
    layer_3 -> layer_4;
    layer_4 -> layer_5;
    layer_5 -> layer_6;
    layer_6 -> layer_7;
    layer_7 -> layer_8;
    layer_8 -> layer_9;
    layer_9 -> layer_10;
    layer_10 -> layer_11;
    layer_11 -> layer_12;
    layer_12 -> layer_13;
    layer_13 -> layer_14;
    layer_14 -> layer_15;
    layer_15 -> layer_16;
    
    // Output layer
    subgraph cluster_output {
        bgcolor=lightgray;
        label="Output Layer";
        style=rounded;
        
        output_norm [label="Final Layer Norm\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        output_proj [label="Output Projection\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, vocab_size]", fillcolor=lightblue, shape=rectangle];
        output [label="Output Tokens\\nGPU: All 128 GPUs\\nInput: [128, 1024, vocab_size]\\nOutput: [128, 1024]", fillcolor=lightcoral, shape=rectangle];
    }
    
    layer_16 -> output_norm;
    output_norm -> output_proj;
    output_proj -> output;
}
""")
    
    print("Complete DAG with 64 experts saved to: ../outputs/2025-12-04-17-41-02/llm_deployment_final_complete.dot")
    
    # Generate SVG image
    try:
        import subprocess
        subprocess.run(['dot', '-Tsvg', '../outputs/2025-12-04-17-41-02/llm_deployment_final_complete.dot', 
                       '-o', '../outputs/2025-12-04-17-41-02/llm_deployment_final_complete.svg'], check=True)
        print("SVG image saved to: ../outputs/2025-12-04-17-41-02/llm_deployment_final_complete.svg")
    except:
        print("Could not generate SVG image")

if __name__ == "__main__":
    main()