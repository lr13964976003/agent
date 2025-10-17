#!/usr/bin/env python3
"""
Generate Baseline Hybrid DAG with TP=8, PP=2 configuration
"""

def generate_baseline_hybrid():
    with open('../outputs/2025-10-17-10-32-58/baseline_hybrid.dot', 'w') as f:
        f.write(''''
digraph Baseline_Hybrid_TP_PP {
    rankdir=TB;
    node [shape=rectangle];
    
    // Global parameters
    // batch_size = 1024, seq_len = 2048, hidden_dim = 4096
    // num_heads = 32, head_dim = 128, num_experts = 16
    // tensor_parallel_degree = 8, pipeline_parallel_degree = 2
    
    // Input
    Input [shape=ellipse, style=filled, fillcolor=lightblue,
           label="Input\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-15"];
    
    // Pipeline Stage 0 (Layers 0-1)
    subgraph cluster_stage0 {
        label="Pipeline Stage 0 (GPUs 0-7)";
        style=dashed;
        
        // Layer 0 - TP across 8 GPUs
        subgraph cluster_layer0_stage0 {
            label="Layer 0 - Tensor Parallel (8 GPUs)";
            style=solid;
            
            // LayerNorm - replicated
            LN_Attn0 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
            
            // QKV projections - tensor parallel
            QKV_TP0_0 [label="QKV Proj (slice 0/8)\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 0"];
            QKV_TP1_0 [label="QKV Proj (slice 1/8)\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 1"];
            QKV_TP2_0 [label="QKV Proj (slice 2/8)\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 2"];
            QKV_TP3_0 [label="QKV Proj (slice 3/8)\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 3"];
            QKV_TP4_0 [label="QKV Proj (slice 4/8)\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 4"];
            QKV_TP5_0 [label="QKV Proj (slice 5/8)\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 5"];
            QKV_TP6_0 [label="QKV Proj (slice 6/8)\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 6"];
            QKV_TP7_0 [label="QKV Proj (slice 7/8)\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 7"];
            
            // Attention computation
            Attn_TP0_0 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 0"];
            Attn_TP1_0 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 1"];
            Attn_TP2_0 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 2"];
            Attn_TP3_0 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 3"];
            Attn_TP4_0 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 4"];
            Attn_TP5_0 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 5"];
            Attn_TP6_0 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 6"];
            Attn_TP7_0 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: 7"];
            
            // All-reduce for attention output
            AllReduce_Attn0 [shape=parallelogram, label="All-Reduce\\nInput: [local slice]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
            
            // Output projection
            Attn_Out0 [label="Output Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
            
            // Residual connection
            Residual_Attn0 [label="Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
            
            // MoE layer 0
            LN_MoE0 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
            
            // Gate and routing
            Gate0 [label="Gate\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\\nGPU: 0-7"];
            TopK0 [shape=parallelogram, style=dashed, label="Top-2 Selection\\nInput: [batch_size=1024, seq_len=2048, num_experts=16]\\nOutput: [batch_size=1024, seq_len=2048, top_k=2]\\nGPU: 0-7"];
            
            // Expert routing
            Route_MoE0 [shape=parallelogram, label="Expert Routing\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [tokens_per_expert, hidden_dim=4096]\\nGPU: 0-7"];
            
            // Expert computations - TP distributed
            Expert_TP0_0 [label="Expert 0\\nInput: [tokens_per_expert, hidden_dim=512]\\nOutput: [tokens_per_expert, hidden_dim=512]\\nGPU: 0-7"];
            Expert_TP1_0 [label="Expert 1\\nInput: [tokens_per_expert, hidden_dim=512]\\nOutput: [tokens_per_expert, hidden_dim=512]\\nGPU: 0-7"];
            Expert_TP2_0 [label="Expert 2\\nInput: [tokens_per_expert, hidden_dim=512]\\nOutput: [tokens_per_expert, hidden_dim=512]\\nGPU: 0-7"];
            Expert_TP3_0 [label="Expert 3\\nInput: [tokens_per_expert, hidden_dim=512]\\nOutput: [tokens_per_expert, hidden_dim=512]\\nGPU: 0-7"];
            
            // Aggregation and output
            Aggregate_MoE0 [shape=parallelogram, label="Expert Aggregation\\nInput: [tokens_per_expert, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
            MoE_Out0 [label="MoE Output\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
            Residual_MoE0 [label="Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
        }
        
        // Layer 1 - similar to layer 0
        LN_Attn1 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
        Attn1_Complete [label="Attention Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
        MoE1_Complete [label="MoE Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7"];
    }
    
    // Pipeline communication
    Pipeline_Comm [shape=parallelogram, label="Pipeline Communication\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7 -> 8-15"];
    
    // Pipeline Stage 1 (Layers 2-3)
    subgraph cluster_stage1 {
        label="Pipeline Stage 1 (GPUs 8-15)";
        style=dashed;
        
        // Similar to stage 0
        LN_Attn2 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15"];
        Attn2_Complete [label="Attention Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15"];
        MoE2_Complete [label="MoE Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15"];
        
        LN_Attn3 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15"];
        Attn3_Complete [label="Attention Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15"];
        MoE3_Complete [label="MoE Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15"];
    }
    
    // Output
    Output [shape=ellipse, style=filled, fillcolor=lightgreen,
            label="Output\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15"];
    
    // Connections
    Input -> LN_Attn0;
    LN_Attn0 -> {QKV_TP0_0 QKV_TP1_0 QKV_TP2_0 QKV_TP3_0 QKV_TP4_0 QKV_TP5_0 QKV_TP6_0 QKV_TP7_0};
    
    QKV_TP0_0 -> Attn_TP0_0;
    QKV_TP1_0 -> Attn_TP1_0;
    QKV_TP2_0 -> Attn_TP2_0;
    QKV_TP3_0 -> Attn_TP3_0;
    QKV_TP4_0 -> Attn_TP4_0;
    QKV_TP5_0 -> Attn_TP5_0;
    QKV_TP6_0 -> Attn_TP6_0;
    QKV_TP7_0 -> Attn_TP7_0;
    
    Attn_TP0_0 -> AllReduce_Attn0;
    Attn_TP1_0 -> AllReduce_Attn0;
    Attn_TP2_0 -> AllReduce_Attn0;
    Attn_TP3_0 -> AllReduce_Attn0;
    Attn_TP4_0 -> AllReduce_Attn0;
    Attn_TP5_0 -> AllReduce_Attn0;
    Attn_TP6_0 -> AllReduce_Attn0;
    Attn_TP7_0 -> AllReduce_Attn0;
    
    AllReduce_Attn0 -> Attn_Out0;
    Attn_Out0 -> Residual_Attn0;
    Input -> Residual_Attn0;
    
    Residual_Attn0 -> LN_Attn1;
    LN_Attn1 -> Attn1_Complete;
    Attn1_Complete -> MoE1_Complete;
    
    MoE1_Complete -> Pipeline_Comm;
    Pipeline_Comm -> LN_Attn2;
    
    LN_Attn2 -> Attn2_Complete;
    Attn2_Complete -> MoE2_Complete;
    LN_Attn2 -> Attn3_Complete;
    Attn3_Complete -> MoE3_Complete;
    
    MoE3_Complete -> Output;
}
''')

if __name__ == "__main__":
    generate_baseline_hybrid()