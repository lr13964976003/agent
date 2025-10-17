#!/usr/bin/env python3
"""
Generate complete MA Separation Model DAG with detailed parallel deployment
"""

def generate_ma_separation_complete():
    with open('../outputs/2025-10-17-10-32-58/ma_separation_complete.dot', 'w') as f:
        f.write(''''
digraph MA_Separation_Model {
    rankdir=TB;
    node [shape=rectangle];
    
    // Global parameters displayed as comment
    // batch_size = 1024, seq_len = 2048, hidden_dim = 4096
    // num_heads = 32, head_dim = 128, num_experts = 16
    // expert_hidden_dim = 16384, top_k = 2
    
    // Input layer
    Input [shape=ellipse, style=filled, fillcolor=lightblue, 
           label="Input\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-15"];
    
    // Layer 0
    subgraph cluster_layer0 {
        label="Transformer Layer 0";
        style=dashed;
        
        // LayerNorm for attention
        LN_Attn0 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: all GPUs"];
        
        // Attention module distributed across 12 GPUs
        subgraph cluster_attention0 {
            label="Attention (12 GPUs)";
            style=solid;
            
            // Input broadcast to all attention GPUs
            Broadcast_Attn0 [shape=parallelogram, label="Broadcast\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: same\\nGPU: 0-11"];
            
            // QKV projections for each GPU
            QKV_0 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 0"];
            QKV_1 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 1"];
            QKV_2 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 2"];
            QKV_3 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 3"];
            QKV_4 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 4"];
            QKV_5 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 5"];
            QKV_6 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 6"];
            QKV_7 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 7"];
            QKV_8 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nGPU: 8"];
            QKV_9 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nGPU: 9"];
            QKV_10 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nGPU: 10"];
            QKV_11 [label="QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nGPU: 11"];
            
            // Multi-head attention computation
            Attn_0 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 0"];
            Attn_1 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 1"];
            Attn_2 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 2"];
            Attn_3 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 3"];
            Attn_4 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 4"];
            Attn_5 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 5"];
            Attn_6 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 6"];
            Attn_7 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=3, head_dim=128]\\nGPU: 7"];
            Attn_8 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nGPU: 8"];
            Attn_9 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nGPU: 9"];
            Attn_10 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nGPU: 10"];
            Attn_11 [label="Multi-Head Attention\\nInput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=2, head_dim=128]\\nGPU: 11"];
            
            // All-reduce across attention GPUs
            AllReduce_Attn0 [shape=parallelogram, label="All-Reduce\\nInput: [local heads, head_dim]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-11"];
            
            // Output projection
            Attn_Out0 [label="Output Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-11"];
        }
        
        // Residual connection
        Residual_Attn0 [label="Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: all GPUs"];
        
        // LayerNorm for MoE
        LN_MoE0 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 12-15"];
        
        // MoE module distributed across 4 GPUs
        subgraph cluster_moe0 {
            label="MoE (4 GPUs)";
            style=solid;
            
            // Gate computation
            Gate0 [label="Gate\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\\nGPU: 12-15"];
            
            // Top-2 routing
            TopK0 [shape=parallelogram, style=dashed, label="Top-2 Selection\\nInput: [batch_size=1024, seq_len=2048, num_experts=16]\\nOutput: [batch_size=1024, seq_len=2048, top_k=2]\\nGPU: 12-15"];
            
            // All-to-all communication for routing
            Route0 [shape=parallelogram, label="All-to-All Routing\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [tokens_per_expert, hidden_dim=4096]\\nGPU: 12-15"];
            
            // Expert computations
            Experts_GPU12 [label="Experts 0-3\\nInput: [tokens_per_expert, hidden_dim=4096]\\nOutput: [tokens_per_expert, hidden_dim=4096]\\nGPU: 12"];
            Experts_GPU13 [label="Experts 4-7\\nInput: [tokens_per_expert, hidden_dim=4096]\\nOutput: [tokens_per_expert, hidden_dim=4096]\\nGPU: 13"];
            Experts_GPU14 [label="Experts 8-11\\nInput: [tokens_per_expert, hidden_dim=4096]\\nOutput: [tokens_per_expert, hidden_dim=4096]\\nGPU: 14"];
            Experts_GPU15 [label="Experts 12-15\\nInput: [tokens_per_expert, hidden_dim=4096]\\nOutput: [tokens_per_expert, hidden_dim=4096]\\nGPU: 15"];
            
            // All-to-all aggregation
            Aggregate0 [shape=parallelogram, label="All-to-All Aggregation\\nInput: [tokens_per_expert, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 12-15"];
            
            // LayerNorm after MoE
            LN_After_MoE0 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 12-15"];
        }
        
        // MoE output projection
        MoE_Out0 [label="MoE Output Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 12-15"];
        
        // Final residual connection
        Residual_MoE0 [label="Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 12-15"];
    }
    
    // Layer 1
    subgraph cluster_layer1 {
        label="Transformer Layer 1";
        style=dashed;
        
        LN_Attn1 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: all GPUs"];
        
        // Similar to layer 0
        Attn1_Complete [label="Attention Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-11"];
        
        MoE1_Complete [label="MoE Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 12-15"];
    }
    
    // Layer 2
    subgraph cluster_layer2 {
        label="Transformer Layer 2";
        style=dashed;
        
        LN_Attn2 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: all GPUs"];
        
        Attn2_Complete [label="Attention Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-11"];
        
        MoE2_Complete [label="MoE Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 12-15"];
    }
    
    // Layer 3
    subgraph cluster_layer3 {
        label="Transformer Layer 3";
        style=dashed;
        
        LN_Attn3 [label="LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: all GPUs"];
        
        Attn3_Complete [label="Attention Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-11"];
        
        MoE3_Complete [label="MoE Complete\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 12-15"];
    }
    
    // Final output
    Output [shape=ellipse, style=filled, fillcolor=lightgreen,
            label="Output\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: all GPUs"];
    
    // Connections for Layer 0
    Input -> LN_Attn0;
    LN_Attn0 -> Broadcast_Attn0;
    Broadcast_Attn0 -> {QKV_0 QKV_1 QKV_2 QKV_3 QKV_4 QKV_5 QKV_6 QKV_7 QKV_8 QKV_9 QKV_10 QKV_11};
    
    QKV_0 -> Attn_0;
    QKV_1 -> Attn_1;
    QKV_2 -> Attn_2;
    QKV_3 -> Attn_3;
    QKV_4 -> Attn_4;
    QKV_5 -> Attn_5;
    QKV_6 -> Attn_6;
    QKV_7 -> Attn_7;
    QKV_8 -> Attn_8;
    QKV_9 -> Attn_9;
    QKV_10 -> Attn_10;
    QKV_11 -> Attn_11;
    
    Attn_0 -> AllReduce_Attn0;
    Attn_1 -> AllReduce_Attn0;
    Attn_2 -> AllReduce_Attn0;
    Attn_3 -> AllReduce_Attn0;
    Attn_4 -> AllReduce_Attn0;
    Attn_5 -> AllReduce_Attn0;
    Attn_6 -> AllReduce_Attn0;
    Attn_7 -> AllReduce_Attn0;
    Attn_8 -> AllReduce_Attn0;
    Attn_9 -> AllReduce_Attn0;
    Attn_10 -> AllReduce_Attn0;
    Attn_11 -> AllReduce_Attn0;
    
    AllReduce_Attn0 -> Attn_Out0;
    Attn_Out0 -> Residual_Attn0;
    Input -> Residual_Attn0;
    
    Residual_Attn0 -> LN_MoE0;
    LN_MoE0 -> Gate0;
    Gate0 -> TopK0 [style=dashed];
    TopK0 -> Route0;
    Route0 -> {Experts_GPU12 Experts_GPU13 Experts_GPU14 Experts_GPU15};
    
    Experts_GPU12 -> Aggregate0;
    Experts_GPU13 -> Aggregate0;
    Experts_GPU14 -> Aggregate0;
    Experts_GPU15 -> Aggregate0;
    
    Aggregate0 -> LN_After_MoE0;
    LN_After_MoE0 -> MoE_Out0;
    MoE_Out0 -> Residual_MoE0;
    Residual_Attn0 -> Residual_MoE0;
    
    // Connections between layers
    Residual_MoE0 -> LN_Attn1;
    LN_Attn1 -> Attn1_Complete;
    Attn1_Complete -> MoE1_Complete;
    MoE1_Complete -> LN_Attn2;
    
    LN_Attn2 -> Attn2_Complete;
    Attn2_Complete -> MoE2_Complete;
    MoE2_Complete -> LN_Attn3;
    
    LN_Attn3 -> Attn3_Complete;
    Attn3_Complete -> MoE3_Complete;
    MoE3_Complete -> Output;
}
''')

if __name__ == "__main__":
    generate_ma_separation_complete()