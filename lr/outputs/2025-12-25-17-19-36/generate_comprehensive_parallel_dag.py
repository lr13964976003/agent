#!/usr/bin/env python3
"""
Comprehensive Parallel Strategy DAG Generator
Generates a complete DAG for 24-GPU LLM deployment with corrected parallel strategy
"""

import os

def generate_comprehensive_parallel_dag():
    """Generate comprehensive DAG with all 24 GPUs and complete parallel strategy"""
    
    dot_content = '''digraph ComprehensiveParallelDAG {
    graph [bb="0,0,8000,6000",
           dpi=300,
           rankdir=TB,
           size="30,40",
           bgcolor=white,
           fontname="Arial"];
    node [fontname="Arial",
          fontsize=10,
          style=filled];
    edge [fontname="Arial",          fontsize=8];

    // Node shape definitions
    node [fillcolor=lightyellow,
          shape=parallelogram]; // Default for routing/aggregation
    
    // Global settings
    ranksep=1.5;
    nodesep=0.8;

    // ==================== INPUT LAYER ====================
    subgraph cluster_input {
        graph [fillcolor=lightgray,
               label="Input Layer (Batch Size 128, Seq Len 10240)",
               style=rounded];
        
        input_node [fillcolor=lightgreen,
                   shape=rectangle,
                   label="Input Tokens\nGPU: ALL\nInput: [batch_size=128, seq_len=10240]\nOutput: [batch_size=128, seq_len=10240, vocab_size]"];
    }

    // ==================== DATA PARALLEL SPLIT ====================
    subgraph cluster_dp_split {
        graph [fillcolor=lightcyan,
               label="Data Parallel Split (3 groups)",
               style=rounded];
一纸书
        dp_split [fillcolor=lightblue,
                 shape=ellipse,
                 label="DP Data Split\nGPUs: 0-23\nInput: [batch_size=128, seq_len=10240]\nOutput: [batch_size=43, seq_len=10240] per DP group"];
    }

    // ==================== GPU CLUSTERS (24 GPUs) ====================
    // Each GPU will have complete layer definitions
    
    // === GPU 0-7: Pipeline Stage 0, Data Parallel Group 0 ===
    '''
    
    # Generate GPU clusters for all 24 GPUs
    for gpu_id in range(24):
        stage = gpu_id // 8  # Pipeline stage (0 or 1)
        dp_group = gpu_id // 8  # Data parallel group (0, 1, or 2)
        tp_partner = gpu_id + 1 if gpu_id % 2 == 0 else gpu_id - 1  # Tensor parallel partner
        
        dot_content += f'''
    // GPU {gpu_id}: Pipeline Stage {stage}, DP Group {dp_group}
    subgraph cluster_gpu_{gpu_id} {{
        graph [fillcolor=white,
               label="GPU {gpu_id} (PP Stage {stage}, DP Group {dp_group}, TP Partner {tp_partner})",
               style=rounded];
        
        // Input handling for this GPU
        gpu_{gpu_id}_input [fillcolor=lightgreen,
                           shape=rectangle,
                           label="Input Handler\nGPU: {gpu_id}\nInput: [batch_size=43, seq_len=10240]\nOutput: [batch_size=43, seq_len=10240, d_model=512]"];
        
        // Embedding layer
        gpu_{gpu_id}_embed [fillcolor=lightgreen,
                           shape=rectangle,
                           label="Embedding Layer\nGPU: {gpu_id}\nInput: [batch_size=43, seq_len=10240]\nOutput: [batch_size=43, seq_len=10240, d_model=512]"];
        '''
        
        # Generate all 8 layers for this GPU (since we have 16 total layers split across 2 pipeline stages)
        for layer in range(8):
            actual_layer = layer + (stage * 8)  # Actual layer number (0-15)
收兵
            # Attention decomposition - 5 sub-modules per attention layer
            dot_content += f'''
        // Layer {actual_layer} Attention - GPU {gpu_id}
        gpu_{gpu_id}_l{actual_layer}_q_proj [fillcolor=lightgreen,
                                            shape=rectangle,
                                            label="Layer {actual_layer} Q Projection\nGPU: {gpu_id}\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, heads=8, d_k=32]"];
        
        gpu_{gpu_id}_l{actual_layer}_k_proj [fillcolor=lightgreen,
                                            shape=rectangle,
                                            label="Layer {actual_layer} K Projection\nGPU: {gpu_id}\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, heads=8, d_k=32]"];
        
        gpu_{gpu_id}_l{actual_layer}_v_proj [fillcolor=lightgreen,
                                            shape=rectangle,
                                            label="Layer {actual_layer} V Projection\nGPU: {gpu_id}\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, heads=8, d_k=32]"];
        
        gpu_{gpu_id}_l{actual_layer}_attn_scores [fillcolor=lightgreen, legally                                                 shape=rectangle,
                                                 label="Layer {actual_layer} Attention Scores\nGPU: {gpu_id}\nInput: Q,K,V [batch_size=43, seq_len=10240, heads=8, d_k=32]\nOutput: [batch_size=43, seq_len=10240, heads=8, d_k=32]"];
        
        gpu_{gpu_id}_l{actual_layer}_attn_out [fillcolor=lightgreen,
                                              shape=rectangle,
                                              label="Layer {actual_layer} Attention Output\nGPU: {gpu_id}\nInput: [batch_size=43, seq_len=10240, heads=8, d_k=32]\nOutput: [batch_size=43, seq_len=10240, d_model=512]"];
        '''
            
            # MoE layer with gate router
            dot_content += f'''
        // Layer {actual_layer} MoE - GPU {gpu_id}
        gpu_{gpu_id}_l{actual_layer}_gate [shape=parallelogram,
                                          style=dashed,
                                          label="Layer {actual_layer} Gate Router\nGPU: {gpu_id}\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: routing decisions for 8 experts"];
        
        gpu_{gpu_id}_l{actual_layer}_moe [fillcolor=lightgreen,
                                         shape=rectangle,
                                         label="Layer {actual_layer} MoE (8 Experts)\nGPU: {gpu_id}\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, d_model=512]"];
        '''
    
    dot_content += '''
    }
    '''
    
    # Complete the DOT content with all connections and communications
    dot_content += generate_connections_and_communications()
    
    return dot_content

def generate_connections_and_communications():
    """Generate all connections and communication patterns"""
    
    connections = '''
    
    // ==================== COMMUNICATION PATTERNS ====================
    
    // Data Parallel Communication Groups
    subgraph cluster_dp_comm {
        graph [fillcolor=lightcyan,
               label="Data Parallel Communication",
               style=rounded];
        
        dp_allreduce_grads [fillcolor=lightblue,
                           shape=ellipse,
                           label="DP All-Reduce Gradients\nGPUs: 0-23\nInput: gradient chunks from all GPUs\nOutput: synchronized gradients across DP groups"];
    }\n\n'''
    
    # Tensor Parallel Communications
    connections += '''
    // Tensor Parallel Communication Groups
    subgraph cluster_tp_comm {
        graph [fillcolor=lightyellow,
               label="Tensor Parallel All-Reduce Groups",
               style=rounded];
        '''
    
    # Generate TP communications for each TP pair
    for tp_group in range(12):  # 24 GPUs / 2 = 12 TP groups
        gpu0 = tp_group * 2
        gpu1 = gpu0 + 1
        connections += f'''
        tp_allreduce_{tp_group} [fillcolor=lightblue,
                                shape=ellipse,
                                label="TP All-Reduce Group {tp_group}\nGPUs: {gpu0},{gpu1}\nInput: partial attention results\nOutput: complete attention results"];
        '''
    
    connections += '''
    }\n\n'''
    
    # Pipeline Parallel Communications
    connections += '''
    // Pipeline Parallel Communications
    subgraph cluster_pp_comm {
        graph [fillcolor=lightpink,
               label="Pipeline Parallel Communications",
               style=rounded];
        
        pp_sendrecv_stage0_to_1 [fillcolor=lightblue,
                                shape=ellipse,
                                label="PP Send/Recv Stage0→Stage1\nGPUs: 0-7 → 8-15\nInput: activations from stage 0\nOutput: forwarded activations to stage 1"];
        
        pp_sendrecv_stage1_to_2 [fillcolor=lightblue,
                                shape=ellipse,
                                label="PP Send/Recv Stage1→Stage2\nGPUs: 8-15 → 16-23\nInput: activations from stage 1\nOutput: forwarded activations to stage 2"];
    }\n\n'''
    
    # Expert Parallel Communications
    connections += '''
    // Expert Parallel Communications
    subgraph cluster_ep_comm {
        graph [fillcolor=lightgoldenrodyellow,
               label="Expert Parallel All-to-All",
               style=rounded];
        
        ep_all2all_stage0 [fillcolor=lightblue,
                          shape=ellipse,
                          label="EP All-to-All Stage 0\nGPUs: 0-7\nInput: token representations\nOutput: routed tokens to expert GPUs"];
        
        ep_all2all_stage1 [fillcolor=lightblue,
                          shape=ellipse,
                          label="EP All-to-All Stage 1\nGPUs: 8-15\nInput: token representations\nOutput: routed tokens to expert GPUs"];
        
        ep_all2all_stage2 [fillcolor=lightblue,
                          shape=ellipse,
                          label="EP All-to-All Stage 2\nGPUs: 16-23\nInput: token representations\nOutput: routed tokens to expert GPUs"];
    }\n\n'''
    
    # Load Balancing and Global Components
    connections += '''
    // Global Components
    subgraph cluster_global {
        graph [fillcolor=lightgray,
               label="Global Aggregation & Control",
               style=rounded];
        
        load_balancer [shape=parallelogram,
                      label="Global Load Balancer\nInput: GPU load metrics\nOutput: balancing decisions for experts and data"];
        
        final_output [fillcolor=lightgreen,
                     shape=rectangle,
                     label="Final Output Layer\nInput: [batch_size=128, seq_len=10240, d_model=512]\nOutput: [batch_size=128, seq_len=10240, vocab_size]"];
    }\n\n'''
    
    # Generate all the connections
    connections += generate_all_connections()
    
    return connections

def generate_all_connections():
    """Generate all node connections following the corrected parallel strategy"""
    
    all_connections = '''
    // ==================== CONNECTIONS ====================
    
    // Input to DP split
    input_node -> dp_split [label="full batch"];
    
    // DP split to individual GPUs
    '''
    
    # Connect DP split to all GPU inputs
    for gpu_id in range(24):
        all_connections += f'''
    dp_split -> gpu_{gpu_id}_input [label="DP split 43 seqs"];
    '''
    
    # Connect inputs to embeddings
    for gpu_id in range(24):
        all_connections += f'''
    gpu_{gpu_id}_input -> gpu_{gpu_id}_embed [label="processed input"];
    '''
    
    # Layer connections for each GPU
    for gpu_id in range(24):
        stage = gpu_id // 8
        
        for layer in range(8):
            actual_layer = layer + (stage * 8)
            
            # Attention connections
            if layer == 0 and stage == 0:
                # First layer gets input from embedding
                all_connections += f'''
    gpu_{gpu_id}_embed -> gpu_{gpu_id}_l{actual_layer}_q_proj [label="embedded tokens"];
    gpu_{gpu_id}_embed -> gpu_{gpu_id}_l{actual_layer}_k_proj [label="embedded tokens"];
    gpu_{gpu_id}_embed -> gpu_{gpu_id}_l{actual_layer}_v_proj [label="embedded tokens"];
                '''
            else:
                # Subsequent layers get input from previous layer
                prev_layer = actual_layer - 1
                all_connections += f'''
    gpu_{gpu_id}_l{prev_layer}_moe -> gpu_{gpu_id}_l{actual_layer}_q_proj [label="layer {prev_layer} output"];
    gpu_{gpu_id}_l{prev_layer}_moe -> gpu_{gpu_id}_l{actual_layer}_k_proj [label="layer {prev_layer} output"];
    gpu_{gpu_id}_l{prev_layer}_moe -> gpu_{gpu_id}_l{actual_layer}_v_proj [label="layer {prev_layer} output"];                '''
            
            # Attention internal connections
            all_connections += f'''
    gpu_{gpu_id}_l{actual_layer}_q_proj -> gpu_{gpu_id}_l{actual_layer}_attn_scores [label="Q projections"];
    gpu_{gpu_id}_l{actual_layer}_k_proj -> gpu_{gpu_id}_l{actual_layer}_attn_scores [label="K projections"];
    gpu_{gpu_id}_l{actual_layer}_v_proj -> gpu_{gpu_id}_l{actual_layer}_attn_scores [label="V projections"];
    gpu_{gpu_id}_l{actual_layer}_attn_scores -> gpu_{gpu_id}_l{actual_layer}_attn_out [label="attention weights"];
            '''
            
            # connections to gate router (dashed)
            all_connections += f'''
    gpu_{gpu_id}_l{actual_layer}_attn_out -> gpu_{gpu_id}_l{actual_layer}_gate [label="routing decision", style=dashed];
            '''
            
            # MoE connections
            all_connections += f'''
    gpu_{gpu_id}_l{actual_layer}_attn_out -> gpu_{gpu_id}_l{actual_layer}_moe [label="attention output"];
    gpu_{gpu_id}_l{actual_layer}_gate -> gpu_{gpu_id}_l{actual_layer}_moe [label="expert selection", style=dashed];
            '''
    
    # Tensor Parallel connections
    for tp_group in range(12):
        gpu0 = tp_group * 2
        gpu1 = gpu0 + 1
        
        # Connect attention outputs to TP all-reduce
        for stage in range(2):  # 2 pipeline stages
            for layer in range(8):
                actual_layer = layer + (stage * 8)
                
                all_connections += f'''
    gpu_{gpu0}_l{actual_layer}_attn_out -> tp_allreduce_{tp_group} [label="partial attention {actual_layer}"];
    gpu_{gpu1}_l{actual_layer}_attn_out -> tp_allreduce_{tp_group} [label="partial attention {actual_layer}"];
    tp_allreduce_{tp_group} -> gpu_{gpu0}_l{actual_layer}_moe [label="complete attention {actual_layer}"];
    tp_allreduce_{tp_group} -> gpu_{gpu1}_l{actual_layer}_moe [label="complete attention {actual_layer}"];
                '''
    
    # Pipeline Parallel connections
    # Stage 0 to Stage 1
    for gpu_src in range(8):  # GPUs 0-7 (stage 0)
        gpu_dst = gpu_src + 8  # Corresponding GPU in stage 1
        all_connections += f'''
    gpu_{gpu_src}_l7_moe -> pp_sendrecv_stage0_to_1 [label="stage 0 activations"];
        '''
    
    for gpu_dst in range(8, 16):  # GPUs 8-15 (stage 1)
        all_connections += f'''
    pp_sendrecv_stage0_to_1 -> gpu_{gpu_dst}_input [label="forwarded activations"];
        '''
    
    # Stage 1 to Stage 2 (DP groups)
    for gpu_src in range(8, 16):  # GPUs 8-15 (stage 1)
        all_connections += f'''
    gpu_{gpu_src}_l15_moe -> pp_sendrecv_stage1_to_2 [label="stage 1 activations"];
        '''
    
    for gpu_dst in range(16, 24):  # GPUs 16-23 (stage 2 - DP group 1)
        all_connections += f'''
    pp_sendrecv_stage1_to_2 -> gpu_{gpu_dst}_input [label="forwarded activations"];
        '''
    
    # Expert Parallel connections (one-way to avoid cycles)
    for stage in range(3):  # 3 expert parallel stages
        stage_gpus = list(range(stage * 8, (stage + 1) * 8))
        
        # Connect MoE outputs to EP all-to-all
        for gpu_id in stage_gpus:
            all_connections += f'''
    gpu_{gpu_id}_l7_moe -> ep_all2all_stage{stage} [label="tokens for routing"];
    gpu_{gpu_id}_l15_moe -> ep_all2all_stage{stage} [label="tokens for routing"];
            '''
    
    # Data Parallel gradient connections
    for gpu_id in range(24):
        all_connections += f'''
    gpu_{gpu_id}_l15_moe -> dp_allreduce_grads [label="gradients from GPU {gpu_id}"];
        '''
    
    # Load balancer connections
    all_connections += '''
    load_balancer -> gpu_0_l0_gate [label="expert balancing", style=dashed];
    load_balancer -> gpu_1_l0_gate [label="expert balancing", style=dashed];
    '''
    
    # Add more load balancer connections for all layers and GPUs
    for gpu_id in range(24):
        for layer in range(16):
            all_connections += f'''
    load_balancer -> gpu_{gpu_id}_l{layer}_gate [label="balancing", style=dashed];
            '''
    
    # Final output connections
    # Connect final layer outputs to final output
    for gpu_id in range(24):
        all_connections += f'''
    gpu_{gpu_id}_l15_moe -> final_output [label="final features from GPU {gpu_id}"];
        '''
    
    # Connect DP all-reduce to load balancer
    all_connections += '''
    dp_allreduce_grads -> load_balancer [label="gradient sync complete"];
    '''
    
    return all_connections

if __name__ == "__main__":
    # Generate the comprehensive DAG
    dag_content = generate_comprehensive_parallel_dag()
    
    # Save to file
    output_dir = "outputs/2025-12-25-17-19-36"
    dot_file = os.path.join(output_dir, "comprehensive_parallel_dag_final.dot")
    
    with open(dot_file, 'w') as f:
        f.write(dag_content)
    
    print(f"Generated comprehensive DAG: {dot_file}")
    print(f"DAG size: {len(dag_content)} characters")
    print(f"Estimated nodes: {dag_content.count('[')} components")