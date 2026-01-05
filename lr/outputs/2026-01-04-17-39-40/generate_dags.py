#!/usr/bin/env python3

import graphviz

def create_prefill_dag():
    """Create prefill phase DAG with PP=4, TP=8, EP=32"""
    dot = graphviz.Digraph(comment='Prefill Phase DAG')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Define colors for different GPU stages
    stage_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    # Input node
    dot.node('input', 'INPUT\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
             shape='ellipse', fillcolor='white')
    
    # Stage 0: GPUs 0-7 (PP stage 0)
    with dot.subgraph(name='stage0') as c:
        c.attr(rank='same')
        for gpu_id in range(8):
            # Embedding layer
            c.node(f's0_emb_{gpu_id}', f'GPU{gpu_id}: Embedding\\nInput: [batch_size=1, seq_len=4096]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[0])
            
            # Layer norm
            c.node(f's0_ln0_{gpu_id}', f'GPU{gpu_id}: LayerNorm\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[0])
            
            # Attention operations (TP=8)
            c.node(f's0_q_{gpu_id}', f'GPU{gpu_id}: Q Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[0])
            c.node(f's0_k_{gpu_id}', f'GPU{gpu_id}: K Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[0])
            c.node(f's0_v_{gpu_id}', f'GPU{gpu_id}: V Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[0])
            
            # Communication nodes
            c.node(f's0_allgather_q_{gpu_id}', f'GPU{gpu_id}: AllGather Q\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's0_allgather_k_{gpu_id}', f'GPU{gpu_id}: AllGather K\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's0_allgather_v_{gpu_id}', f'GPU{gpu_id}: AllGather V\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            
            # Attention computation
            c.node(f's0_attn_{gpu_id}', f'GPU{gpu_id}: Attention\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[0])
            
            # MoE operations (EP=32)
            c.node(f's0_gate_{gpu_id}', f'GPU{gpu_id}: MoE Gate\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, experts=8]', 
                   shape='parallelogram', fillcolor='orange')
            
            # Expert computation (1 expert per GPU)
            c.node(f's0_expert_{gpu_id}', f'GPU{gpu_id}: Expert FFN\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[0])
            
            # Communication for MoE
            c.node(f's0_alltoall_{gpu_id}', f'GPU{gpu_id}: AllToAll\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            
            # Final layer norm
            c.node(f's0_ln1_{gpu_id}', f'GPU{gpu_id}: LayerNorm\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[0])
    
    # Stage 1: GPUs 8-15 (PP stage 1)
    with dot.subgraph(name='stage1') as c:
        c.attr(rank='same')
        for gpu_id in range(8, 16):
            actual_gpu = gpu_id - 8
            # Similar operations for stage 1
            c.node(f's1_ln0_{gpu_id}', f'GPU{gpu_id}: LayerNorm\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[1])
            
            # Attention operations
            c.node(f's1_q_{gpu_id}', f'GPU{gpu_id}: Q Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[1])
            c.node(f's1_k_{gpu_id}', f'GPU{gpu_id}: K Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[1])
            c.node(f's1_v_{gpu_id}', f'GPU{gpu_id}: V Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[1])
            
            # Communication
            c.node(f's1_allgather_q_{gpu_id}', f'GPU{gpu_id}: AllGather Q\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's1_allgather_k_{gpu_id}', f'GPU{gpu_id}: AllGather K\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's1_allgather_v_{gpu_id}', f'GPU{gpu_id}: AllGather V\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            
            # Attention
            c.node(f's1_attn_{gpu_id}', f'GPU{gpu_id}: Attention\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[1])
            
            # MoE
            c.node(f's1_gate_{gpu_id}', f'GPU{gpu_id}: MoE Gate\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, experts=8]', 
                   shape='parallelogram', fillcolor='orange')
            c.node(f's1_expert_{gpu_id}', f'GPU{gpu_id}: Expert FFN\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[1])
            c.node(f's1_alltoall_{gpu_id}', f'GPU{gpu_id}: AllToAll\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's1_ln1_{gpu_id}', f'GPU{gpu_id}: LayerNorm\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[1])
    
    # Stage 2: GPUs 16-23 (PP stage 2)
    with dot.subgraph(name='stage2') as c:
        c.attr(rank='same')
        for gpu_id in range(16, 24):
            # Similar operations for stage 2
            c.node(f's2_ln0_{gpu_id}', f'GPU{gpu_id}: LayerNorm\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[2])
            
            # Attention operations
            c.node(f's2_q_{gpu_id}', f'GPU{gpu_id}: Q Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[2])
            c.node(f's2_k_{gpu_id}', f'GPU{gpu_id}: K Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[2])
            c.node(f's2_v_{gpu_id}', f'GPU{gpu_id}: V Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[2])
            
            # Communication
            c.node(f's2_allgather_q_{gpu_id}', f'GPU{gpu_id}: AllGather Q\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's2_allgather_k_{gpu_id}', f'GPU{gpu_id}: AllGather K\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's2_allgather_v_{gpu_id}', f'GPU{gpu_id}: AllGather V\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            
            # Attention
            c.node(f's2_attn_{gpu_id}', f'GPU{gpu_id}: Attention\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[2])
            
            # MoE
            c.node(f's2_gate_{gpu_id}', f'GPU{gpu_id}: MoE Gate\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, experts=8]', 
                   shape='parallelogram', fillcolor='orange')
            c.node(f's2_expert_{gpu_id}', f'GPU{gpu_id}: Expert FFN\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[2])
            c.node(f's2_alltoall_{gpu_id}', f'GPU{gpu_id}: AllToAll\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's2_ln1_{gpu_id}', f'GPU{gpu_id}: LayerNorm\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[2])
    
    # Stage 3: GPUs 24-31 (PP stage 3)
    with dot.subgraph(name='stage3') as c:
        c.attr(rank='same')
        for gpu_id in range(24, 32):
            # Similar operations for stage 3
            c.node(f's3_ln0_{gpu_id}', f'GPU{gpu_id}: LayerNorm\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[3])
            
            # Attention operations
            c.node(f's3_q_{gpu_id}', f'GPU{gpu_id}: Q Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[3])
            c.node(f's3_k_{gpu_id}', f'GPU{gpu_id}: K Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[3])
            c.node(f's3_v_{gpu_id}', f'GPU{gpu_id}: V Linear\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=768]', 
                   fillcolor=stage_colors[3])
            
            # Communication
            c.node(f's3_allgather_q_{gpu_id}', f'GPU{gpu_id}: AllGather Q\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's3_allgather_k_{gpu_id}', f'GPU{gpu_id}: AllGather K\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's3_allgather_v_{gpu_id}', f'GPU{gpu_id}: AllGather V\\nInput: [batch_size=1, seq_len=4096, hidden=768]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            
            # Attention
            c.node(f's3_attn_{gpu_id}', f'GPU{gpu_id}: Attention\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[3])
            
            # MoE
            c.node(f's3_gate_{gpu_id}', f'GPU{gpu_id}: MoE Gate\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, experts=8]', 
                   shape='parallelogram', fillcolor='orange')
            c.node(f's3_expert_{gpu_id}', f'GPU{gpu_id}: Expert FFN\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[3])
            c.node(f's3_alltoall_{gpu_id}', f'GPU{gpu_id}: AllToAll\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's3_ln1_{gpu_id}', f'GPU{gpu_id}: LayerNorm\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
                   fillcolor=stage_colors[3])
    
    # Output node
    dot.node('output', 'OUTPUT\\nInput: [batch_size=1, seq_len=4096, hidden=6144]\\nOutput: [batch_size=1, seq_len=4096, hidden=6144]', 
             shape='ellipse', fillcolor='white')
    
    # Connect nodes (simplified for brevity - showing key connections)
    # Input to stage 0
    for gpu_id in range(8):
        dot.edge('input', f's0_emb_{gpu_id}')
        dot.edge(f's0_emb_{gpu_id}', f's0_ln0_{gpu_id}')
        dot.edge(f's0_ln0_{gpu_id}', f's0_q_{gpu_id}')
        dot.edge(f's0_ln0_{gpu_id}', f's0_k_{gpu_id}')
        dot.edge(f's0_ln0_{gpu_id}', f's0_v_{gpu_id}')
        
        # TP communication
        dot.edge(f's0_q_{gpu_id}', f's0_allgather_q_{gpu_id}')
        dot.edge(f's0_k_{gpu_id}', f's0_allgather_k_{gpu_id}')
        dot.edge(f's0_v_{gpu_id}', f's0_allgather_v_{gpu_id}')
        
        # Attention
        dot.edge(f's0_allgather_q_{gpu_id}', f's0_attn_{gpu_id}')
        dot.edge(f's0_allgather_k_{gpu_id}', f's0_attn_{gpu_id}')
        dot.edge(f's0_allgather_v_{gpu_id}', f's0_attn_{gpu_id}')
        
        # MoE
        dot.edge(f's0_attn_{gpu_id}', f's0_gate_{gpu_id}')
        dot.edge(f's0_gate_{gpu_id}', f's0_expert_{gpu_id}', style='dashed')
        dot.edge(f's0_expert_{gpu_id}', f's0_alltoall_{gpu_id}')
        dot.edge(f's0_alltoall_{gpu_id}', f's0_ln1_{gpu_id}')
    
    # Pipeline stage connections
    for gpu_id in range(8):
        dot.edge(f's0_ln1_{gpu_id}', f's1_ln0_{gpu_id+8}')
    
    for gpu_id in range(8, 16):
        dot.edge(f's1_ln1_{gpu_id}', f's2_ln0_{gpu_id+8}')
    
    for gpu_id in range(16, 24):
        dot.edge(f's2_ln1_{gpu_id}', f's3_ln0_{gpu_id+8}')
    
    # Final output
    for gpu_id in range(24, 32):
        dot.edge(f's3_ln1_{gpu_id}', 'output')
    
    return dot

def create_decode_dag():
    """Create decode phase DAG with PP=2, TP=4, EP=32, DP=4"""
    dot = graphviz.Digraph(comment='Decode Phase DAG')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Define colors for different GPU stages
    stage_colors = ['lightblue', 'lightgreen']
    
    # Input node
    dot.node('input', 'INPUT\\nInput: [batch_size=4, seq_len=1, hidden=6144]\\nOutput: [batch_size=4, seq_len=1, hidden=6144]', 
             shape='ellipse', fillcolor='white')
    
    # Stage 0: GPUs 0-15 (PP stage 0, DP=4, TP=4)
    with dot.subgraph(name='stage0') as c:
        c.attr(rank='same')
        for gpu_id in range(16):
            dp_group = gpu_id // 4
            tp_group = gpu_id % 4
            
            # Embedding layer (only for first token in each sequence)
            c.node(f's0_emb_{gpu_id}', f'GPU{gpu_id}: Embedding (DP{dp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   fillcolor=stage_colors[0])
            
            # Layer norm
            c.node(f's0_ln0_{gpu_id}', f'GPU{gpu_id}: LayerNorm (DP{dp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   fillcolor=stage_colors[0])
            
            # Attention operations (TP=4)
            c.node(f's0_q_{gpu_id}', f'GPU{gpu_id}: Q Linear (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=1536]', 
                   fillcolor=stage_colors[0])
            c.node(f's0_k_{gpu_id}', f'GPU{gpu_id}: K Linear (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=1536]', 
                   fillcolor=stage_colors[0])
            c.node(f's0_v_{gpu_id}', f'GPU{gpu_id}: V Linear (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=1536]', 
                   fillcolor=stage_colors[0])
            
            # Communication nodes
            c.node(f's0_allgather_q_{gpu_id}', f'GPU{gpu_id}: AllGather Q (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=1536]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's0_allgather_k_{gpu_id}', f'GPU{gpu_id}: AllGather K (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=1536]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's0_allgather_v_{gpu_id}', f'GPU{gpu_id}: AllGather V (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=1536]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            
            # Attention computation
            c.node(f's0_attn_{gpu_id}', f'GPU{gpu_id}: Attention (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   fillcolor=stage_colors[0])
            
            # MoE operations (EP=32)
            c.node(f's0_gate_{gpu_id}', f'GPU{gpu_id}: MoE Gate\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, experts=8]', 
                   shape='parallelogram', fillcolor='orange')
            
            # Expert computation (1 expert per GPU)
            c.node(f's0_expert_{gpu_id}', f'GPU{gpu_id}: Expert FFN\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   fillcolor=stage_colors[0])
            
            # Communication for MoE
            c.node(f's0_alltoall_{gpu_id}', f'GPU{gpu_id}: AllToAll\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            
            # Final layer norm
            c.node(f's0_ln1_{gpu_id}', f'GPU{gpu_id}: LayerNorm (DP{dp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   fillcolor=stage_colors[0])
    
    # Stage 1: GPUs 16-31 (PP stage 1, DP=4, TP=4)
    with dot.subgraph(name='stage1') as c:
        c.attr(rank='same')
        for gpu_id in range(16, 32):
            dp_group = (gpu_id - 16) // 4
            tp_group = (gpu_id - 16) % 4
            
            # Similar operations for stage 1
            c.node(f's1_ln0_{gpu_id}', f'GPU{gpu_id}: LayerNorm (DP{dp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   fillcolor=stage_colors[1])
            
            # Attention operations
            c.node(f's1_q_{gpu_id}', f'GPU{gpu_id}: Q Linear (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=1536]', 
                   fillcolor=stage_colors[1])
            c.node(f's1_k_{gpu_id}', f'GPU{gpu_id}: K Linear (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=1536]', 
                   fillcolor=stage_colors[1])
            c.node(f's1_v_{gpu_id}', f'GPU{gpu_id}: V Linear (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=1536]', 
                   fillcolor=stage_colors[1])
            
            # Communication
            c.node(f's1_allgather_q_{gpu_id}', f'GPU{gpu_id}: AllGather Q (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=1536]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's1_allgather_k_{gpu_id}', f'GPU{gpu_id}: AllGather K (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=1536]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's1_allgather_v_{gpu_id}', f'GPU{gpu_id}: AllGather V (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=1536]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            
            # Attention
            c.node(f's1_attn_{gpu_id}', f'GPU{gpu_id}: Attention (TP{tp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   fillcolor=stage_colors[1])
            
            # MoE
            c.node(f's1_gate_{gpu_id}', f'GPU{gpu_id}: MoE Gate\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, experts=8]', 
                   shape='parallelogram', fillcolor='orange')
            c.node(f's1_expert_{gpu_id}', f'GPU{gpu_id}: Expert FFN\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   fillcolor=stage_colors[1])
            c.node(f's1_alltoall_{gpu_id}', f'GPU{gpu_id}: AllToAll\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   shape='ellipse', fillcolor='white')
            c.node(f's1_ln1_{gpu_id}', f'GPU{gpu_id}: LayerNorm (DP{dp_group})\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
                   fillcolor=stage_colors[1])
    
    # Output node
    dot.node('output', 'OUTPUT\\nInput: [batch_size=1, seq_len=1, hidden=6144]\\nOutput: [batch_size=1, seq_len=1, hidden=6144]', 
             shape='ellipse', fillcolor='white')
    
    # Connect nodes (simplified for brevity)
    # Input to stage 0
    for gpu_id in range(16):
        dot.edge('input', f's0_emb_{gpu_id}')
        dot.edge(f's0_emb_{gpu_id}', f's0_ln0_{gpu_id}')
        dot.edge(f's0_ln0_{gpu_id}', f's0_q_{gpu_id}')
        dot.edge(f's0_ln0_{gpu_id}', f's0_k_{gpu_id}')
        dot.edge(f's0_ln0_{gpu_id}', f's0_v_{gpu_id}')
        
        # TP communication
        dot.edge(f's0_q_{gpu_id}', f's0_allgather_q_{gpu_id}')
        dot.edge(f's0_k_{gpu_id}', f's0_allgather_k_{gpu_id}')
        dot.edge(f's0_v_{gpu_id}', f's0_allgather_v_{gpu_id}')
        
        # Attention
        dot.edge(f's0_allgather_q_{gpu_id}', f's0_attn_{gpu_id}')
        dot.edge(f's0_allgather_k_{gpu_id}', f's0_attn_{gpu_id}')
        dot.edge(f's0_allgather_v_{gpu_id}', f's0_attn_{gpu_id}')
        
        # MoE
        dot.edge(f's0_attn_{gpu_id}', f's0_gate_{gpu_id}')
        dot.edge(f's0_gate_{gpu_id}', f's0_expert_{gpu_id}', style='dashed')
        dot.edge(f's0_expert_{gpu_id}', f's0_alltoall_{gpu_id}')
        dot.edge(f's0_alltoall_{gpu_id}', f's0_ln1_{gpu_id}')
    
    # Pipeline stage connections
    for gpu_id in range(16):
        dot.edge(f's0_ln1_{gpu_id}', f's1_ln0_{gpu_id+16}')
    
    # Final output
    for gpu_id in range(16, 32):
        dot.edge(f's1_ln1_{gpu_id}', 'output')
    
    return dot

if __name__ == "__main__":
    # Create prefill DAG
    prefill_dag = create_prefill_dag()
    prefill_dag.render('./outputs/2026-01-04-17-39-40/prefill_dag', format='dot', cleanup=False)
    prefill_dag.render('./outputs/2026-01-04-17-39-40/prefill_dag', format='svg', cleanup=False)
    
    # Create decode DAG
    decode_dag = create_decode_dag()
    decode_dag.render('./outputs/2026-01-04-17-39-40/decode_dag', format='dot', cleanup=False)
    decode_dag.render('./outputs/2026-01-04-17-39-40/decode_dag', format='svg', cleanup=False)
    
    print("DAGs generated successfully!")