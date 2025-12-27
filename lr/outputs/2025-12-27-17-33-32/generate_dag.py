#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_parallelism_dag():
    """
    Create a complete DAG for the MoE model deployment with all parallelism strategies.
    
    Parallelism Configuration:
    - EP=16 (Expert Parallelism): 16 experts per layer, each on separate GPU
    - PP=16 (Pipeline Parallelism): 16 layers, each layer is a pipeline stage  
    - TP=2 (Tensor Parallelism): Attention and FFN operators split across 2 GPUs
    - SP=2 (Sequence Parallelism): Token dimension split during prefill
    - DP=8 (Data Parallelism): 8 replicas of the full model
    
    Total GPUs: 2048 (256 experts × 8 DP replicas)
    """
    
    dot = Digraph(comment='MoE Model Parallelism Deployment DAG')
    dot.attr(bgcolor='white', rankdir='TB', size='30,40')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10240, dim=512]\\nOutput: [batch_size=128, seq_len=10240, dim=512]', 
             shape='ellipse', fillcolor='lightblue')
    
    # For simplicity, I'll show one DP replica (256 GPUs) in detail
    # and indicate the other 7 DP replicas exist
    
    # Layer 1 - showing detailed breakdown
    with dot.subgraph(name='cluster_layer1') as c:
        c.attr(label='Layer 1 (PP Stage 1)', style='rounded,filled', fillcolor='lightgray')
        
        # SP split - token dimension
        c.node('sp_split_1', 'SP Split\\nGPU: 0-1\\nInput: [batch_size=128, seq_len=10240, dim=512]\\nOutput: [batch_size=128, seq_len=5120, dim=512]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Attention with TP
        c.node('attn_q_1', 'Attention Q Projection\\nGPU: 0\\nTP Rank 0\\nInput: [batch_size=128, seq_len=5120, dim=512]\\nOutput: [batch_size=128, seq_len=5120, dim=256]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('attn_q_1_tp1', 'Attention Q Projection\\nGPU: 1\\nTP Rank 1\\nInput: [batch_size=128, seq_len=5120, dim=512]\\nOutput: [batch_size=128, seq_len=5120, dim=256]', 
               shape='rectangle', fillcolor='lightgreen')
        
        c.node('attn_k_1', 'Attention K Projection\\nGPU: 0\\nTP Rank 0\\nInput: [batch_size=128, seq_len=5120, dim=512]\\nOutput: [batch_size=128, seq_len=5120, dim=256]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('attn_k_1_tp1', 'Attention K Projection\\nGPU: 1\\nTP Rank 1\\nInput: [batch_size=128, seq_len=5120, dim=512]\\nOutput: [batch_size=128, seq_len=5120, dim=256]', 
               shape='rectangle', fillcolor='lightgreen')
        
        c.node('attn_v_1', 'Attention V Projection\\nGPU: 0\\nTP Rank 0\\nInput: [batch_size=128, seq_len=5120, dim=512]\\nOutput: [batch_size=128, seq_len=5120, dim=256]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('attn_v_1_tp1', 'Attention V Projection\\nGPU: 1\\nTP Rank 1\\nInput: [batch_size=128, seq_len=5120, dim=512]\\nOutput: [batch_size=128, seq_len=5120, dim=256]', 
               shape='rectangle', fillcolor='lightgreen')
        
        # Attention computation
        c.node('attn_comp_1', 'Attention Computation\\nGPU: 0\\nTP Rank 0\\nInput: [batch_size=128, seq_len=5120, heads=16, d_k=32]\\nOutput: [batch_size=128, seq_len=5120, dim=256]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('attn_comp_1_tp1', 'Attention Computation\\nGPU: 1\\nTP Rank 1\\nInput: [batch_size=128, seq_len=5120, heads=16, d_k=32]\\nOutput: [batch_size=128, seq_len=5120, dim=256]', 
               shape='rectangle', fillcolor='lightgreen')
        
        # Attention output projection
        c.node('attn_out_1', 'Attention Output Proj\\nGPU: 0\\nTP Rank 0\\nInput: [batch_size=128, seq_len=5120, dim=256]\\nOutput: [batch_size=128, seq_len=5120, dim=256]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('attn_out_1_tp1', 'Attention Output Proj\\nGPU: 1\\nTP Rank 1\\nInput: [batch_size=128, seq_len=5120, dim=256]\\nOutput: [batch_size=128, seq_len=5120, dim=256]', 
               shape='rectangle', fillcolor='lightgreen')
        
        # TP All-reduce for attention
        c.node('attn_allreduce_1', 'TP All-Reduce\\nGPU: 0-1\\nInput: [batch_size=128, seq_len=5120, dim=256]\\nOutput: [batch_size=128, seq_len=5120, dim=512]', 
               shape='ellipse', fillcolor='lightblue')
        
        # Gate for expert selection
        c.node('gate_1', 'Expert Gate\\nGPU: 0-1\\nInput: [batch_size=128, seq_len=5120, dim=512]\\nOutput: [batch_size=128, seq_len=5120, num_experts=16]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Expert computations - showing first 4 experts explicitly
        for expert_id in range(4):
            gpu_id = expert_id + 2  # Experts start from GPU 2
            c.node(f'expert_{expert_id}_1', f'Expert {expert_id}\\nGPU: {gpu_id}\\nEP Rank {expert_id}\\nInput: [batch_size=128, seq_len=5120, dim=512]\\nOutput: [batch_size=128, seq_len=5120, dim=512]', 
                   shape='rectangle', fillcolor='lightgreen')
        
        # Expert aggregation
        c.node('expert_agg_1', 'Expert Aggregation\\nGPU: 0-1\\nInput: [batch_size=128, seq_len=5120, dim=512] x 16\\nOutput: [batch_size=128, seq_len=5120, dim=512]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # SP merge
        c.node('sp_merge_1', 'SP Merge\\nGPU: 0-1\\nInput: [batch_size=128, seq_len=5120, dim=512]\\nOutput: [batch_size=128, seq_len=10240, dim=512]', 
               shape='parallelogram', fillcolor='lightyellow')
    
    # Communication between layers (PP)
    for layer in range(1, 16):
        dot.node(f'pp_comm_{layer}_{layer+1}', f'PP Communication\\nLayer {layer} -> {layer+1}\\nGPU: All GPUs\\nInput: [batch_size=128, seq_len=10240, dim=512]\\nOutput: [batch_size=128, seq_len=10240, dim=512]', 
                shape='ellipse', fillcolor='lightblue')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=10240, dim=512]\\nOutput: [batch_size=128, seq_len=10240, vocab_size=50000]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Add remaining layers (showing simplified structure)
    for layer in range(2, 17):
        with dot.subgraph(name=f'cluster_layer{layer}') as c:
            c.attr(label=f'Layer {layer} (PP Stage {layer})', style='rounded,filled', fillcolor='lightgray')
            
            # Simplified representation for other layers
            c.node(f'layer_{layer}_block', f'Layer {layer} Complete\\nGPU: 0-255\\nInput: [batch_size=128, seq_len=10240, dim=512]\\nOutput: [batch_size=128, seq_len=10240, dim=512]', 
                   shape='rectangle', fillcolor='lightgreen')
    
    # Data Parallelism indication
    dot.node('dp_replicas', 'Data Parallelism\\n7 Additional Replicas\\nGPU: 256-2047\\nEach replica: 256 GPUs', 
             shape='ellipse', fillcolor='lightcoral')
    
    # Connections
    dot.edge('input', 'sp_split_1')
    dot.edge('sp_split_1', 'attn_q_1')
    dot.edge('sp_split_1', 'attn_q_1_tp1')
    dot.edge('sp_split_1', 'attn_k_1')
    dot.edge('sp_split_1', 'attn_k_1_tp1')
    dot.edge('sp_split_1', 'attn_v_1')
    dot.edge('sp_split_1', 'attn_v_1_tp1')
    
    # Attention connections
    dot.edge('attn_q_1', 'attn_comp_1')
    dot.edge('attn_q_1_tp1', 'attn_comp_1_tp1')
    dot.edge('attn_k_1', 'attn_comp_1')
    dot.edge('attn_k_1_tp1', 'attn_comp_1_tp1')
    dot.edge('attn_v_1', 'attn_comp_1')
    dot.edge('attn_v_1_tp1', 'attn_comp_1_tp1')
    
    dot.edge('attn_comp_1', 'attn_out_1')
    dot.edge('attn_comp_1_tp1', 'attn_out_1_tp1')
    
    dot.edge('attn_out_1', 'attn_allreduce_1')
    dot.edge('attn_out_1_tp1', 'attn_allreduce_1')
    
    # Expert routing (dashed line)
    dot.edge('attn_allreduce_1', 'gate_1', style='dashed')
    
    # Expert computation and aggregation
    dot.edge('attn_allreduce_1', 'expert_0_1')
    dot.edge('attn_allreduce_1', 'expert_1_1')
    dot.edge('attn_allreduce_1', 'expert_2_1')
    dot.edge('attn_allreduce_1', 'expert_3_1')
    
    # Expert outputs to aggregation
    dot.edge('expert_0_1', 'expert_agg_1')
    dot.edge('expert_1_1', 'expert_agg_1')
    dot.edge('expert_2_1', 'expert_agg_1')
    dot.edge('expert_3_1', 'expert_agg_1')
    
    dot.edge('expert_agg_1', 'sp_merge_1')
    
    # Pipeline connections
    dot.edge('sp_merge_1', 'pp_comm_1_2')
    for layer in range(2, 16):
        dot.edge(f'pp_comm_{layer-1}_{layer}', f'layer_{layer}_block')
        dot.edge(f'layer_{layer}_block', f'pp_comm_{layer}_{layer+1}')
    dot.edge('pp_comm_15_16', 'layer_16_block')
    dot.edge('layer_16_block', 'output')
    
    return dot

def create_simplified_dag():
    """Create a more readable simplified DAG showing key concepts"""
    
    dot = Digraph(comment='MoE Model Parallelism Deployment DAG - Simplified')
    dot.attr(bgcolor='white', rankdir='TB', size='20,30')
    dot.attr('node', fontname='Arial', fontsize='12')
    
    # Input
    dot.node('input', 'Input Batch\\n128 sequences, 10240 tokens', 
             shape='ellipse', fillcolor='lightblue')
    
    # Show one complete layer in detail
    with dot.subgraph(name='cluster_layer_detail') as c:
        c.attr(label='Layer 1 Detail (Representative of all 16 layers)', 
               style='rounded,filled', fillcolor='lightgray')
        
        # SP split
        c.node('sp_split', 'SP Split\\nGPU: 0-1\\nSeq len: 10240 → 5120', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Attention with TP
        c.node('attn_tp', 'Attention (TP=2)\\nGPU: 0-1\\nDim: 512 → 256 each', 
               shape='rectangle', fillcolor='lightgreen')
        
        # TP all-reduce
        c.node('tp_allreduce', 'TP All-Reduce\\nGPU: 0-1', 
               shape='ellipse', fillcolor='lightblue')
        
        # Gate
        c.node('gate', 'Expert Gate\\nGPU: 0-1\\nSelects experts', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Experts (showing 4 of 16)
        c.node('experts', 'Expert Computation\\nGPU: 2-17 (16 experts)\\nEP=16', 
               shape='rectangle', fillcolor='lightgreen')
        
        # Expert aggregation
        c.node('expert_agg', 'Expert Aggregation\\nGPU: 0-1', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # SP merge
        c.node('sp_merge', 'SP Merge\\nGPU: 0-1\\nSeq len: 5120 → 10240', 
               shape='parallelogram', fillcolor='lightyellow')
    
    # Pipeline stages
    dot.node('pipeline', 'Pipeline Stages 2-16\\nPP=16\\nGPU: 18-255', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Data parallelism
    dot.node('dp', 'Data Parallelism\\nDP=8\\nGPU: 256-2047\\n8 replicas', 
             shape='ellipse', fillcolor='lightcoral')
    
    # Output
    dot.node('output', 'Output\\n128 sequences, 10240 tokens', 
             shape='ellipse', fillcolor='lightblue')
    
    # Connections
    dot.edge('input', 'sp_split')
    dot.edge('sp_split', 'attn_tp')
    dot.edge('attn_tp', 'tp_allreduce')
    dot.edge('tp_allreduce', 'gate', style='dashed')
    dot.edge('tp_allreduce', 'experts')
    dot.edge('experts', 'expert_agg')
    dot.edge('expert_agg', 'sp_merge')
    dot.edge('sp_merge', 'pipeline')
    dot.edge('pipeline', 'output')
    
    return dot

if __name__ == '__main__':
    # Create the detailed DAG
    detailed_dag = create_parallelism_dag()
    detailed_dag.render('./outputs/2025-12-27-17-33-32/moe_parallelism_detailed', format='dot', cleanup=False)
    detailed_dag.render('./outputs/2025-12-27-17-33-32/moe_parallelism_detailed', format='svg', cleanup=False)
    
    # Create the simplified DAG
    simplified_dag = create_simplified_dag()
    simplified_dag.render('./outputs/2025-12-27-17-33-32/moe_parallelism_simplified', format='dot', cleanup=False)
    simplified_dag.render('./outputs/2025-12-27-17-33-32/moe_parallelism_simplified', format='svg', cleanup=False)
    
    print("DAG files generated successfully!")
    print("- Detailed DAG: moe_parallelism_detailed.dot and .svg")
    print("- Simplified DAG: moe_parallelism_simplified.dot and .svg")