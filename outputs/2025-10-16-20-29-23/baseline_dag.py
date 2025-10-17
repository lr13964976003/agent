#!/usr/bin/env python3
"""
Baseline Model DAG Generator (TP=8, PP=2)
16 GPUs total: 8 GPUs per pipeline stage
"""

import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('baseline_tp8_pp2')
    dot.attr(rankdir='TB', compound='true', ranksep='2.0', nodesep='0.8')
    
    # Model parameters
    batch_size = 1024
    seq_len = 2048
    hidden_dim = 4096
    num_heads = 32
    head_dim = 128
    vocab_size = 50265
    
    # Add subgraphs for pipeline stages
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='rounded', color='blue', bgcolor='lightblue')
        stage0.node_attr.update(shape='rectangle', style='filled', fillcolor='white')
        
        # Layer 0 nodes
        stage0.node('input_l0', 'Input\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: all GPUs',
                   shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Token embedding (distributed across stage 0 GPUs)
        stage0.node('embed_l0', 'Token Embedding\\nInput: [batch_size=1024, seq_len=2048]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7 (TP)')
        
        # Layer norm (before attention)
        stage0.node('ln1_l0', 'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7 (TP)')
        
        # QKV projection (split across 8 GPUs by column)
        stage0.node('qkv_l0', 'QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, num_heads=32, head_dim=128, 3]\\nGPU: 0-7 (TP split: 512 dim per GPU)')
        
        # Attention computation (heads split across GPUs)
        stage0.node('attn_scores_l0', 'Attention Scores\\nInput: [batch_size=1024, seq_len=2048, num_heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, num_heads=4, seq_len=2048]\\nGPU: 0-7 (4 heads per GPU)')
        
        stage0.node('attn_weights_l0', 'Softmax\\nInput: [batch_size=1024, seq_len=2048, num_heads=4, seq_len=2048]\\nOutput: [batch_size=1024, seq_len=2048, num_heads=4, seq_len=2048]\\nGPU: 0-7')
        
        stage0.node('attn_output_l0', 'Attention Output\\nInput: [batch_size=1024, seq_len=2048, num_heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, num_heads=4, head_dim=128]\\nGPU: 0-7')
        
        # Output projection (row split across GPUs)
        stage0.node('o_proj_l0', 'Output Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7 (TP row split)')
        
        # All-reduce for attention output
        stage0.node('all_reduce_l0', 'All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7',
                   shape='ellipse', style='filled', fillcolor='yellow')
        
        # Residual add
        stage0.node('res1_l0', 'Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7',
                   shape='ellipse', style='filled', fillcolor='pink')
        
        # Second layer norm
        stage0.node('ln2_l0', 'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7')
        
        # MLP (column then row split)
        stage0.node('mlp_up_l0', 'MLP Up-Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, 16384]\\nGPU: 0-7 (TP column split)')
        
        stage0.node('mlp_act_l0', 'GeLU Activation\\nInput: [batch_size=1024, seq_len=2048, 16384]\\nOutput: [batch_size=1024, seq_len=2048, 16384]\\nGPU: 0-7')
        
        stage0.node('mlp_down_l0', 'MLP Down-Projection\\nInput: [batch_size=1024, seq_len=2048, 16384]\\nOutput: [batch_size=1024, seq_len=2048, 4096]\\nGPU: 0-7 (TP row split)')
        
        # All-reduce for MLP output
        stage0.node('all_reduce_mlp_l0', 'All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7',
                   shape='ellipse', style='filled', fillcolor='yellow')
        
        # Second residual add
        stage0.node('res2_l0', 'Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7',
                   shape='ellipse', style='filled', fillcolor='pink')
        
        # Layer 1 (similar structure)
        stage0.node('ln1_l1', 'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7')
        
        stage0.node('qkv_l1', 'QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, num_heads=32, head_dim=128, 3]\\nGPU: 0-7 (TP split)')
        
        stage0.node('attn_scores_l1', 'Attention Scores\\nInput: [batch_size=1024, seq_len=2048, num_heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, num_heads=4, seq_len=2048]\\nGPU: 0-7 (4 heads per GPU)')
        
        stage0.node('o_proj_l1', 'Output Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7')
        
        stage0.node('all_reduce_l1', 'All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7',
                   shape='ellipse', style='filled', fillcolor='yellow')
        
        stage0.node('res1_l1', 'Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7',
                   shape='ellipse', style='filled', fillcolor='pink')
        
        stage0.node('ln2_l1', 'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7')
        
        stage0.node('mlp_up_l1', 'MLP Up-Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, 16384]\\nGPU: 0-7')
        
        stage0.node('mlp_down_l1', 'MLP Down-Projection\\nInput: [batch_size=1024, seq_len=2048, 16384]\\nOutput: [batch_size=1024, seq_len=2048, 4096]\\nGPU: 0-7')
        
        stage0.node('all_reduce_mlp_l1', 'All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7',
                   shape='ellipse', style='filled', fillcolor='yellow')
        
        stage0.node('res2_l1', 'Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7',
                   shape='ellipse', style='filled', fillcolor='pink')
        
        # Pipeline send to stage 1
        stage0.node('send_stage1', 'Send to Stage 1\\nFrom: GPUs 0-7\\nTo: GPUs 8-15\\nData: [batch_size=1024, seq_len=2048, hidden_dim=4096]',
                   shape='parallelogram', style='filled', fillcolor='lightcyan')

    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='rounded', color='green', bgcolor='lightgreen')
        stage1.node_attr.update(shape='rectangle', style='filled', fillcolor='white')
        
        # Pipeline receive from stage 0
        stage1.node('recv_stage1', 'Receive from Stage 0\\nFrom: GPUs 0-7\\nTo: GPUs 8-15\\nData: [batch_size=1024, seq_len=2048, hidden_dim=4096]',
                   shape='parallelogram', style='filled', fillcolor='lightcyan')
        
        # Layer 2 (similar to layer 0)
        stage1.node('ln1_l2', 'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15')
        
        stage1.node('qkv_l2', 'QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, num_heads=32, head_dim=128, 3]\\nGPU: 8-15')
        
        stage1.node('attn_scores_l2', 'Attention Scores\\nInput: [batch_size=1024, seq_len=2048, num_heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, num_heads=4, seq_len=2048]\\nGPU: 8-15')
        
        stage1.node('o_proj_l2', 'Output Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15')
        
        stage1.node('all_reduce_l2', 'All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15',
                   shape='ellipse', style='filled', fillcolor='yellow')
        
        stage1.node('res1_l2', 'Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15',
                   shape='ellipse', style='filled', fillcolor='pink')
        
        stage1.node('ln2_l2', 'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15')
        
        stage1.node('mlp_up_l2', 'MLP Up-Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, 16384]\\nGPU: 8-15')
        
        stage1.node('mlp_down_l2', 'MLP Down-Projection\\nInput: [batch_size=1024, seq_len=2048, 16384]\\nOutput: [batch_size=1024, seq_len=2048, 4096]\\nGPU: 8-15')
        
        stage1.node('all_reduce_mlp_l2', 'All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15',
                   shape='ellipse', style='filled', fillcolor='yellow')
        
        stage1.node('res2_l2', 'Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15',
                   shape='ellipse', style='filled', fillcolor='pink')
        
        # Layer 3
        stage1.node('ln1_l3', 'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15')
        
        stage1.node('qkv_l3', 'QKV Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, num_heads=32, head_dim=128, 3]\\nGPU: 8-15')
        
        stage1.node('attn_scores_l3', 'Attention Scores\\nInput: [batch_size=1024, seq_len=2048, num_heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, num_heads=4, seq_len=2048]\\nGPU: 8-15')
        
        stage1.node('o_proj_l3', 'Output Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15')
        
        stage1.node('all_reduce_l3', 'All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15',
                   shape='ellipse', style='filled', fillcolor='yellow')
        
        stage1.node('res1_l3', 'Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15',
                   shape='ellipse', style='filled', fillcolor='pink')
        
        stage1.node('ln2_l3', 'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15')
        
        stage1.node('mlp_up_l3', 'MLP Up-Projection\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, 16384]\\nGPU: 8-15')
        
        stage1.node('mlp_down_l3', 'MLP Down-Projection\\nInput: [batch_size=1024, seq_len=2048, 16384]\\nOutput: [batch_size=1024, seq_len=2048, 4096]\\nGPU: 8-15')
        
        stage1.node('all_reduce_mlp_l3', 'All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15',
                   shape='ellipse', style='filled', fillcolor='yellow')
        
        stage1.node('res2_l3', 'Residual Add\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15',
                   shape='ellipse', style='filled', fillcolor='pink')
        
        # Final output
        stage1.node('output', 'Model Output\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, vocab_size=50265]\\nGPU: 8-15',
                   shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Add edges
    # Stage 0 connections
    dot.edge('input_l0', 'embed_l0')
    dot.edge('embed_l0', 'ln1_l0')
    dot.edge('ln1_l0', 'qkv_l0')
    dot.edge('qkv_l0', 'attn_scores_l0')
    dot.edge('attn_scores_l0', 'attn_weights_l0')
    dot.edge('attn_weights_l0', 'o_proj_l0')
    dot.edge('o_proj_l0', 'all_reduce_l0')
    dot.edge('all_reduce_l0', 'res1_l0')
    dot.edge('res1_l0', 'ln2_l0')
    dot.edge('ln2_l0', 'mlp_up_l0')
    dot.edge('mlp_up_l0', 'mlp_act_l0')
    dot.edge('mlp_act_l0', 'mlp_down_l0')
    dot.edge('mlp_down_l0', 'all_reduce_mlp_l0')
    dot.edge('all_reduce_mlp_l0', 'res2_l0')
    
    # Layer 1
    dot.edge('res2_l0', 'ln1_l1')
    dot.edge('ln1_l1', 'qkv_l1')
    dot.edge('qkv_l1', 'attn_scores_l1')
    dot.edge('attn_scores_l1', 'o_proj_l1')
    dot.edge('o_proj_l1', 'all_reduce_l1')
    dot.edge('all_reduce_l1', 'res1_l1')
    dot.edge('res1_l1', 'ln2_l1')
    dot.edge('ln2_l1', 'mlp_up_l1')
    dot.edge('mlp_up_l1', 'mlp_down_l1')
    dot.edge('mlp_down_l1', 'all_reduce_mlp_l1')
    dot.edge('all_reduce_mlp_l1', 'res2_l1')
    dot.edge('res2_l1', 'send_stage1')
    
    # Stage 1 connections
    dot.edge('send_stage1', 'recv_stage1', lhead='cluster_stage1', ltail='cluster_stage0')
    dot.edge('recv_stage1', 'ln1_l2')
    dot.edge('ln1_l2', 'qkv_l2')
    dot.edge('qkv_l2', 'attn_scores_l2')
    dot.edge('attn_scores_l2', 'o_proj_l2')
    dot.edge('o_proj_l2', 'all_reduce_l2')
    dot.edge('all_reduce_l2', 'res1_l2')
    dot.edge('res1_l2', 'ln2_l2')
    dot.edge('ln2_l2', 'mlp_up_l2')
    dot.edge('mlp_up_l2', 'mlp_down_l2')
    dot.edge('mlp_down_l2', 'all_reduce_mlp_l2')
    dot.edge('all_reduce_mlp_l2', 'res2_l2')
    
    # Layer 3
    dot.edge('res2_l2', 'ln1_l3')
    dot.edge('ln1_l3', 'qkv_l3')
    dot.edge('qkv_l3', 'attn_scores_l3')
    dot.edge('attn_scores_l3', 'o_proj_l3')
    dot.edge('o_proj_l3', 'all_reduce_l3')
    dot.edge('all_reduce_l3', 'res1_l3')
    dot.edge('res1_l3', 'ln2_l3')
    dot.edge('ln2_l3', 'mlp_up_l3')
    dot.edge('mlp_up_l3', 'mlp_down_l3')
    dot.edge('mlp_down_l3', 'all_reduce_mlp_l3')
    dot.edge('all_reduce_mlp_l3', 'res2_l3')
    dot.edge('res2_l3', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('../outputs/2025-10-16-20-29-23/baseline_dag', format='svg', cleanup=True)
    dag.save('../outputs/2025-10-16-20-29-23/baseline_dag.dot')
    print("Baseline DAG generated successfully")