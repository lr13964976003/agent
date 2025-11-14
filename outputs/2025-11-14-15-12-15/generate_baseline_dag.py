import graphviz
from typing import Dict, List, Tuple
import math

# Constants from deployment config
BATCH_SIZE = 128
SEQ_LEN = 10000
HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128
MLP_HIDDEN_SIZE = 16384

# Create baseline DAG (TP=8, PP=2)
def create_baseline_dag():
    dot = graphviz.Digraph('baseline_dag', 
                          comment='4-layer Dense Model - Baseline TP=8 PP=2',
                          graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.5', 'ranksep': '1.0'})
    
    # Set node attributes for different types
    dot.attr('node', shape='ellipse')  # Communication nodes
    dot.attr('node', shape='rectangle')  # Computation nodes
    dot.attr('node', shape='parallelogram')  # Routing/aggregation nodes
    
    # Input node
    dot.node('input', 'Input\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: All', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Stage 0: Layers 0-1 on GPUs 0-7
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='dashed', color='red')
        
        # Layer 0 with tensor parallelism across 8 GPUs
        with stage0.subgraph(name='cluster_layer0') as layer0:
            layer0.attr(label='Layer 0', style='dashed', color='blue')
            
            # Input split for Layer 0
            dot.node('l0_input_split', 'Input Split\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            # Multi-Head Attention
            dot.node('l0_mha_qkv', 'MHA QKV Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l0_mha_attn', 'MHA Attention\nInput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # FFN
            dot.node('l0_ffn_up', 'FFN Up Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l0_ffn_gate', 'FFN Gate Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l0_ffn_act', 'FFN Activation\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l0_ffn_down', 'FFN Down Linear\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Residual and LayerNorm
            dot.node('l0_residual', 'Residual Add\nInput: [batch_size=128, seq_len=10000, hidden_size=4096] (x2)\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l0_layernorm', 'LayerNorm\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # All-reduce aggregations
            dot.node('l0_mha_agg', 'MHA All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
            dot.node('l0_ffn_agg', 'FFN All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Layer 1 (similar structure)
        with stage0.subgraph(name='cluster_layer1') as layer1:
            layer1.attr(label='Layer 1', style='dashed', color='blue')
            
            dot.node('l1_mha_qkv', 'MHA QKV Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l1_mha_attn', 'MHA Attention\nInput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            dot.node('l1_ffn_up', 'FFN Up Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l1_ffn_gate', 'FFN Gate Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l1_ffn_act', 'FFN Activation\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l1_ffn_down', 'FFN Down Linear\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            dot.node('l1_residual', 'Residual Add\nInput: [batch_size=128, seq_len=10000, hidden_size=4096] (x2)\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l1_layernorm', 'LayerNorm\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            dot.node('l1_mha_agg', 'MHA All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
            dot.node('l1_ffn_agg', 'FFN All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-7', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Inter-stage communication
    dot.node('stage0_to_stage1', 'Pipeline Send\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0→8', 
            shape='ellipse', style='filled', fillcolor='orange')
    
    # Stage 1: Layers 2-3 on GPUs 8-15
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='dashed', color='red')
        
        # Layer 2
        with stage1.subgraph(name='cluster_layer2') as layer2:
            layer2.attr(label='Layer 2', style='dashed', color='blue')
            
            dot.node('l2_mha_qkv', 'MHA QKV Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l2_mha_attn', 'MHA Attention\nInput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            dot.node('l2_ffn_up', 'FFN Up Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l2_ffn_gate', 'FFN Gate Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l2_ffn_act', 'FFN Activation\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l2_ffn_down', 'FFN Down Linear\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            dot.node('l2_residual', 'Residual Add\nInput: [batch_size=128, seq_len=10000, hidden_size=4096] (x2)\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l2_layernorm', 'LayerNorm\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            dot.node('l2_mha_agg', 'MHA All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
            dot.node('l2_ffn_agg', 'FFN All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Layer 3
        with stage1.subgraph(name='cluster_layer3') as layer3:
            layer3.attr(label='Layer 3', style='dashed', color='blue')
            
            dot.node('l3_mha_qkv', 'MHA QKV Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l3_mha_attn', 'MHA Attention\nInput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            dot.node('l3_ffn_up', 'FFN Up Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l3_ffn_gate', 'FFN Gate Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l3_ffn_act', 'FFN Activation\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l3_ffn_down', 'FFN Down Linear\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            dot.node('l3_residual', 'Residual Add\nInput: [batch_size=128, seq_len=10000, hidden_size=4096] (x2)\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dot.node('l3_layernorm', 'LayerNorm\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            
            dot.node('l3_mha_agg', 'MHA All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
            dot.node('l3_ffn_agg', 'FFN All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Output node
    dot.node('output', 'Output\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-15', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect the DAG
    # Input to Layer 0
    dot.edge('input', 'l0_input_split')
    dot.edge('l0_input_split', 'l0_mha_qkv')
    dot.edge('l0_mha_qkv', 'l0_mha_attn')
    dot.edge('l0_mha_attn', 'l0_mha_agg')
    dot.edge('l0_mha_agg', 'l0_residual')
    dot.edge('l0_residual', 'l0_layernorm')
    dot.edge('l0_layernorm', 'l0_ffn_up')
    dot.edge('l0_layernorm', 'l0_ffn_gate')
    dot.edge('l0_ffn_up', 'l0_ffn_act')
    dot.edge('l0_ffn_gate', 'l0_ffn_act')
    dot.edge('l0_ffn_act', 'l0_ffn_down')
    dot.edge('l0_ffn_down', 'l0_ffn_agg')
    dot.edge('l0_ffn_agg', 'l0_residual')
    
    # Layer 0 to Layer 1
    dot.edge('l0_residual', 'l1_mha_qkv')
    dot.edge('l1_mha_qkv', 'l1_mha_attn')
    dot.edge('l1_mha_attn', 'l1_mha_agg')
    dot.edge('l1_mha_agg', 'l1_residual')
    dot.edge('l1_residual', 'l1_layernorm')
    dot.edge('l1_layernorm', 'l1_ffn_up')
    dot.edge('l1_layernorm', 'l1_ffn_gate')
    dot.edge('l1_ffn_up', 'l1_ffn_act')
    dot.edge('l1_ffn_gate', 'l1_ffn_act')
    dot.edge('l1_ffn_act', 'l1_ffn_down')
    dot.edge('l1_ffn_down', 'l1_ffn_agg')
    dot.edge('l1_ffn_agg', 'l1_residual')
    
    # Stage 0 to Stage 1
    dot.edge('l1_residual', 'stage0_to_stage1')
    dot.edge('stage0_to_stage1', 'l2_mha_qkv')
    
    # Layer 2
    dot.edge('l2_mha_qkv', 'l2_mha_attn')
    dot.edge('l2_mha_attn', 'l2_mha_agg')
    dot.edge('l2_mha_agg', 'l2_residual')
    dot.edge('l2_residual', 'l2_layernorm')
    dot.edge('l2_layernorm', 'l2_ffn_up')
    dot.edge('l2_layernorm', 'l2_ffn_gate')
    dot.edge('l2_ffn_up', 'l2_ffn_act')
    dot.edge('l2_ffn_gate', 'l2_ffn_act')
    dot.edge('l2_ffn_act', 'l2_ffn_down')
    dot.edge('l2_ffn_down', 'l2_ffn_agg')
    dot.edge('l2_ffn_agg', 'l2_residual')
    
    # Layer 3
    dot.edge('l2_residual', 'l3_mha_qkv')
    dot.edge('l3_mha_qkv', 'l3_mha_attn')
    dot.edge('l3_mha_attn', 'l3_mha_agg')
    dot.edge('l3_mha_agg', 'l3_residual')
    dot.edge('l3_residual', 'l3_layernorm')
    dot.edge('l3_layernorm', 'l3_ffn_up')
    dot.edge('l3_layernorm', 'l3_ffn_gate')
    dot.edge('l3_ffn_up', 'l3_ffn_act')
    dot.edge('l3_ffn_gate', 'l3_ffn_act')
    dot.edge('l3_ffn_act', 'l3_ffn_down')
    dot.edge('l3_ffn_down', 'l3_ffn_agg')
    dot.edge('l3_ffn_agg', 'l3_residual')
    dot.edge('l3_residual', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_dag()
    dag.render('../outputs/2025-11-14-15-12-15/baseline_dag', format='svg', cleanup=False)
    dag.save('../outputs/2025-11-14-15-12-15/baseline_dag.dot')