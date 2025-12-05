#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_complete_moe_dag():
    """Create a comprehensive MoE deployment DAG with all requirements"""
    
    # Create DAG
    dot = Digraph(comment='Complete MoE 30B Model Deployment DAG')
    dot.attr(rankdir='TB', bgcolor='white', fontname='Arial')
    
    # Model specifications
    batch_size = 128
    seq_len = 1024
    hidden_size = 4096
    ffn_hidden_size = 16384
    attention_heads = 32
    experts_per_layer = 64
    pipeline_stages = 8
    tensor_parallel = 2
    
    # Input node
    dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Create pipeline stages
    for stage in range(pipeline_stages):
        stage_label = f'stage{stage}'
        layers = f'Layers {stage*2}-{(stage+1)*2-1}'
        
        # Stage input routing
        if stage == 0:
            dot.node(f'{stage_label}_input', f'Stage {stage} Input Router\\nGPU: Pipeline {stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                     shape='parallelogram', fillcolor='yellow')
            dot.edge('input', f'{stage_label}_input')
        else:
            prev_stage = stage - 1
            dot.node(f'{stage_label}_input', f'Stage {stage} Input Router\\nGPU: Pipeline {stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                     shape='parallelogram', fillcolor='yellow')
            dot.edge(f'stage{prev_stage}_output', f'{stage_label}_input')
        
        # Create layers within stage (2 layers per stage)
        for layer in range(2):
            layer_id = stage * 2 + layer
            layer_label = f'{stage_label}_layer{layer}'
            
            # Layer normalization
            dot.node(f'{layer_label}_ln', f'Layer {layer_id} LayerNorm\\nGPU: Pipeline {stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                     shape='rectangle', fillcolor='lightgreen')
            
            if layer == 0:
                dot.edge(f'{stage_label}_input', f'{layer_label}_ln')
            else:
                dot.edge(f'{stage_label}_layer0_output', f'{layer_label}_ln')
            
            # Attention block with multi-head decomposition
            dot.node(f'{layer_label}_attn_qkv', f'Layer {layer_id} Attention QKV Projection\\nGPU: Pipeline {stage} (TP: GPU {stage*128}-{stage*128+127})\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={hidden_size//attention_heads}]',
                     shape='rectangle', fillcolor='lightgreen')
            dot.edge(f'{layer_label}_ln', f'{layer_label}_attn_qkv')
            
            # Tensor parallelism communication for QKV
            dot.node(f'{layer_label}_tp_qkv_comm', f'Layer {layer_id} TP QKV All-Gather\\nGPU: Pipeline {stage} (TP pairs)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={hidden_size//attention_heads}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={hidden_size//attention_heads}]',
                     shape='ellipse', fillcolor='orange')
            dot.edge(f'{layer_label}_attn_qkv', f'{layer_label}_tp_qkv_comm')
            
            # Multi-head attention split (32 heads) - sample first 8 and last 8 for readability
            for head in range(8):  # First 8 heads
                dot.node(f'{layer_label}_attn_head{head}', f'Layer {layer_id} Head {head} Attention\\nGPU: Pipeline {stage} (Head {head})\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_k={hidden_size//attention_heads}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_k={hidden_size//attention_heads}]',
                         shape='rectangle', fillcolor='lightgreen', fontsize='10')
                dot.edge(f'{layer_label}_tp_qkv_comm', f'{layer_label}_attn_head{head}')
            
            # Add ellipsis for remaining heads
            dot.node(f'{layer_label}_attn_ellipsis', f'... 16 more heads ...\\nGPU: Pipeline {stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_k={hidden_size//attention_heads}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_k={hidden_size//attention_heads}]',
                     shape='rectangle', fillcolor='lightgreen', fontsize='10', style='dashed')
            dot.edge(f'{layer_label}_tp_qkv_comm', f'{layer_label}_attn_ellipsis')
            
            for head in range(24, 32):  # Last 8 heads
                dot.node(f'{layer_label}_attn_head{head}', f'Layer {layer_id} Head {head} Attention\\nGPU: Pipeline {stage} (Head {head})\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_k={hidden_size//attention_heads}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_k={hidden_size//attention_heads}]',
                         shape='rectangle', fillcolor='lightgreen', fontsize='10')
                dot.edge(f'{layer_label}_tp_qkv_comm', f'{layer_label}_attn_head{head}')
            
            # Attention aggregation
            dot.node(f'{layer_label}_attn_agg', f'Layer {layer_id} Attention Aggregate\\nGPU: Pipeline {stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={hidden_size//attention_heads}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                     shape='parallelogram', fillcolor='yellow')
            
            for head in range(8):
                dot.edge(f'{layer_label}_attn_head{head}', f'{layer_label}_attn_agg')
            dot.edge(f'{layer_label}_attn_ellipsis', f'{layer_label}_attn_agg')
            for head in range(24, 32):
                dot.edge(f'{layer_label}_attn_head{head}', f'{layer_label}_attn_agg')
            
            # Attention output projection
            dot.node(f'{layer_label}_attn_out', f'Layer {layer_id} Attention Output Proj\\nGPU: Pipeline {stage} (TP: GPU {stage*128}-{stage*128+127})\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                     shape='rectangle', fillcolor='lightgreen')
            dot.edge(f'{layer_label}_attn_agg', f'{layer_label}_attn_out')
            
            # Tensor parallelism communication for attention output
            dot.node(f'{layer_label}_tp_attn_comm', f'Layer {layer_id} TP Attention All-Reduce\\nGPU: Pipeline {stage} (TP pairs)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                     shape='ellipse', fillcolor='orange')
            dot.edge(f'{layer_label}_attn_out', f'{layer_label}_tp_attn_comm')
            
            # Expert routing and load balancing
            dot.node(f'{layer_label}_expert_router', f'Layer {layer_id} Expert Router (Top-k)\\nGPU: Pipeline {stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, experts={experts_per_layer}]',
                     shape='parallelogram', fillcolor='yellow')
            dot.edge(f'{layer_label}_tp_attn_comm', f'{layer_label}_expert_router')
            
            # Create sample experts (first 4, middle 2, last 4 for readability)
            expert_samples = [0, 1, 2, 3, 31, 32, 60, 61, 62, 63]
            
            for expert in expert_samples:
                gpu_base = stage * 128 + expert * 2  # 2 GPUs per expert for TP
                
                # Expert computation
                dot.node(f'{layer_label}_expert{expert}', f'Layer {layer_id} Expert {expert}\\nGPU: {gpu_base}-{gpu_base+1} (TP pair)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                         shape='rectangle', fillcolor='lightgreen', fontsize='10')
                
                # Expert routing with dashed line (gate selection)
                dot.edge(f'{layer_label}_expert_router', f'{layer_label}_expert{expert}', style='dashed')
                
                # Expert MLP decomposition
                dot.node(f'{layer_label}_expert{expert}_mlp1', f'Expert {expert} MLP Layer 1\\nGPU: {gpu_base}-{gpu_base+1} (TP: Column Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn={ffn_hidden_size//2}]',
                         shape='rectangle', fillcolor='lightgreen', fontsize='10')
                dot.edge(f'{layer_label}_expert{expert}', f'{layer_label}_expert{expert}_mlp1')
                
                # Tensor communication within expert
                dot.node(f'{layer_label}_expert{expert}_tp1', f'Expert {expert} TP All-Gather\\nGPU: {gpu_base}-{gpu_base+1}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn={ffn_hidden_size//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn={ffn_hidden_size}]',
                         shape='ellipse', fillcolor='orange', fontsize='10')
                dot.edge(f'{layer_label}_expert{expert}_mlp1', f'{layer_label}_expert{expert}_tp1')
                
                # Activation
                dot.node(f'{layer_label}_expert{expert}_act', f'Expert {expert} GELU Activation\\nGPU: {gpu_base}-{gpu_base+1}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn={ffn_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn={ffn_hidden_size}]',
                         shape='rectangle', fillcolor='lightgreen', fontsize='10')
                dot.edge(f'{layer_label}_expert{expert}_tp1', f'{layer_label}_expert{expert}_act')
                
                # Second MLP layer
                dot.node(f'{layer_label}_expert{expert}_mlp2', f'Expert {expert} MLP Layer 2\\nGPU: {gpu_base}-{gpu_base+1} (TP: Row Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn={ffn_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                         shape='rectangle', fillcolor='lightgreen', fontsize='10')
                dot.edge(f'{layer_label}_expert{expert}_act', f'{layer_label}_expert{expert}_mlp2')
                
                # Tensor communication for output
                dot.node(f'{layer_label}_expert{expert}_tp2', f'Expert {expert} TP All-Reduce\\nGPU: {gpu_base}-{gpu_base+1}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                         shape='ellipse', fillcolor='orange', fontsize='10')
                dot.edge(f'{layer_label}_expert{expert}_mlp2', f'{layer_label}_expert{expert}_tp2')
            
            # Add ellipsis for remaining experts
            dot.node(f'{layer_label}_expert_ellipsis', f'... 54 more experts ...\\nGPU: Pipeline {stage} (Experts 4-59)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                     shape='rectangle', fillcolor='lightgreen', fontsize='10', style='dashed')
            dot.edge(f'{layer_label}_expert_router', f'{layer_label}_expert_ellipsis', style='dashed')
            
            # Expert aggregation
            dot.node(f'{layer_label}_expert_agg', f'Layer {layer_id} Expert Aggregation\\nGPU: Pipeline {stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, experts={experts_per_layer}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                     shape='parallelogram', fillcolor='yellow')
            
            for expert in expert_samples:
                dot.edge(f'{layer_label}_expert{expert}_tp2', f'{layer_label}_expert_agg')
            dot.edge(f'{layer_label}_expert_ellipsis', f'{layer_label}_expert_agg')
            
            # Layer output
            dot.node(f'{layer_label}_output', f'Layer {layer_id} Output\\nGPU: Pipeline {stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                     shape='rectangle', fillcolor='lightgreen')
            dot.edge(f'{layer_label}_expert_agg', f'{layer_label}_output')
            
            if layer == 0:
                dot.edge(f'{layer_label}_tp_attn_comm', f'{layer_label}_output', style='invis')
        
        # Stage output
        dot.node(f'{stage_label}_output', f'Stage {stage} Output\\nGPU: Pipeline {stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                 shape='parallelogram', fillcolor='yellow')
        dot.edge(f'{stage_label}_layer1_output', f'{stage_label}_output')
    
    # Final output
    dot.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
             shape='ellipse', fillcolor='lightblue')
    dot.edge('stage7_output', 'output')
    
    return dot

if __name__ == '__main__':
    # Generate the complete DAG
    dag = create_complete_moe_dag()
    
    # Save DOT file
    dag.save('../outputs/2025-12-05-14-17-12/complete_moe_deployment_dag.dot')
    
    # Save SVG image
    dag.render('../outputs/2025-12-05-14-17-12/complete_moe_deployment_dag', format='svg', cleanup=True)
    
    print("Complete MoE deployment DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-05-14-17-12/complete_moe_deployment_dag.dot")
    print(f"SVG image: ../outputs/2025-12-05-14-17-12/complete_moe_deployment_dag.svg")