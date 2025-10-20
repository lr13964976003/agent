#!/usr/bin/env python3
"""
Generate complete DAG for Ring Attention + Sequence Parallelism deployment
This DAG represents the full transformer model with 4 layers across 16 GPUs
using sequence parallelism (625 tokens per device) and ring attention.
"""

import graphviz
import os

def create_ra_sp_dag():
    """Create the complete DAG for Ring Attention + Sequence Parallelism"""
    
    # Create directed graph
    dag = graphviz.Digraph('RA_SP_Transformer', 
                          filename='ra_sp_transformer_complete.dot',
                          format='svg')
    
    # Graph attributes
    dag.attr(rankdir='TB', splines='ortho', compound='true')
    
    # Node styles
    dag.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Define colors for different operations
    colors = {
        'input': 'lightgreen',
        'projection': 'lightyellow',
        'attention': 'lightcoral',
        'mlp': 'lightsteelblue',
        'residual': 'lightpink',
        'norm': 'lightgray',
        'communication': 'gold',
        'aggregation': 'orange',
        'split': 'wheat'
    }
    
    # Shared dimensions
    B = 1024  # batch_size
    L_total = 10000  # total sequence length
    L_local = 625  # sequence length per device
    d_model = 8192
    heads = 16
    d_k = 512
    d_ff = 32768
    
    # Create subgraphs for each device
    devices = list(range(16))
    
    # Global input node
    dag.node('global_input', 
             f'Total Input\\nInput: [batch_size={B}, seq_len={L_total}, d_model={d_model}]\\nAll GPUs',
             shape='ellipse', fillcolor=colors['input'], style='filled')
    
    # Split operation
    dag.node('split_tokens', 
             f'Split Sequence\\nInput: [batch_size={B}, seq_len={L_total}, d_model={d_model}]\\nOutput: 16×[batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nAll GPUs',
             shape='parallelogram', fillcolor=colors['split'], style='filled')
    
    dag.edge('global_input', 'split_tokens')
    
    # Process each layer
    for layer in range(4):
        # Create layer subgraph
        with dag.subgraph(name=f'cluster_layer_{layer}') as layer_sg:
            layer_sg.attr(label=f'Layer {layer}')
            
            # Process each device
            for device in devices:
                device_prefix = f'd{device}_l{layer}'
                
                # Layer Input (after split)
                if layer == 0:
                    dag.node(f'{device_prefix}_input',
                             f'Layer{layer} Input D{device}\\nInput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nGPU {device}',
                             fillcolor=colors['input'])
                    dag.edge('split_tokens', f'{device_prefix}_input')
                else:
                    dag.node(f'{device_prefix}_input',
                             f'Layer{layer} Input D{device}\\nInput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nGPU {device}',
                             fillcolor=colors['input'])
                    dag.edge(f'd{device}_l{layer-1}_output', f'{device_prefix}_input')
                
                # Layer Norm 1
                dag.node(f'{device_prefix}_norm1',
                         f'Layer Norm 1 D{device}\\nInput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nOutput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nGPU {device}',
                         fillcolor=colors['norm'])
                dag.edge(f'{device_prefix}_input', f'{device_prefix}_norm1')
                
                # QKV Projections (replicated across all devices)
                dag.node(f'{device_prefix}_q_proj',
                         f'Q Projection D{device}\\nInput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nOutput: [batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]\\nGPU {device}',
                         fillcolor=colors['projection'])
                dag.edge(f'{device_prefix}_norm1', f'{device_prefix}_q_proj')
                
                dag.node(f'{device_prefix}_k_proj',
                         f'K Projection D{device}\\nInput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nOutput: [batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]\\nGPU {device}',
                         fillcolor=colors['projection'])
                dag.edge(f'{device_prefix}_norm1', f'{device_prefix}_k_proj')
                
                dag.node(f'{device_prefix}_v_proj',
                         f'V Projection D{device}\\nInput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nOutput: [batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]\\nGPU {device}',
                         fillcolor=colors['projection'])
                dag.edge(f'{device_prefix}_norm1', f'{device_prefix}_v_proj')
                
                # Ring Attention Process (16 stages)
                for stage in range(16):
                    # Ring communication nodes
                    src_device = (device - stage) % 16
                    
                    dag.node(f'{device_prefix}_recv_kv_{stage}',
                             f'Receive KV Stage{stage} D{device}\\nFrom: GPU {(device-1)%16}\\nData: [batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]×2\\nGPU {device}',
                             shape='ellipse', fillcolor=colors['communication'])
                    
                    dag.node(f'{device_prefix}_attn_stage_{stage}',
                             f'Local Attention Stage{stage} D{device}\\nQ: [batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]\\nK,V: [batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]\\nOutput: [batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]\\nGPU {device}',
                             fillcolor=colors['attention'])
                    
                    dag.node(f'{device_prefix}_send_kv_{stage}',
                             f'Send KV Stage{stage} D{device}\\nTo: GPU {(device+1)%16}\\nData: [batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]×2\\nGPU {device}',
                             shape='ellipse', fillcolor=colors['communication'])
                    
                    # Connect attention stages
                    if stage == 0:
                        dag.edge(f'{device_prefix}_q_proj', f'{device_prefix}_attn_stage_{stage}')
                        dag.edge(f'{device_prefix}_k_proj', f'{device_prefix}_attn_stage_{stage}')
                        dag.edge(f'{device_prefix}_v_proj', f'{device_prefix}_attn_stage_{stage}')
                    else:
                        dag.edge(f'{device_prefix}_recv_kv_{stage}', f'{device_prefix}_attn_stage_{stage}')
                        dag.edge(f'{device_prefix}_attn_stage_{stage-1}', f'{device_prefix}_attn_stage_{stage}')
                    
                    dag.edge(f'{device_prefix}_attn_stage_{stage}', f'{device_prefix}_send_kv_{stage}')
                
                # Attention output projection
                dag.node(f'{device_prefix}_attn_out_proj',
                         f'Attention Output Projection D{device}\\nInput: [batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]\\nOutput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nGPU {device}',
                         fillcolor=colors['projection'])
                dag.edge(f'{device_prefix}_attn_stage_15', f'{device_prefix}_attn_out_proj')
                
                # Residual connection 1
                dag.node(f'{device_prefix}_residual1',
                         f'Residual Add 1 D{device}\\nInput1: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nInput2: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nOutput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nGPU {device}',
                         fillcolor=colors['residual'])
                dag.edge(f'{device_prefix}_input', f'{device_prefix}_residual1')
                dag.edge(f'{device_prefix}_attn_out_proj', f'{device_prefix}_residual1')
                
                # Layer Norm 2
                dag.node(f'{device_prefix}_norm2',
                         f'Layer Norm 2 D{device}\\nInput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nOutput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nGPU {device}',
                         fillcolor=colors['norm'])
                dag.edge(f'{device_prefix}_residual1', f'{device_prefix}_norm2')
                
                # MLP projections
                dag.node(f'{device_prefix}_mlp_gate_up',
                         f'MLP Gate+Up Projection D{device}\\nInput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nOutput: [batch_size={B}, seq_len={L_local}, d_ff={d_ff}]\\nGPU {device}',
                         fillcolor=colors['mlp'])
                dag.edge(f'{device_prefix}_norm2', f'{device_prefix}_mlp_gate_up')
                
                dag.node(f'{device_prefix}_mlp_act',
                         f'MLP Activation D{device}\\nInput: [batch_size={B}, seq_len={L_local}, d_ff={d_ff}]\\nOutput: [batch_size={B}, seq_len={L_local}, d_ff={d_ff}]\\nGPU {device}',
                         fillcolor=colors['mlp'])
                dag.edge(f'{device_prefix}_mlp_gate_up', f'{device_prefix}_mlp_act')
                
                dag.node(f'{device_prefix}_mlp_down',
                         f'MLP Down Projection D{device}\\nInput: [batch_size={B}, seq_len={L_local}, d_ff={d_ff}]\\nOutput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nGPU {device}',
                         fillcolor=colors['mlp'])
                dag.edge(f'{device_prefix}_mlp_act', f'{device_prefix}_mlp_down')
                
                # Residual connection 2
                dag.node(f'{device_prefix}_output',
                         f'Layer{layer} Output D{device}\\nInput1: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nInput2: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nOutput: [batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nGPU {device}',
                         fillcolor=colors['residual'])
                dag.edge(f'{device_prefix}_residual1', f'{device_prefix}_output')
                dag.edge(f'{device_prefix}_mlp_down', f'{device_prefix}_output')
    
    # Global aggregation after all layers
    dag.node('aggregate_tokens',
             f'Aggregate Sequence\\nInput: 16×[batch_size={B}, seq_len={L_local}, d_model={d_model}]\\nOutput: [batch_size={B}, seq_len={L_total}, d_model={d_model}]\\nAll GPUs',
             shape='parallelogram', fillcolor=colors['aggregation'], style='filled')
    
    # Connect final outputs to aggregation
    for device in devices:
        dag.edge(f'd{device}_l3_output', 'aggregate_tokens')
    
    # Global output
    dag.node('global_output',
             f'Total Output\\nInput: [batch_size={B}, seq_len={L_total}, d_model={d_model}]\\nOutput: [batch_size={B}, seq_len={L_total}, d_model={d_model}]\\nAll GPUs',
             shape='ellipse', fillcolor=colors['input'], style='filled')
    
    dag.edge('aggregate_tokens', 'global_output')
    
    return dag

def create_attention_detail_dag():
    """Create detailed attention computation DAG for one device"""
    dag = graphviz.Digraph('RA_Attention_Detail', 
                          filename='ra_attention_detail.dot',
                          format='svg')
    
    dag.attr(rankdir='LR', splines='ortho')
    dag.attr('node', shape='rectangle', style='filled')
    
    # Colors
    colors = {
        'input': 'lightgreen',
        'compute': 'lightcoral',
        'comm': 'gold',
        'accum': 'orange'
    }
    
    B = 1024
    L_local = 625
    heads = 16
    d_k = 512
    
    device = 0  # Focus on device 0 for detail
    
    # Input QKV
    dag.node('q_input', f'Q Local\\n[batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]',
             fillcolor=colors['input'])
    dag.node('k_input', f'K Local\\n[batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]',
             fillcolor=colors['input'])
    dag.node('v_input', f'V Local\\n[batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]',
             fillcolor=colors['input'])
    
    # Ring stages
    for stage in range(16):
        # Communication
        dag.node(f'recv_kv_{stage}', 
                 f'Recv K,V Stage {stage}\\nFrom GPU {(device-1-stage)%16}',
                 shape='ellipse', fillcolor=colors['comm'])
        
        # Attention computation
        dag.node(f'attn_stage_{stage}',
                 f'Attention Stage {stage}\\nQ×K_stage→Softmax\\n×V_stage\\n[batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]',
                 fillcolor=colors['compute'])
        
        # Connect stages
        if stage == 0:
            dag.edge('q_input', f'attn_stage_{stage}')
            dag.edge('k_input', f'attn_stage_{stage}')
            dag.edge('v_input', f'attn_stage_{stage}')
        else:
            dag.edge(f'recv_kv_{stage}', f'attn_stage_{stage}')
            dag.edge(f'attn_stage_{stage-1}', f'attn_stage_{stage}')
    
    # Accumulation
    dag.node('accumulate',
             f'Accumulate All Stages\\nSum 16 partial results\\n[batch_size={B}, seq_len={L_local}, heads={heads}, d_k={d_k}]',
             fillcolor=colors['accum'])
    
    dag.edge('attn_stage_15', 'accumulate')
    
    return dag

if __name__ == '__main__':
    # Create output directory
    os.makedirs('./generated_docs/SP', exist_ok=True)
    
    # Generate complete DAG
    ra_sp_dag = create_ra_sp_dag()
    ra_sp_dag.render(directory='./generated_docs/SP', cleanup=True)
    
    # Generate attention detail DAG
    attn_detail = create_attention_detail_dag()
    attn_detail.render(directory='./generated_docs/SP', cleanup=True)
    
    print("DAG files generated:")
    print("- ra_sp_transformer_complete.dot")
    print("- ra_sp_transformer_complete.svg")
    print("- ra_attention_detail.dot")
    print("- ra_attention_detail.svg")