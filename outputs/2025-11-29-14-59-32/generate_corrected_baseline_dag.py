#!/usr/bin/env python3

import graphviz

def create_corrected_baseline_tensor_pipeline_dag():
    """
    Create corrected baseline tensor pipeline DAG with proper connectivity
    Fixes: AllGather nodes properly connected, complete tensor parallelism flow
    """
    dot = graphviz.Digraph('baseline_tensor_pipeline_dag', 
                          comment='Corrected Baseline Tensor Pipeline Parallelism DAG')
    
    # Set graph attributes
    dot.attr(rankdir='TB', bgcolor='white', fontname='Arial')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Node shapes: rectangle=compute, ellipse=communication, parallelogram=routing/aggregation
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Define model specifications
    model_specs = {
        'hidden_dim': 4096,
        'num_heads': 32,
        'head_dim': 128,
        'mlp_hidden': 16384,
        'batch_size': 128,
        'seq_len': 10000,
        'tensor_parallel': 8,
        'pipeline_parallel': 2
    }
    
    # Input node
    dot.node('input', f'Input\\nBatch: {model_specs["batch_size"]}\\nSeq: {model_specs["seq_len"]}\\nHidden: {model_specs["hidden_dim"]}', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Create nodes for all 16 layers with proper tensor parallelism
    for layer in range(16):
        # Determine pipeline stage and GPU base
        pipeline_stage = 0 if layer < 8 else 1
        gpu_base = 0 if layer < 8 else 8
        
        # Layer input distribution node
        dot.node(f'layer_{layer}_input_dist', f'Layer {layer}\\nInput Distribution', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # QKV Projection (tensor parallel across 8 GPUs)
        for tp_rank in range(8):
            gpu_id = gpu_base + tp_rank
            dot.node(f'layer_{layer}_qkv_gpu{gpu_id}', 
                    f'Layer {layer} QKV Proj\\nGPU {gpu_id}\\nTP Rank {tp_rank}\\n[4096→12288]', 
                    fillcolor='lightblue')
        
        # AllGather for QKV
        dot.node(f'layer_{layer}_allgather_qkv', f'Layer {layer}\\nAllGather QKV\\nTP=8', 
                shape='ellipse', fillcolor='orange')
        
        # Attention computation
        dot.node(f'layer_{layer}_attention', f'Layer {layer}\\nAttention\\n32×128 dim\\nSoftmax', 
                fillcolor='lightblue')
        
        # Attention output projection (tensor parallel)
        for tp_rank in range(8):
            gpu_id = gpu_base + tp_rank
            dot.node(f'layer_{layer}_attn_out_gpu{gpu_id}', 
                    f'Layer {layer} Attn Out\\nGPU {gpu_id}\\nTP Rank {tp_rank}\\n[4096→4096]', 
                    fillcolor='lightblue')
        
        # AllReduce for attention output
        dot.node(f'layer_{layer}_allreduce_attn', f'Layer {layer}\\nAllReduce Attn\\nTP=8', 
                shape='ellipse', fillcolor='orange')
        
        # Residual connection addition
        dot.node(f'layer_{layer}_residual1', f'Layer {layer}\\nResidual Add', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # Layer normalization 1
        dot.node(f'layer_{layer}_ln1', f'Layer {layer}\\nLayerNorm 1\\nγ,β: 4096', 
                fillcolor='lightblue')
        
        # MLP FC1 (tensor parallel)
        for tp_rank in range(8):
            gpu_id = gpu_base + tp_rank
            dot.node(f'layer_{layer}_mlp_fc1_gpu{gpu_id}', 
                    f'Layer {layer} MLP FC1\\nGPU {gpu_id}\\nTP Rank {tp_rank}\\n[4096→16384]', 
                    fillcolor='lightblue')
        
        # AllGather for MLP
        dot.node(f'layer_{layer}_allgather_mlp', f'Layer {layer}\\nAllGather MLP\\nTP=8', 
                shape='ellipse', fillcolor='orange')
        
        # GELU activation
        dot.node(f'layer_{layer}_gelu', f'Layer {layer}\\nGELU Activation', 
                fillcolor='lightblue')
        
        # MLP FC2 (tensor parallel)
        for tp_rank in range(8):
            gpu_id = gpu_base + tp_rank
            dot.node(f'layer_{layer}_mlp_fc2_gpu{gpu_id}', 
                    f'Layer {layer} MLP FC2\\nGPU {gpu_id}\\nTP Rank {tp_rank}\\n[16384→4096]', 
                    fillcolor='lightblue')
        
        # AllReduce for MLP output
        dot.node(f'layer_{layer}_allreduce_mlp', f'Layer {layer}\\nAllReduce MLP\\nTP=8', 
                shape='ellipse', fillcolor='orange')
        
        # Residual connection addition 2
        dot.node(f'layer_{layer}_residual2', f'Layer {layer}\\nResidual Add', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # Layer normalization 2
        dot.node(f'layer_{layer}_ln2', f'Layer {layer}\\nLayerNorm 2\\nγ,β: 4096', 
                fillcolor='lightblue')
        
        # Layer output distribution for next layer
        if layer < 15:
            dot.node(f'layer_{layer}_output_dist', f'Layer {layer}\\nOutput Distribution', 
                    shape='parallelogram', fillcolor='lightyellow')
    
    # Output node
    dot.node('output', f'Output\\nBatch: {model_specs["batch_size"]}\\nSeq: {model_specs["seq_len"]}\\nHidden: {model_specs["hidden_dim"]}', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Connect the nodes with proper data flow
    
    # Input to first layer
    dot.edge('input', 'layer_0_input_dist')
    
    for layer in range(16):
        # Connect input distribution
        if layer == 0:
            dot.edge('input', 'layer_0_input_dist')
        else:
            dot.edge(f'layer_{layer-1}_output_dist', f'layer_{layer}_input_dist')
        
        # Connect to QKV projections
        for tp_rank in range(8):
            gpu_id = (0 if layer < 8 else 8) + tp_rank
            dot.edge(f'layer_{layer}_input_dist', f'layer_{layer}_qkv_gpu{gpu_id}')
        
        # Connect QKV to AllGather
        for tp_rank in range(8):
            gpu_id = (0 if layer < 8 else 8) + tp_rank
            dot.edge(f'layer_{layer}_qkv_gpu{gpu_id}', f'layer_{layer}_allgather_qkv')
        
        # Connect AllGather to Attention
        dot.edge(f'layer_{layer}_allgather_qkv', f'layer_{layer}_attention')
        
        # Connect Attention to output projections
        dot.edge(f'layer_{layer}_attention', f'layer_{layer}_attn_out_gpu{gpu_id}')
        
        # Connect attention outputs to AllReduce
        for tp_rank in range(8):
            gpu_id = (0 if layer < 8 else 8) + tp_rank
            dot.edge(f'layer_{layer}_attn_out_gpu{gpu_id}', f'layer_{layer}_allreduce_attn')
        
        # Connect AllReduce to residual
        dot.edge(f'layer_{layer}_allreduce_attn', f'layer_{layer}_residual1')
        
        # Connect residual to input distribution for skip connection
        dot.edge(f'layer_{layer}_input_dist', f'layer_{layer}_residual1')
        
        # Connect residual to layer norm
        dot.edge(f'layer_{layer}_residual1', f'layer_{layer}_ln1')
        
        # Connect layer norm to MLP FC1
        for tp_rank in range(8):
            gpu_id = (0 if layer < 8 else 8) + tp_rank
            dot.edge(f'layer_{layer}_ln1', f'layer_{layer}_mlp_fc1_gpu{gpu_id}')
        
        # Connect MLP FC1 to AllGather
        for tp_rank in range(8):
            gpu_id = (0 if layer < 8 else 8) + tp_rank
            dot.edge(f'layer_{layer}_mlp_fc1_gpu{gpu_id}', f'layer_{layer}_allgather_mlp')
        
        # Connect AllGather to GELU
        dot.edge(f'layer_{layer}_allgather_mlp', f'layer_{layer}_gelu')
        
        # Connect GELU to MLP FC2
        dot.edge(f'layer_{layer}_gelu', f'layer_{layer}_mlp_fc2_gpu{gpu_id}')
        
        # Connect MLP FC2 to AllReduce
        for tp_rank in range(8):
            gpu_id = (0 if layer < 8 else 8) + tp_rank
            dot.edge(f'layer_{layer}_mlp_fc2_gpu{gpu_id}', f'layer_{layer}_allreduce_mlp')
        
        # Connect AllReduce to residual
        dot.edge(f'layer_{layer}_allreduce_mlp', f'layer_{layer}_residual2')
        
        # Connect first residual to second residual
        dot.edge(f'layer_{layer}_residual1', f'layer_{layer}_residual2')
        
        # Connect residual to layer norm 2
        dot.edge(f'layer_{layer}_residual2', f'layer_{layer}_ln2')
        
        # Connect to output distribution (except last layer)
        if layer < 15:
            dot.edge(f'layer_{layer}_ln2', f'layer_{layer}_output_dist')
        else:
            # Last layer connects to output
            dot.edge(f'layer_{layer}_ln2', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_corrected_baseline_tensor_pipeline_dag()
    
    # Save DOT file
    dot_file_path = '../outputs/2025-11-29-14-59-32/corrected_baseline_tensor_pipeline_dag.dot'
    dag.save(dot_file_path)
    
    # Save SVG image
    svg_file_path = '../outputs/2025-11-29-14-59-32/corrected_baseline_tensor_pipeline_dag.svg'
    dag.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Corrected baseline DAG generated:")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")