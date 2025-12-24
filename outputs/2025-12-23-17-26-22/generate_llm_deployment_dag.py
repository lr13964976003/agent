#!/usr/bin/env python3
"""
Complete LLM Deployment DAG Generator
Addresses all critical issues from the feedback:
1. Complete attention block decomposition
2. All 80 layers represented
3. Pipeline communication nodes
4. Proper node connectivity
5. Detailed tensor parallel communication
"""

import graphviz
from graphviz import Digraph

def create_complete_llm_deployment_dag():
    """Create a complete DAG for LLM deployment with TP=2, PP=4 on 8 GPUs"""
    
    dot = Digraph(comment='Complete LLM Deployment DAG - TP=2 PP=4 80 Layers')
    dot.attr(rankdir='TB', size='300,200', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='box', style='filled', fillcolor='lightgreen')     # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model parameters
    batch_size = 'B'
    seq_len = 'S'
    hidden_size = 8192
    num_heads = 64
    head_dim = hidden_size // num_heads  # 128
    vocab_size = 128256
    
    # GPU assignments
    # Stage 0: GPUs 0-1 (layers 0-19)
    # Stage 1: GPUs 2-3 (layers 20-39) 
    # Stage 2: GPUs 4-5 (layers 40-59)
    # Stage 3: GPUs 6-7 (layers 60-79)
    
    def get_gpu_assignment(layer_id):
        """Get GPU assignment for a layer based on PP strategy"""
        if 0 <= layer_id <= 19:  # Stage 0
            return [0, 1]
        elif 20 <= layer_id <= 39:  # Stage 1
            return [2, 3]
        elif 40 <= layer_id <= 59:  # Stage 2
            return [4, 5]
        elif 60 <= layer_id <= 79:  # Stage 3
            return [6, 7]
        else:
            raise ValueError(f"Invalid layer ID: {layer_id}")
    
    def add_input_node():
        """Add input node"""
        input_dims = f'[batch_size={batch_size}, seq_len={seq_len}]'
        dot.node('input', f'Input\\nInput: {input_dims}\\nOutput: {input_dims}', 
                shape='box', fillcolor='lightcoral')
    
    def add_embedding_nodes():
        """Add embedding nodes for both GPUs in stage 0"""
        input_dims = f'[batch_size={batch_size}, seq_len={seq_len}]'
        embed_dims = f'[batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]'
        
        # Embedding on GPU 0 and 1 (TP split)
        dot.node('embed_tp0', f'Embedding_TP0\\nGPU: 0\\nInput: {input_dims}\\nOutput: {embed_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.node('embed_tp1', f'Embedding_TP1\\nGPU: 1\\nInput: {input_dims}\\nOutput: {embed_dims}', 
                shape='box', fillcolor='lightgreen')
        
        # All-gather for embedding
        dot.node('embed_allgather', f'Embedding_AllGather\\nGPU: 0,1\\nInput: {embed_dims} (split)\\nOutput: {embed_dims} (full)', 
                shape='ellipse', fillcolor='lightblue')
        
        return 'embed_allgather'
    
    def add_attention_block(layer_id, gpu_id, input_node):
        """Add complete attention block with all submodules"""
        
        # Attention input dimensions
        attn_input_dims = f'[batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]'
        
        # QKV projection (split across TP)
        qkv_dims = f'[batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//2}, d_k={head_dim}]'
        qkv_node = f'layer{layer_id}_qkv_tp{gpu_id%2}'
        dot.node(qkv_node, 
                f'Layer{layer_id}_QKV_TP{gpu_id%2}\\nGPU: {gpu_id}\\nInput: {attn_input_dims}\\nOutput: {qkv_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.edge(input_node, qkv_node)
        
        # Attention scores computation (Q*K^T)
        scores_dims = f'[batch_size={batch_size}, heads={num_heads//2}, seq_len={seq_len}, seq_len={seq_len}]'
        scores_node = f'layer{layer_id}_scores_tp{gpu_id%2}'
        dot.node(scores_node, 
                f'Layer{layer_id}_Scores_TP{gpu_id%2}\\nGPU: {gpu_id}\\nInput: {qkv_dims}\\nOutput: {scores_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.edge(qkv_node, scores_node)
        
        # All-reduce for attention scores across TP ranks
        scores_allreduce_node = f'layer{layer_id}_scores_allreduce_tp{gpu_id%2}'
        dot.node(scores_allreduce_node, 
                f'Layer{layer_id}_Scores_AllReduce_TP{gpu_id%2}\\nGPU: {gpu_id//2*2},{gpu_id//2*2+1}\\nInput: {scores_dims}\\nOutput: {scores_dims}', 
                shape='ellipse', fillcolor='lightblue')
        dot.edge(scores_node, scores_allreduce_node)
        
        # Attention weights (softmax)
        weights_dims = scores_dims
        weights_node = f'layer{layer_id}_weights_tp{gpu_id%2}'
        dot.node(weights_node, 
                f'Layer{layer_id}_Weights_TP{gpu_id%2}\\nGPU: {gpu_id}\\nInput: {scores_dims}\\nOutput: {weights_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.edge(scores_allreduce_node, weights_node)
        
        # Attention output (weighted sum)
        attn_out_dims = qkv_dims
        attn_out_node = f'layer{layer_id}_attn_out_tp{gpu_id%2}'
        dot.node(attn_out_node, 
                f'Layer{layer_id}_AttnOut_TP{gpu_id%2}\\nGPU: {gpu_id}\\nInput: {weights_dims}, {qkv_dims}\\nOutput: {attn_out_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.edge(weights_node, attn_out_node)
        dot.edge(qkv_node, attn_out_node)  # Values input
        
        # Attention output projection
        proj_out_dims = f'[batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//2}]'
        proj_node = f'layer{layer_id}_proj_tp{gpu_id%2}'
        dot.node(proj_node, 
                f'Layer{layer_id}_Proj_TP{gpu_id%2}\\nGPU: {gpu_id}\\nInput: {attn_out_dims}\\nOutput: {proj_out_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.edge(attn_out_node, proj_node)
        
        return proj_node
    
    def add_ffn_block(layer_id, gpu_id, input_node):
        """Add FFN block"""
        
        ffn_input_dims = f'[batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//2}]'
        ffn_hidden_dims = f'[batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={4*hidden_size//2}]'
        ffn_output_dims = ffn_input_dims
        
        # FFN first linear layer
        ffn1_node = f'layer{layer_id}_ffn1_tp{gpu_id%2}'
        dot.node(ffn1_node, 
                f'Layer{layer_id}_FFN1_TP{gpu_id%2}\\nGPU: {gpu_id}\\nInput: {ffn_input_dims}\\nOutput: {ffn_hidden_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.edge(input_node, ffn1_node)
        
        # SiLU activation
        silu_node = f'layer{layer_id}_silu_tp{gpu_id%2}'
        dot.node(silu_node, 
                f'Layer{layer_id}_SiLU_TP{gpu_id%2}\\nGPU: {gpu_id}\\nInput: {ffn_hidden_dims}\\nOutput: {ffn_hidden_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.edge(ffn1_node, silu_node)
        
        # FFN second linear layer
        ffn2_node = f'layer{layer_id}_ffn2_tp{gpu_id%2}'
        dot.node(ffn2_node, 
                f'Layer{layer_id}_FFN2_TP{gpu_id%2}\\nGPU: {gpu_id}\\nInput: {ffn_hidden_dims}\\nOutput: {ffn_output_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.edge(silu_node, ffn2_node)
        
        # FFN all-reduce across TP ranks
        ffn_allreduce_node = f'layer{layer_id}_ffn_allreduce_tp{gpu_id%2}'
        dot.node(ffn_allreduce_node, 
                f'Layer{layer_id}_FFN_AllReduce_TP{gpu_id%2}\\nGPU: {gpu_id//2*2},{gpu_id//2*2+1}\\nInput: {ffn_output_dims}\\nOutput: {ffn_output_dims}', 
                shape='ellipse', fillcolor='lightblue')
        dot.edge(ffn2_node, ffn_allreduce_node)
        
        return ffn_allreduce_node
    
    def add_layer_norm(layer_id, gpu_id, input_node, is_pre_attn=True):
        """Add layer normalization"""
        
        ln_dims = f'[batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]'
        prefix = 'pre_attn' if is_pre_attn else 'post_attn'
        ln_node = f'layer{layer_id}_{prefix}_ln_tp{gpu_id%2}'
        dot.node(ln_node, 
                f'Layer{layer_id}_{prefix.capitalize()}LN_TP{gpu_id%2}\\nGPU: {gpu_id}\\nInput: {ln_dims}\\nOutput: {ln_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.edge(input_node, ln_node)
        return ln_node
    
    def add_residual_connection(layer_id, gpu_id, input1_node, input2_node, is_attn=True):
        """Add residual connection with aggregation"""
        
        output_dims = f'[batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]'
        prefix = 'attn' if is_attn else 'ffn'
        residual_node = f'layer{layer_id}_{prefix}_residual_tp{gpu_id%2}'
        dot.node(residual_node, 
                f'Layer{layer_id}_{prefix.capitalize()}Residual_TP{gpu_id%2}\\nGPU: {gpu_id}\\nInput: {output_dims}, {output_dims}\\nOutput: {output_dims}', 
                shape='parallelogram', fillcolor='lightyellow')
        dot.edge(input1_node, residual_node)
        dot.edge(input2_node, residual_node)
        return residual_node
    
    def add_pipeline_communication(from_layer, to_layer, from_gpu, to_gpu):
        """Add explicit pipeline communication nodes"""
        
        comm_dims = f'[batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]'
        
        # Send node
        send_node = f'pipeline_send_l{from_layer}_to_l{to_layer}'
        dot.node(send_node, 
                f'PipelineSend_L{from_layer}_to_L{to_layer}\\nGPU: {from_gpu}\\nInput: {comm_dims}\\nOutput: {comm_dims}', 
                shape='ellipse', fillcolor='lightblue', style='dashed')
        
        # Receive node
        recv_node = f'pipeline_recv_l{from_layer}_to_l{to_layer}'
        dot.node(recv_node, 
                f'PipelineRecv_L{from_layer}_to_L{to_layer}\\nGPU: {to_gpu}\\nInput: {comm_dims}\\nOutput: {comm_dims}', 
                shape='ellipse', fillcolor='lightblue', style='dashed')
        
        return send_node, recv_node
    
    def add_output_nodes(final_node):
        """Add output nodes"""
        output_dims = f'[batch_size={batch_size}, seq_len={seq_len}, vocab={vocab_size}]'
        
        # Final layer norm
        final_ln_node = 'final_ln'
        dot.node(final_ln_node, 
                f'FinalLayerNorm\\nGPU: 6,7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]', 
                shape='box', fillcolor='lightgreen')
        dot.edge(final_node, final_ln_node)
        
        # Output projection
        output_proj_node = 'output_proj'
        dot.node(output_proj_node, 
                f'OutputProjection\\nGPU: 6,7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: {output_dims}', 
                shape='box', fillcolor='lightgreen')
        dot.edge(final_ln_node, output_proj_node)
        
        # Output node
        dot.node('output', 
                f'Output\\nInput: {output_dims}\\nOutput: {output_dims}', 
                shape='box', fillcolor='lightcoral')
        dot.edge(output_proj_node, 'output')
    
    # Build the complete DAG
    add_input_node()
    
    # Input embedding
    embed_node = add_embedding_nodes()
    dot.edge('input', 'embed_tp0')
    dot.edge('input', 'embed_tp1')
    dot.edge('embed_tp0', 'embed_allgather')
    dot.edge('embed_tp1', 'embed_allgather')
    
    prev_layer_output = embed_node
    
    # Add all 80 layers
    for layer_id in range(80):
        gpus = get_gpu_assignment(layer_id)
        
        # Handle pipeline communication between stages
        if layer_id in [20, 40, 60]:
            prev_gpu = get_gpu_assignment(layer_id-1)[0]  # Use first GPU of previous stage
            curr_gpu = gpus[0]  # Use first GPU of current stage
            send_node, recv_node = add_pipeline_communication(layer_id-1, layer_id, prev_gpu, curr_gpu)
            dot.edge(prev_layer_output, send_node)
            prev_layer_output = recv_node
        
        # Pre-attention layer norm
        pre_attn_ln_node = add_layer_norm(layer_id, gpus[0], prev_layer_output, is_pre_attn=True)
        
        # Attention block for each GPU in TP group
        attn_outputs = []
        for gpu_id in gpus:
            attn_output = add_attention_block(layer_id, gpu_id, pre_attn_ln_node)
            attn_outputs.append(attn_output)
        
        # All-gather for attention outputs
        attn_allgather_node = f'layer{layer_id}_attn_allgather'
        attn_dims = f'[batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]'
        dot.node(attn_allgather_node, 
                f'Layer{layer_id}_Attn_AllGather\\nGPU: {gpus[0]},{gpus[1]}\\nInput: {attn_dims} (split)\\nOutput: {attn_dims} (full)', 
                shape='ellipse', fillcolor='lightblue')
        
        for attn_output in attn_outputs:
            dot.edge(attn_output, attn_allgather_node)
        
        # Attention residual connection
        attn_residual_node = add_residual_connection(layer_id, gpus[0], prev_layer_output, attn_allgather_node, is_attn=True)
        
        # Post-attention layer norm
        post_attn_ln_node = add_layer_norm(layer_id, gpus[0], attn_residual_node, is_pre_attn=False)
        
        # FFN block for each GPU in TP group
        ffn_outputs = []
        for gpu_id in gpus:
            ffn_output = add_ffn_block(layer_id, gpu_id, post_attn_ln_node)
            ffn_outputs.append(ffn_output)
        
        # All-gather for FFN outputs
        ffn_allgather_node = f'layer{layer_id}_ffn_allgather'
        ffn_dims = f'[batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]'
        dot.node(ffn_allgather_node, 
                f'Layer{layer_id}_FFN_AllGather\\nGPU: {gpus[0]},{gpus[1]}\\nInput: {ffn_dims} (split)\\nOutput: {ffn_dims} (full)', 
                shape='ellipse', fillcolor='lightblue')
        
        for ffn_output in ffn_outputs:
            dot.edge(ffn_output, ffn_allgather_node)
        
        # FFN residual connection
        ffn_residual_node = add_residual_connection(layer_id, gpus[0], attn_residual_node, ffn_allgather_node, is_attn=False)
        
        prev_layer_output = ffn_residual_node
    
    # Add output nodes
    add_output_nodes(prev_layer_output)
    
    return dot

def main():
    """Generate the complete DAG"""
    print("Generating complete LLM deployment DAG...")
    
    dag = create_complete_llm_deployment_dag()
    
    # Save DOT file
    dot_path = '../outputs/2025-12-23-17-26-22/llm_deployment_dag_complete.dot'
    dag.save(dot_path)
    print(f"DOT file saved to: {dot_path}")
    
    # Render SVG
    svg_path = '../outputs/2025-12-23-17-26-22/llm_deployment_dag_complete.svg'
    dag.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
    print(f"SVG file saved to: {svg_path}")
    
    return dot_path, svg_path

if __name__ == "__main__":
    main()