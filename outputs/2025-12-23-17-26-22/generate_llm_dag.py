#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_llm_deployment_dag():
    """
    Create a complete DAG for LLM deployment with TP=2, PP=4 on 8x H100 GPUs
    This includes all 80 layers with proper attention decomposition and pipeline communication
    """
    
    # Create the DAG
    dot = Digraph(comment='LLM Deployment DAG - TP=2, PP=4 on 8x H100 GPUs')
    dot.attr(rankdir='TB', size='100,100', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model parameters
    num_layers = 80
    tp_size = 2
    pp_size = 4
    hidden_size = 8192
    num_heads = 64
    batch_size = 'B'
    seq_len = 'S'
    
    # GPU mapping: 4 pipeline stages, 2 TP ranks each
    # Stage 0: GPUs 0,1 (layers 0-19)
    # Stage 1: GPUs 2,3 (layers 20-39)  
    # Stage 2: GPUs 4,5 (layers 40-59)
    # Stage 3: GPUs 6,7 (layers 60-79)
    
    gpu_mapping = {}
    for stage in range(pp_size):
        for tp_rank in range(tp_size):
            gpu_id = stage * tp_size + tp_rank
            start_layer = stage * (num_layers // pp_size)
            end_layer = (stage + 1) * (num_layers // pp_size) - 1
            gpu_mapping[stage] = {
                'gpus': [stage * tp_size, stage * tp_size + 1],
                'layers': list(range(start_layer, end_layer + 1))
            }
    
    # Input node
    dot.node('input', 
             f'INPUT\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
             shape='rectangle', fillcolor='lightcyan')
    
    # Track previous nodes for connectivity
    prev_nodes = ['input']
    
    # Process each pipeline stage
    for stage in range(pp_size):
        stage_gpus = gpu_mapping[stage]['gpus']
        stage_layers = gpu_mapping[stage]['layers']
        
        # Add pipeline stage boundary marker
        if stage > 0:
            # Pipeline receive from previous stage
            for tp_rank in range(tp_size):
                gpu_id = stage_gpus[tp_rank]
                recv_node = f'pp_stage{stage}_recv_tp{tp_rank}'
                dot.node(recv_node,
                        f'PP Stage {stage} Receive\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='parallelogram', fillcolor='lightcoral')
                
                # Connect to previous stage's last layer
                for prev_tp in range(tp_size):
                    prev_gpu = (stage-1) * tp_size + prev_tp
                    prev_layer = gpu_mapping[stage-1]['layers'][-1]
                    prev_ffn = f'layer{prev_layer}_ffn_allreduce_tp{prev_tp}'
                    dot.edge(prev_ffn, recv_node, label=f'pipeline_send_gpu{prev_gpu}_to_gpu{gpu_id}')
        
        # Process each layer in the stage
        for layer_idx in stage_layers:
            for tp_rank in range(tp_size):
                gpu_id = stage_gpus[tp_rank]
                
                # Layer normalization (RMSNorm)
                norm_node = f'layer{layer_idx}_norm_tp{tp_rank}'
                dot.node(norm_node,
                        f'Layer {layer_idx} RMSNorm\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='rectangle', fillcolor='lightgreen')
                
                # Connect input to norm
                if layer_idx == 0 and stage == 0:
                    # First layer connects to input
                    dot.edge('input', norm_node)
                elif layer_idx == stage_layers[0] and stage > 0:
                    # First layer of stage connects to receive node
                    recv_node = f'pp_stage{stage}_recv_tp{tp_rank}'
                    dot.edge(recv_node, norm_node)
                else:
                    # Connect to previous layer's FFN all-reduce
                    prev_layer = layer_idx - 1
                    prev_ffn = f'layer{prev_layer}_ffn_allreduce_tp{tp_rank}'
                    dot.edge(prev_ffn, norm_node)
                
                # QKV Projection with Tensor Parallelism
                qkv_node = f'layer{layer_idx}_qkv_tp{tp_rank}'
                qkv_size = hidden_size // tp_size * 3  # Q, K, V combined
                dot.node(qkv_node,
                        f'Layer {layer_idx} QKV Projection\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, qkv_dim={qkv_size}]',
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(norm_node, qkv_node)
                
                # Attention Score Computation (Q * K^T)
                scores_node = f'layer{layer_idx}_attention_scores_tp{tp_rank}'
                dot.node(scores_node,
                        f'Layer {layer_idx} Attention Scores\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, qkv_dim={qkv_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//tp_size}, head_dim={hidden_size//num_heads}]',
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(qkv_node, scores_node)
                
                # Attention Weights (Softmax)
                weights_node = f'layer{layer_idx}_attention_weights_tp{tp_rank}'
                dot.node(weights_node,
                        f'Layer {layer_idx} Attention Weights\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//tp_size}, head_dim={hidden_size//num_heads}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//tp_size}, head_dim={hidden_size//num_heads}]',
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(scores_node, weights_node)
                
                # Attention Output (Weighted sum of values)
                attn_out_node = f'layer{layer_idx}_attention_output_tp{tp_rank}'
                dot.node(attn_out_node,
                        f'Layer {layer_idx} Attention Output\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//tp_size}, head_dim={hidden_size//num_heads}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_size}]',
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(weights_node, attn_out_node)
                
                # Attention All-Reduce (TP communication)
                attn_allreduce = f'layer{layer_idx}_attention_allreduce_tp{tp_rank}'
                dot.node(attn_allreduce,
                        f'Layer {layer_idx} Attention All-Reduce\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='ellipse', fillcolor='lightblue')
                dot.edge(attn_out_node, attn_allreduce)
                
                # Add TP communication edges
                other_tp_rank = 1 - tp_rank
                other_attn_allreduce = f'layer{layer_idx}_attention_allreduce_tp{other_tp_rank}'
                dot.edge(attn_allreduce, other_attn_allreduce, style='dashed', label='TP sync')
                
                # Residual connection
                residual_node = f'layer{layer_idx}_residual_tp{tp_rank}'
                dot.node(residual_node,
                        f'Layer {layer_idx} Residual Add\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='rectangle', fillcolor='lightgreen')
                
                # Connect residual (need to handle based on layer position)
                if layer_idx == 0 and stage == 0:
                    dot.edge('input', residual_node)
                elif layer_idx == stage_layers[0] and stage > 0:
                    recv_node = f'pp_stage{stage}_recv_tp{tp_rank}'
                    dot.edge(recv_node, residual_node)
                else:
                    prev_layer = layer_idx - 1
                    prev_ffn = f'layer{prev_layer}_ffn_allreduce_tp{tp_rank}'
                    dot.edge(prev_ffn, residual_node)
                
                dot.edge(attn_allreduce, residual_node)
                
                # Post-attention LayerNorm
                post_norm_node = f'layer{layer_idx}_post_norm_tp{tp_rank}'
                dot.node(post_norm_node,
                        f'Layer {layer_idx} Post-Attention RMSNorm\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(residual_node, post_norm_node)
                
                # FFN Gate Projection (for SwiGLU)
                ffn_gate_node = f'layer{layer_idx}_ffn_gate_tp{tp_rank}'
                ffn_intermediate = 28672 // tp_size  # Intermediate dimension
                dot.node(ffn_gate_node,
                        f'Layer {layer_idx} FFN Gate\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate={ffn_intermediate}]',
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(post_norm_node, ffn_gate_node)
                
                # FFN Up Projection (for SwiGLU)
                ffn_up_node = f'layer{layer_idx}_ffn_up_tp{tp_rank}'
                dot.node(ffn_up_node,
                        f'Layer {layer_idx} FFN Up\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate={ffn_intermediate}]',
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(post_norm_node, ffn_up_node)
                
                # FFN Activation (SwiGLU)
                ffn_act_node = f'layer{layer_idx}_ffn_activation_tp{tp_rank}'
                dot.node(ffn_act_node,
                        f'Layer {layer_idx} FFN SwiGLU\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate={ffn_intermediate}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate={ffn_intermediate}]',
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(ffn_gate_node, ffn_act_node)
                dot.edge(ffn_up_node, ffn_act_node)
                
                # FFN Down Projection
                ffn_down_node = f'layer{layer_idx}_ffn_down_tp{tp_rank}'
                dot.node(ffn_down_node,
                        f'Layer {layer_idx} FFN Down\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate={ffn_intermediate}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_size}]',
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(ffn_act_node, ffn_down_node)
                
                # FFN All-Reduce (TP communication)
                ffn_allreduce = f'layer{layer_idx}_ffn_allreduce_tp{tp_rank}'
                dot.node(ffn_allreduce,
                        f'Layer {layer_idx} FFN All-Reduce\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='ellipse', fillcolor='lightblue')
                dot.edge(ffn_down_node, ffn_allreduce)
                
                # Add TP communication edges for FFN
                other_tp_rank = 1 - tp_rank
                other_ffn_allreduce = f'layer{layer_idx}_ffn_allreduce_tp{other_tp_rank}'
                dot.edge(ffn_allreduce, other_ffn_allreduce, style='dashed', label='TP sync')
                
                # Final residual connection
                final_residual = f'layer{layer_idx}_final_residual_tp{tp_rank}'
                dot.node(final_residual,
                        f'Layer {layer_idx} Final Residual\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(ffn_allreduce, final_residual)
                dot.edge(residual_node, final_residual)
                
                # Add pipeline send if this is the last layer of a stage (except final stage)
                if layer_idx == stage_layers[-1] and stage < pp_size - 1:
                    # This layer outputs to next stage
                    pass  # Will be connected by next stage's receive
    
    # Output node
    dot.node('output', 
             f'OUTPUT\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
             shape='rectangle', fillcolor='lightcyan')
    
    # Connect final layer to output
    for tp_rank in range(tp_size):
        final_layer = num_layers - 1
        final_residual = f'layer{final_layer}_final_residual_tp{tp_rank}'
        dot.edge(final_residual, 'output')
    
    return dot

def create_simplified_llm_dag():
    """
    Create a simplified but complete DAG showing representative layers
    with proper connectivity and attention decomposition
    """
    
    # Create the DAG
    dot = Digraph(comment='LLM Deployment DAG - TP=2, PP=4 on 8x H100 GPUs (Complete)')
    dot.attr(rankdir='TB', size='50,50', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Model parameters
    hidden_size = 8192
    num_heads = 64
    batch_size = 'B'
    seq_len = 'S'
    tp_size = 2
    
    # Input node
    dot.node('input', 
             f'INPUT\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
             shape='rectangle', fillcolor='lightcyan')
    
    # Define representative layers to show (one from each pipeline stage)
    representative_layers = [0, 20, 40, 60, 79]  # Including final layer
    
    # Process each representative layer
    prev_layer_outputs = ['input']  # Track outputs from previous layer
    
    for i, layer_idx in enumerate(representative_layers):
        # Determine GPU assignment based on pipeline stage
        if layer_idx <= 19:
            stage = 0
            gpus = [0, 1]
        elif layer_idx <= 39:
            stage = 1
            gpus = [2, 3]
        elif layer_idx <= 59:
            stage = 2
            gpus = [4, 5]
        else:
            stage = 3
            gpus = [6, 7]
        
        # Add pipeline stage marker if this is first layer of a new stage
        if i > 0 and layer_idx != 79:
            prev_stage = stage - 1
            # Pipeline communication between stages
            for tp_rank in range(tp_size):
                gpu_id = gpus[tp_rank]
                recv_node = f'pp_stage{stage}_recv_tp{tp_rank}'
                dot.node(recv_node,
                        f'PP Stage {stage} Receive\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='parallelogram', fillcolor='lightcoral')
                
                # Connect to previous stage
                for prev_tp in range(tp_size):
                    prev_gpu = prev_stage * tp_size + prev_tp
                    if layer_idx == 20:
                        prev_layer_idx = 19
                    elif layer_idx == 40:
                        prev_layer_idx = 39
                    elif layer_idx == 60:
                        prev_layer_idx = 59
                    
                    prev_ffn = f'layer{prev_layer_idx}_ffn_allreduce_tp{prev_tp}'
                    dot.edge(prev_ffn, recv_node, label=f'pipeline_send_gpu{prev_gpu}_to_gpu{gpu_id}')
        
        # Process layer for each TP rank
        layer_outputs = []
        
        for tp_rank in range(tp_size):
            gpu_id = gpus[tp_rank]
            
            # Determine input source
            if layer_idx == 0:
                layer_input = 'input'
            elif layer_idx in [20, 40, 60]:
                layer_input = f'pp_stage{stage}_recv_tp{tp_rank}'
            else:
                # Connect to previous layer in same stage
                prev_layer_idx = layer_idx - 1
                layer_input = f'layer{prev_layer_idx}_final_residual_tp{tp_rank}'
            
            # Layer normalization
            norm_node = f'layer{layer_idx}_norm_tp{tp_rank}'
            dot.node(norm_node,
                    f'Layer {layer_idx} RMSNorm\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(layer_input, norm_node)
            
            # QKV Projection
            qkv_size = hidden_size // tp_size * 3
            qkv_node = f'layer{layer_idx}_qkv_tp{tp_rank}'
            dot.node(qkv_node,
                    f'Layer {layer_idx} QKV Projection\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, qkv_dim={qkv_size}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(norm_node, qkv_node)
            
            # Attention Score Computation
            scores_node = f'layer{layer_idx}_attention_scores_tp{tp_rank}'
            head_dim = hidden_size // num_heads
            dot.node(scores_node,
                    f'Layer {layer_idx} Attention Scores\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, qkv_dim={qkv_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//tp_size}, head_dim={head_dim}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(qkv_node, scores_node)
            
            # Attention Weights (Softmax)
            weights_node = f'layer{layer_idx}_attention_weights_tp{tp_rank}'
            dot.node(weights_node,
                    f'Layer {layer_idx} Attention Weights\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//tp_size}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//tp_size}, head_dim={head_dim}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(scores_node, weights_node)
            
            # Attention Output
            attn_out_node = f'layer{layer_idx}_attention_output_tp{tp_rank}'
            dot.node(attn_out_node,
                    f'Layer {layer_idx} Attention Output\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//tp_size}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_size}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(weights_node, attn_out_node)
            
            # Attention All-Reduce
            attn_allreduce = f'layer{layer_idx}_attention_allreduce_tp{tp_rank}'
            dot.node(attn_allreduce,
                    f'Layer {layer_idx} Attention All-Reduce\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                    shape='ellipse', fillcolor='lightblue')
            dot.edge(attn_out_node, attn_allreduce)
            
            # TP sync between ranks
            other_tp_rank = 1 - tp_rank
            other_attn_allreduce = f'layer{layer_idx}_attention_allreduce_tp{other_tp_rank}'
            dot.edge(attn_allreduce, other_attn_allreduce, style='dashed', label='TP sync')
            
            # Residual connection
            residual_node = f'layer{layer_idx}_residual_tp{tp_rank}'
            dot.node(residual_node,
                    f'Layer {layer_idx} Residual Add\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(attn_allreduce, residual_node)
            dot.edge(layer_input, residual_node)
            
            # Post-attention LayerNorm
            post_norm_node = f'layer{layer_idx}_post_norm_tp{tp_rank}'
            dot.node(post_norm_node,
                    f'Layer {layer_idx} Post-Attention RMSNorm\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(residual_node, post_norm_node)
            
            # FFN components
            ffn_intermediate = 28672 // tp_size
            
            # FFN Gate
            ffn_gate_node = f'layer{layer_idx}_ffn_gate_tp{tp_rank}'
            dot.node(ffn_gate_node,
                    f'Layer {layer_idx} FFN Gate\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate={ffn_intermediate}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(post_norm_node, ffn_gate_node)
            
            # FFN Up
            ffn_up_node = f'layer{layer_idx}_ffn_up_tp{tp_rank}'
            dot.node(ffn_up_node,
                    f'Layer {layer_idx} FFN Up\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate={ffn_intermediate}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(post_norm_node, ffn_up_node)
            
            # FFN Activation
            ffn_act_node = f'layer{layer_idx}_ffn_activation_tp{tp_rank}'
            dot.node(ffn_act_node,
                    f'Layer {layer_idx} FFN SwiGLU\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate={ffn_intermediate}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate={ffn_intermediate}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(ffn_gate_node, ffn_act_node)
            dot.edge(ffn_up_node, ffn_act_node)
            
            # FFN Down
            ffn_down_node = f'layer{layer_idx}_ffn_down_tp{tp_rank}'
            dot.node(ffn_down_node,
                    f'Layer {layer_idx} FFN Down\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate={ffn_intermediate}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_size}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(ffn_act_node, ffn_down_node)
            
            # FFN All-Reduce
            ffn_allreduce = f'layer{layer_idx}_ffn_allreduce_tp{tp_rank}'
            dot.node(ffn_allreduce,
                    f'Layer {layer_idx} FFN All-Reduce\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                    shape='ellipse', fillcolor='lightblue')
            dot.edge(ffn_down_node, ffn_allreduce)
            
            # TP sync for FFN
            other_tp_rank = 1 - tp_rank
            other_ffn_allreduce = f'layer{layer_idx}_ffn_allreduce_tp{other_tp_rank}'
            dot.edge(ffn_allreduce, other_ffn_allreduce, style='dashed', label='TP sync')
            
            # Final residual
            final_residual = f'layer{layer_idx}_final_residual_tp{tp_rank}'
            dot.node(final_residual,
                    f'Layer {layer_idx} Final Residual\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                    shape='rectangle', fillcolor='lightgreen')
            dot.edge(ffn_allreduce, final_residual)
            dot.edge(residual_node, final_residual)
            
            layer_outputs.append(final_residual)
        
        # Update tracking
        prev_layer_outputs = layer_outputs
    
    # Output node
    dot.node('output', 
             f'OUTPUT\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
             shape='rectangle', fillcolor='lightcyan')
    
    # Connect final layer to output
    for tp_rank in range(tp_size):
        final_residual = f'layer79_final_residual_tp{tp_rank}'
        dot.edge(final_residual, 'output')
    
    return dot

if __name__ == '__main__':
    # Create the complete DAG
    print("Generating complete LLM deployment DAG...")
    dag = create_simplified_llm_dag()
    
    # Save DOT file
    dot_path = '../outputs/2025-12-23-17-26-22/llm_deployment_dag_complete.dot'
    dag.save(dot_path)
    print(f"DOT file saved to: {dot_path}")
    
    # Render to SVG
    svg_path = '../outputs/2025-12-23-17-26-22/llm_deployment_dag_complete.svg'
    dag.render(svg_path, format='svg', cleanup=True)
    print(f"SVG file saved to: {svg_path}")
    
    print("DAG generation complete!")