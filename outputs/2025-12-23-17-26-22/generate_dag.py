#!/usr/bin/env python3
"""
LLM Deployment DAG Generator
Generates a detailed DAG for Llama3 70B deployment with TP=2, PP=4 on 8x H100 GPUs
"""

import graphviz
from graphviz import Digraph

def create_llm_deployment_dag():
    # Create a new directed graph
    dot = Digraph(comment='LLM Deployment DAG - TP=2, PP=4 on 8x H100')
    dot.attr(rankdir='TB', splines='true', nodesep='0.8', ranksep='1.2')
    
    # Model parameters from deployment plan
    hidden_size = 8192
    num_heads = 64
    d_k = hidden_size // num_heads  # 128
    vocab_size = 32000  # Typical for Llama models
    
    # Input node
    dot.node('input', f'Input\\nInput: [batch_size=B, seq_len=S]\\nOutput: [batch_size=B, seq_len=S, hidden={hidden_size}]', 
             shape='ellipse', fillcolor='white', style='filled')
    
    # Embedding layer (distributed across all GPUs via TP)
    dot.node('embed_tp0', f'Embedding TP0\\nGPU: 0,2,4,6\\nInput: [B,S]\\nOutput: [B,S,{hidden_size}]',
             shape='box', fillcolor='lightgreen')
    dot.node('embed_tp1', f'Embedding TP1\\nGPU: 1,3,5,7\\nInput: [B,S]\\nOutput: [B,S,{hidden_size}]',
             shape='box', fillcolor='lightgreen')
    
    # Add communication for embedding synchronization
    dot.node('embed_allreduce', f'Embedding All-Reduce\\nGPUs: 0-7\\nInput: [B,S,{hidden_size//2}]\\nOutput: [B,S,{hidden_size}]',
             shape='ellipse', fillcolor='lightblue')
    
    # Connect input to embedding
    dot.edge('input', 'embed_tp0')
    dot.edge('input', 'embed_tp1')
    dot.edge('embed_tp0', 'embed_allreduce')
    dot.edge('embed_tp1', 'embed_allreduce')
    
    # Process each pipeline stage
    prev_node = 'embed_allreduce'
    
    for stage in range(4):
        stage_start_gpu = stage * 2
        stage_end_gpu = stage_start_gpu + 1
        layer_start = stage * 20
        layer_end = layer_start + 19
        
        # Create nodes for each layer in the stage
        for layer in range(layer_start, layer_end + 1):
            # Attention computation - split across TP ranks
            # QKV projection
            qkv_tp0 = f'layer{layer}_qkv_tp0'
            qkv_tp1 = f'layer{layer}_qkv_tp1'
            
            dot.node(qkv_tp0, f'Layer {layer} QKV Proj TP0\\nGPU: {stage_start_gpu}\\nInput: [B,S,{hidden_size}]\\nOutput: [B,S,{hidden_size//2}]',
                     shape='box', fillcolor='lightgreen')
            dot.node(qkv_tp1, f'Layer {layer} QKV Proj TP1\\nGPU: {stage_end_gpu}\\nInput: [B,S,{hidden_size}]\\nOutput: [B,S,{hidden_size//2}]',
                     shape='box', fillcolor='lightgreen')
            
            # Attention scores computation
            attn_scores_tp0 = f'layer{layer}_attn_scores_tp0'
            attn_scores_tp1 = f'layer{layer}_attn_scores_tp1'
            
            dot.node(attn_scores_tp0, f'Layer {layer} Attention Scores TP0\\nGPU: {stage_start_gpu}\\nInput: [B,S,{hidden_size//2}]\\nOutput: [B,S,{num_heads//2},S]',
                     shape='box', fillcolor='lightgreen')
            dot.node(attn_scores_tp1, f'Layer {layer} Attention Scores TP1\\nGPU: {stage_end_gpu}\\nInput: [B,S,{hidden_size//2}]\\nOutput: [B,S,{num_heads//2},S]',
                     shape='box', fillcolor='lightgreen')
            
   # Attention output
            attn_out_tp0 = f'layer{layer}_attn_out_tp0'
            attn_out_tp1 = f'layer{layer}_attn_out_tp1'
            
            dot.node(attn_out_tp0, f'Layer {layer} Attention Output TP0\\nGPU: {stage_start_gpu}\\nInput: [B,S,{num_heads//2},{d_k}]\\nOutput: [B,S,{hidden_size//2}]',
      shape='box', fillcolor='lightgreen')
            dot.node(attn_out_tp1, f'Layer {layer} Attention Output TP1\\nGPU: {stage_end_gpu}\\nInput: [B,S,{num_heads//2},{d_k}]\\nOutput: [B,S,{hidden_size//2}]',
     shape='box', fillcolor='lightgreen')
            
            # Attention all-reduce
            attn_allreduce = f'layer{layer}_attn_allreduce'
            dot.node(attn_allreduce, f'Layer {layer} Attention All-Reduce\\nGPUs: {stage_start_gpu}-{stage_end_gpu}\\nInput: [B,S,{hidden_size//2}]\\nOutput: [B,S,{hidden_size}]',
                     shape='ellipse', fillcolor='lightblue')
            
            # FFN computation
            ffn1_tp0 = f'layer{layer}_ffn1_tp0'
   ffn1_tp1 = f'layer{layer}_ffn1_tp1'
            
            dot.node(ffn1_tp0, f'Layer {layer} FFN1 TP0\\nGPU: {stage_start_gpu}\\nInput: [B,S,{hidden_size}]\\nOutput: [B,S,{4*hidden_size//2}]',
    shape='box', fillcolor='lightgreen')
            dot.node(ffn1_tp1, f'Layer {layer} FFN1 TP1\\nGPU: {stage_end_gpu}\\nInput: [B,S,{hidden_size}]\\nOutput: [B,S,{4*hidden_size//2}]',
    shape='box', fillcolor='lightgreen')
            
            # FFN activation
            ffn_act_tp0 = f'layer{layer}_ffn_act_tp0'
            ffn_act_tp1 = f'layer{layer}_ffn_act_tp1'
            
  dot.node(ffn_act_tp0, f'Layer {layer} FFN Activation TP0\\nGPU: {stage_start_gpu}\\nInput: [B,S,{4*hidden_size//2}]\\nOutput: [B,S,{4*hidden_size//2}]',
   shape='box', fillcolor='lightgreen')
            dot.node(ffn_act_tp1, f'Layer {layer} FFN Activation TP1\\nGPU: {stage_end_gpu}\\nInput: [B,S,{4*hidden_size//2}]\\nOutput: [B,S,{4*hidden_size//2}]',
    shape='box', fillcolor='lightgreen')
            
            # FFN2 computation
            ffn2_tp0 = f'layer{layer}_ffn2_tp0'
            ffn2_tp1 = f'layer{layer}_ffn2_tp1'
            
            dot.node(ffn2_tp0, f'Layer {layer} FFN2 TP0\\nGPU: {stage_start_gpu}\\nInput: [B,S,{4*hidden_size//2}]\\nOutput: [B,S,{hidden_size//2}]',
   shape='box', fillcolor='lightgreen')
            dot.node(ffn2_tp1, f'Layer {layer} FFN2 TP1\\nGPU: {stage_end_gpu}\\nInput: [B,S,{4*hidden_size//2}]\\nOutput: [B,S,{hidden_size//2}]',
    shape='box', fillcolor='lightgreen')
            
            # FFN all-reduce
            ffn_allreduce = f'layer{layer}_ffn_allreduce'
            dot.node(ffn_allreduce, f'Layer {layer} FFN All-Reduce\\nGPUs: {stage_start_gpu}-{stage_end_gpu}\\nInput: [B,S,{hidden_size//2}]\\nOutput: [B,S,{hidden_size}]',
      shape='ellipse', fillcolor='lightblue')
            
            # Connect the layer
            if layer == layer_start:
         # First layer in stage connects to previous stage or embedding
    dot.edge(prev_node, qkv_tp0)
     dot.edge(prev_node, qkv_tp1)
            else:
                # Connect to previous layer's output
    prev_layer = layer - 1
       prev_ffn_allreduce = f'layer{prev_layer}_ffn_allreduce'
            dot.edge(prev_ffn_allreduce, qkv_tp0)
            dot.edge(prev_ffn_allreduce, qkv_tp1)
            
            # Connect within layer
            dot.edge(qkv_tp0, attn_scores_tp0)
            dot.edge(qkv_tp1, attn_scores_tp1)
            dot.edge(attn_scores_tp0, attn_out_tp0)
            dot.edge(attn_scores_tp1, attn_out_tp1)
            dot.edge(attn_out_tp0, attn_allreduce)
            dot.edge(attn_out_tp1, attn_allreduce)
            dot.edge(attn_allreduce, ffn1_tp0)
            dot.edge(attn_allreduce, ffn1_tp1)
            dot.edge(ffn1_tp0, ffn_act_tp0)
         dot.edge(ffn1_tp1, ffn_act_tp1)
            dot.edge(ffn_act_tp0, ffn2_tp0)
    dot.edge(ffn_act_tp1, ffn2_tp1)
            dot.edge(ffn2_tp0, ffn_allreduce)
            dot.edge(ffn2_tp1, ffn_allreduce)
            
    # Update previous node for next iteration
            prev_node = ffn_allreduce
    
    # Final layer norm (distributed)
    final_ln_tp0 = 'final_ln_tp0'
    final_ln_tp1 = 'final_ln_tp1'
    
    dot.node(final_ln_tp0, f'Final LayerNorm TP0\\nGPU: 6\\nInput: [B,S,{hidden_size}]\\nOutput: [B,S,{hidden_size//2}]',
  shape='box', fillcolor='lightgreen')
    dot.node(final_ln_tp1 f'Final LayerNorm TP1\\nGPU: 7\\nInput: [B,S,{hidden_size}]\\nOutput: [B,S,{hidden_size//2}]',
             shape='box', fillcolor='lightgreen')
    
    # Connect final layer to layer norm
    dot.edge(prev_node, final_ln_tp0)
    dot.edge(prev_node, final_ln_tp1)
    
    # Final all-reduce
    final_allreduce = 'final_allreduce'
    dot.node(final_allreduce, f'Final All-Reduce\\nGPUs: 6-7\\nInput: [B,S,{hidden_size//2}]\\nOutput: [B,S,{hidden_size}]',
  shape='ellipse', fillcolor='lightblue')
    
    dot.edge(final_ln_tp0, final_allreduce)
    dot.edge(final_ln_tp1, final_allreduce)
    
    # Output projection (LM head)
    output_proj_tp0 = 'output_proj_tp0'
    output_proj_tp1 = 'output_proj_tp1'
    
    dot.node(output_proj_tp0, f'Output Projection TP0\\nGPU: 6\\nInput: [B,S,{hidden_size}]\\nOutput: [B,S,{vocab_size//2}]',
             shape='box', fillcolor='lightgreen')
    dot.node(output_proj_tp1, f'Output Projection TP1\\nGPU: 7\\nInput: [B,S,{hidden_size}]\\nOutput: [B,S,{vocab_size//2}]',
             shape='box', fillcolor='lightgreen')
    
    dot.edge(final_allreduce, output_proj_tp0)
    dot.edge(final_allreduce, output_proj_tp1)
    
    # Final output all-reduce
    output_allreduce = 'output_allreduce'
    dot.node(output_allreduce, f'Output All-Reduce\\nGPUs: 6-7\\nInput: [B,S,{vocab_size//2}]\\nOutput: [B,S,{vocab_size}]',
  shape='ellipse', fillcolor='lightblue')
    
    dot.edge(output_proj_tp0, output_allreduce)
    dot.edge(output_proj_tp1, output_allreduce)
    
    # Final output
    output_node = 'output'
    dot.node(output_node, f'Output\\nInput: [B,S,{vocab_size}]\\nOutput: [B,S,{vocab_size}]',
             shape='ellipse', fillcolor='white', style='filled')
    
    dot.edge(output_allreduce, output_node)
    
    # Add pipeline communication between stages
    # These represent the activations being passed between pipeline stages
    for stage in range(3):  # 0, 1, 2 (stage 3 is the last)
        stage_start_layer = stage * 20
  stage_end_layer = stage_start_layer + 19
        next_stage_start_layer = (stage + 1) * 20
        
        current_stage_output = f'layer{stage_end_layer}_ffn_allreduce'
        next_stage_qkv_tp0 = f'layer{next_stage_start_layer}_qkv_tp0'
        next_stage_qkv_tp1 = f'layer{next_stage_start_layer}_qkv_tp1'
        
        # Add pipeline communication nodes
        pipeline_comm_0 = f'pipeline_stage{stage}_to_{stage+1}_tp0'
        pipeline_comm_1 = f'pipeline_stage{stage}_to_{stage+1}_tp1'
        
     dot.node(pipeline_comm_0, f'Pipeline Stage {stage}→{stage+1} TP0\\nGPU: {stage*2}→{(stage+1)*2}\\nInput: [B,S,{hidden_size}]\\nOutput: [B,S,{hidden_size}]',
  shape='ellipse', fillcolor='lightblue', style='dashed')
        dot.node(pipeline_comm_1, f'Pipeline Stage {stage}→{stage+1} TP1\\nGPU: {stage*2+1}→{(stage+1)*2+1}\\nInput: [B,S,{hidden_size}]\\nOutput: [B,S,{hidden_size}]',
             shape='ellipse', fillcolor='lightblue', style='dashed')
        
        # Connect pipeline communication
        dot.edge(current_stage_output, pipeline_comm_0)
        dot.edge(current_stage_output, pipeline_comm_1)
        dot.edge(pipeline_comm_0, next_stage_qkv_tp0)
        dot.edge(pipeline_comm_1, next_stage_qkv_tp1)
    
    return dot

if __name__ == '__main__':
    # Generate the DAG
    dag = create_llm_deployment_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-12-23-17-26-22/llm_deployment_dag.dot')
    
    # Render as SVG
    dag.render('../outputs/2025-12-23-17-26-22/llm_deployment_dag', format='svg', cleanup=False)
    
    print("DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-23-17-26-22/llm_deployment_dag.dot")
    print(f"SVG file: ../outputs/2025-12-23-17-26-22/llm_deployment_dag.svg")