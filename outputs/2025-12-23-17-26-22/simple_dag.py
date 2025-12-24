#!/usr/bin/env python3
import graphviz

dot = graphviz.Digraph(comment='LLM Deployment DAG - TP=2, PP=4 on 8x H100')
dot.attr(rankdir='TB', splines='true', nodesep='0.8', ranksep='1.2')

# Model parameters
hidden_size = 8192
num_heads = 64
d_k = hidden_size // num_heads
vocab_size = 32000

# Input node
dot.node('input', f'Input\\n[B,S] -> [B,S,{hidden_size}]', 
         shape='ellipse', fillcolor='white', style='filled')

# Embedding layer (distributed across all GPUs via TP)
dot.node('embed_tp0', f'Embedding TP0\\nGPU: 0,2,4,6\\n[B,S] -> [B,S,{hidden_size}]',
         shape='box', fillcolor='lightgreen')
dot.node('embed_tp1', f'Embedding TP1\\nGPU: 1,3,5,7\\n[B,S] -> [B,S,{hidden_size}]',
         shape='box', fillcolor='lightgreen')

# Add communication for embedding synchronization
dot.node('embed_allreduce', f'Embedding All-Reduce\\nGPUs: 0-7\\n[B,S,{hidden_size//2}] -> [B,S,{hidden_size}]',
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
    
    # Create representative nodes for each stage (simplified for visibility)
    # Create first and last layer of each stage to show the pattern
    for layer in [layer_start, layer_end]:
        # Attention computation - split across TP ranks
        # QKV projection
        qkv_tp0 = f'layer{layer}_qkv_tp0'
        qkv_tp1 = f'layer{layer}_qkv_tp1'
        
        dot.node(qkv_tp0, f'Layer {layer} QKV Proj TP0\\nGPU: {stage_start_gpu}\\n[B,S,{hidden_size}] -> [B,S,{hidden_size//2}]',
                 shape='box', fillcolor='lightgreen')
        dot.node(qkv_tp1, f'Layer {layer} QKV Proj TP1\\nGPU: {stage_end_gpu}\\n[B,S,{hidden_size}] -> [B,S,{hidden_size//2}]',
                 shape='box', fillcolor='lightgreen')
        
        # Attention all-reduce
        attn_allreduce = f'layer{layer}_attn_allreduce'
        dot.node(attn_allreduce, f'Layer {layer} Attention All-Reduce\\nGPUs: {stage_start_gpu}-{stage_end_gpu}\\n[B,S,{hidden_size//2}] -> [B,S,{hidden_size}]',
                 shape='ellipse', fillcolor='lightblue')
        
  # FFN computation
        ffn1_tp0 = f'layer{layer}_ffn1_tp0'
        ffn1_tp1 = f'layer{layer}_ffn1_tp1'
        
        dot.node(ffn1_tp0, f'Layer {layer} FFN1 TP0\\nGPU: {stage_start_gpu}\\n[B,S,{hidden_size}] -> [B,S,{4*hidden_size//2}]',
    shape='box', fillcolor='lightgreen')
        dot.node(ffn1_tp1, f'Layer {layer} FFN1 TP1\\nGPU: {stage_end_gpu}\\n[B,S,{hidden_size}] -> [B,S,{4*hidden_size//2}]',
                 shape='box', fillcolor='lightgreen')
        
  # FFN all-reduce
        ffn_allreduce = f'layer{layer}_ffn_allreduce'
        dot.node(ffn_allreduce, f'Layer {layer} FFN All-Reduce\\nGPUs: {stage_start_gpu}-{stage_end_gpu}\\n[B,S,{hidden_size//2}] -> [B,S,{hidden_size}]',
                 shape='ellipse', fillcolor='lightblue')
        
        # Connect the layer
        if layer == layer_start and stage == 0:
      # First layer in first stage connects to embedding
            dot.edge(prev_node, qkv_tp0)
   dot.edge(prev_node, qkv_tp1)
        elif layer == layer_start:
            # First layer in stage connects to previous stage
   prev_stage_end = (stage-1)*20 + 19
            prev_ffn_allreduce = f'layer{prev_stage_end}_ffn_allreduce'
            dot.edge(prev_ffn_allreduce, qkv_tp0)
            dot.edge(prev_ffn_allreduce, qkv_tp1)
        
        # Connect within layer (simplified)
        dot.edge(qkv_tp0, attn_allreduce)
        dot.edge(qkv_tp1, attn_allreduce)
        dot.edge(attn_allreduce, ffn1_tp0)
        dot.edge(attn_allreduce, ffn1_tp1)
        dot.edge(ffn1_tp0, ffn_allreduce)
        dot.edge(ffn1_tp1, ffn_allreduce)
        
        # Update previous node for next iteration
        if layer == layer_end:
            prev_node = ffn_allreduce

# Final layer norm (distributed)
final_ln_tp0 = 'final_ln_tp0'
final_ln_tp1 = 'final_ln_tp1'

dot.node(final_ln_tp0, f'Final LayerNorm TP0\\nGPU: 6\\n[B,S,{hidden_size}] -> [B,S,{hidden_size//2}]',
         shape='box', fillcolor='lightgreen')
dot.node(final_ln_tp1, f'Final LayerNorm TP1\\nGPU: 7\\n[B,S,{hidden_size}] -> [B,S,{hidden_size//2}]',
         shape='box', fillcolor='lightgreen')

# Connect final layer to layer norm
dot.edge(prev_node, final_ln_tp0)
dot.edge(prev_node, final_ln_tp1)

# Final all-reduce
final_allreduce = 'final_allreduce'
dot.node(final_allreduce, f'Final All-Reduce\\nGPUs: 6-7\\n[B,S,{hidden_size//2}] -> [B,S,{hidden_size}]',
         shape='ellipse', fillcolor='lightblue')

dot.edge(final_ln_tp0, final_allreduce)
dot.edge(final_ln_tp1, final_allreduce)

# Output projection (LM head)
output_proj_tp0 = 'output_proj_tp0'
output_proj_tp1 = 'output_proj_tp1'

dot.node(output_proj_tp0, f'Output Projection TP0\\nGPU: 6\\n[B,S,{hidden_size}] -> [B,S,{vocab_size//2}]',
         shape='box', fillcolor='lightgreen')
dot.node(output_proj_tp1, f'Output Projection TP1\\nGPU: 7\\n[B,S,{hidden_size}] -> [B,S,{vocab_size//2}]',
         shape='box', fillcolor='lightgreen')

dot.edge(final_allreduce, output_proj_tp0)
dot.edge(final_allreduce, output_proj_tp1)

# Final output all-reduce
output_allreduce = 'output_allreduce'
dot.node(output_allreduce, f'Output All-Reduce\\nGPUs: 6-7\\n[B,S,{vocab_size//2}] -> [B,S,{vocab_size}]',
         shape='ellipse', fillcolor='lightblue')

dot.edge(output_proj_tp0, output_allreduce)
dot.edge(output_proj_tp1, output_allreduce)

# Final output
output_node = 'output'
dot.node(output_node, f'Output\\n[B,S,{vocab_size}] -> [B,S,{vocab_size}]',
         shape='ellipse', fillcolor='white', style='filled')

dot.edge(output_allreduce, output_node)

# Save as DOT file
dot.save('../outputs/2025-12-23-17-26-22/llm_deployment_dag.dot')

# Render as SVG
dot.render('../outputs/2025-12-23-17-26-22/llm_deployment_dag', format='svg', cleanup=False)

print("DAG generated successfully!")
print(f"DOT file: ../outputs/2025-12-23-17-26-22/llm_deployment_dag.dot")
print(f"SVG file: ../outputs/2025-12-23-17-26-22/llm_deployment_dag.svg")