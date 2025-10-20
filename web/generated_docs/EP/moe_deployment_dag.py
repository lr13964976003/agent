import graphviz
import os

# Create DAG for the proposed cross-node expert parallelism deployment
# EP=16, 4 layers, 16 experts per layer, deployed on 16 GPUs

dot = graphviz.Digraph('MoE_Cross_Node_Expert_Parallelism', 
                      comment='4-layer MoE with EP=16 deployment on 16 GPUs')

# Set graph attributes
dot.attr(rankdir='TB', size='20,20')
dot.attr('node', shape='rectangle')

# Input node
dot.node('input', 'Model Input\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]', 
         shape='ellipse', style='filled', fillcolor='lightblue')

# Global nodes for layer processing
for layer_idx in range(4):
    layer_name = f"layer_{layer_idx}"
    
    # Layer input processing
    dot.node(f"{layer_name}_input", f"Layer {layer_idx} Input\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nGPU: all GPUs",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Multi-Head Attention (executed on all GPUs)
    dot.node(f"{layer_name}_mha_qkv", f"MHA QKV Projection\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPU: all GPUs",
             style='filled', fillcolor='lightgreen')
    
    dot.node(f"{layer_name}_mha_attention", f"MHA Attention\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nGPU: all GPUs",
             style='filled', fillcolor='lightgreen')
    
    dot.node(f"{layer_name}_mha_proj", f"MHA Output Projection\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nGPU: all GPUs",
             style='filled', fillcolor='lightgreen')
    
    # Residual connection after MHA
    dot.node(f"{layer_name}_mha_residual", f"MHA Residual Add\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nGPU: all GPUs",
             style='filled', fillcolor='lightcoral')
    
    # Layer normalization after MHA
    dot.node(f"{layer_name}_mha_layernorm", f"Layer Norm (Post-MHA)\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nGPU: all GPUs",
             style='filled', fillcolor='lightgray')
    
    # Expert routing (gate)
    dot.node(f"{layer_name}_gate", f"Expert Gate\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nOutput: [batch_size=1024, seq_len=10000, k=2]\nGPU: all GPUs",
             shape='parallelogram', style='filled', fillcolor='orange')
    
    # Token splitting for expert routing
    dot.node(f"{layer_name}_token_split", f"Token Split by Expert\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nOutput: [batch_size=variable, hidden_size=8192] per expert\nGPU: all GPUs",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Experts (distributed across 16 GPUs)
    for expert_idx in range(16):
        gpu_id = expert_idx + layer_idx * 16 % 16  # Round-robin placement across GPUs
        
        # Expert computation
        dot.node(f"{layer_name}_expert_{expert_idx}", 
                f"Expert {expert_idx}\nInput: [batch_size=variable, hidden_size=8192]\nOutput: [batch_size=variable, hidden_size=8192]\nGPU: {gpu_id}",
                style='filled', fillcolor='lightcyan')
        
        # Expert MLP components
        dot.node(f"{layer_name}_expert_{expert_idx}_gate_proj", 
                f"Expert {expert_idx} Gate Proj\nInput: [batch_size=variable, hidden_size=8192]\nOutput: [batch_size=variable, ffn_hidden=32768]\nGPU: {gpu_id}",
                style='filled', fillcolor='lightsteelblue')
        
        dot.node(f"{layer_name}_expert_{expert_idx}_up_proj", 
                f"Expert {expert_idx} Up Proj\nInput: [batch_size=variable, hidden_size=8192]\nOutput: [batch_size=variable, ffn_hidden=32768]\nGPU: {gpu_id}",
                style='filled', fillcolor='lightsteelblue')
        
        dot.node(f"{layer_name}_expert_{expert_idx}_activation", 
                f"Expert {expert_idx} GELU Activation\nInput: [batch_size=variable, ffn_hidden=32768]\nOutput: [batch_size=variable, ffn_hidden=32768]\nGPU: {gpu_id}",
                style='filled', fillcolor='lightsteelblue')
        
        dot.node(f"{layer_name}_expert_{expert_idx}_down_proj", 
                f"Expert {expert_idx} Down Proj\nInput: [batch_size=variable, ffn_hidden=32768]\nOutput: [batch_size=variable, hidden_size=8192]\nGPU: {gpu_id}",
                style='filled', fillcolor='lightsteelblue')
    
    # Expert aggregation
    dot.node(f"{layer_name}_expert_agg", f"Expert Output Aggregation\nInput: [batch_size=variable, hidden_size=8192] from all experts\nOutput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nGPU: all GPUs",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Residual connection after experts
    dot.node(f"{layer_name}_expert_residual", f"Expert Residual Add\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nGPU: all GPUs",
             style='filled', fillcolor='lightcoral')
    
    # Layer normalization after experts
    dot.node(f"{layer_name}_expert_layernorm", f"Layer Norm (Post-Experts)\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden_size=8192]\nGPU: all GPUs",
             style='filled', fillcolor='lightgray')

# Output node
dot.node('output', 'Model Output\nOutput: [batch_size=1024, seq_len=10000, hidden_size=8192]', 
         shape='ellipse', style='filled', fillcolor='lightblue')

# Connect the DAG
# Input to first layer
for layer_idx in range(4):
    layer_name = f"layer_{layer_idx}"
    
    if layer_idx == 0:
        dot.edge('input', f"{layer_name}_input")
    else:
        prev_layer = f"layer_{layer_idx-1}"
        dot.edge(f"{prev_layer}_expert_layernorm", f"{layer_name}_input")
    
    # MHA connections
    dot.edge(f"{layer_name}_input", f"{layer_name}_mha_qkv")
    dot.edge(f"{layer_name}_mha_qkv", f"{layer_name}_mha_attention")
    dot.edge(f"{layer_name}_mha_attention", f"{layer_name}_mha_proj")
    dot.edge(f"{layer_name}_input", f"{layer_name}_mha_residual")
    dot.edge(f"{layer_name}_mha_proj", f"{layer_name}_mha_residual")
    dot.edge(f"{layer_name}_mha_residual", f"{layer_name}_mha_layernorm")
    
    # Expert routing
    dot.edge(f"{layer_name}_mha_layernorm", f"{layer_name}_gate")
    dot.edge(f"{layer_name}_mha_layernorm", f"{layer_name}_token_split")
    dot.edge(f"{layer_name}_gate", f"{layer_name}_token_split", style='dashed')
    
    # Connect to experts
    for expert_idx in range(16):
        dot.edge(f"{layer_name}_token_split", f"{layer_name}_expert_{expert_idx}")
        dot.edge(f"{layer_name}_expert_{expert_idx}", f"{layer_name}_expert_{expert_idx}_gate_proj")
        dot.edge(f"{layer_name}_expert_{expert_idx}_gate_proj", f"{layer_name}_expert_{expert_idx}_activation")
        dot.edge(f"{layer_name}_expert_{expert_idx}", f"{layer_name}_expert_{expert_idx}_up_proj")
        dot.edge(f"{layer_name}_expert_{expert_idx}_up_proj", f"{layer_name}_expert_{expert_idx}_activation")
        dot.edge(f"{layer_name}_expert_{expert_idx}_activation", f"{layer_name}_expert_{expert_idx}_down_proj")
        dot.edge(f"{layer_name}_expert_{expert_idx}_down_proj", f"{layer_name}_expert_agg")
    
    # Final connections
    dot.edge(f"{layer_name}_mha_layernorm", f"{layer_name}_expert_residual")
    dot.edge(f"{layer_name}_expert_agg", f"{layer_name}_expert_residual")
    dot.edge(f"{layer_name}_expert_residual", f"{layer_name}_expert_layernorm")

# Final output
final_layer = "layer_3"
dot.edge(f"{final_layer}_expert_layernorm", 'output')

# Save the DOT file
dot.render('./generated_docs/EP/moe_cross_node_expert_parallelism', format='dot', cleanup=False)

# Also save as SVG for visualization
dot.render('./generated_docs/EP/moe_cross_node_expert_parallelism', format='svg', cleanup=False)

print("Generated DAG saved to:")
print("- DOT file: ./generated_docs/EP/moe_cross_node_expert_parallelism.dot")
print("- SVG file: ./generated_docs/EP/moe_cross_node_expert_parallelism.svg")