import graphviz
from graphviz import Digraph
import os

def create_simplified_llm_deployment_dag():
    # Create a new directed graph
    dot = Digraph(comment='LLM Deployment DAG - Simplified View')
    dot.attr(rankdir='TB', size='20,30', dpi='300')
    dot.attr('node', fontsize='10', margin='0.1,0.05')
    
    # Model specifications
    total_layers = 16
    total_experts = 64
    hidden_size = 1024
    attention_heads = 16
    batch_size = "?"
    seq_len = "?"
    
    # Parallel configuration
    tp_size = 8   # Tensor Parallel
    pp_size = 4   # Pipeline Parallel
    ep_size = 16  # Expert Parallel
    dp_size = 4   # Data Parallel
    
    # Create a representative view showing one pipeline stage with tensor parallelism
    # and expert parallelism interactions
    
    # Input node
    dot.node('input', 
             f'Input\\nDP Replica\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
             shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Process each layer
    for layer_id in range(min(4, total_layers)):  # Show first 4 layers for clarity
        
        # Layer normalization (Tensor Parallel)
        dot.node(f'layernorm_l{layer_id}', 
                 f'LayerNorm L{layer_id}\\nTP Split\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Attention components (Tensor Parallel)
        dot.node(f'attn_q_l{layer_id}', 
                 f'Attention Query\\nL{layer_id} Col-Parallel\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, d_k={hidden_size//attention_heads}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'attn_k_l{layer_id}', 
                 f'Attention Key\\nL{layer_id} Col-Parallel\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, d_k={hidden_size//attention_heads}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'attn_v_l{layer_id}', 
                 f'Attention Value\\nL{layer_id} Col-Parallel\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, d_k={hidden_size//attention_heads}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'attn_score_l{layer_id}', 
                 f'Attention Score\\nL{layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, d_k={hidden_size//attention_heads}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, seq={seq_len}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'attn_softmax_l{layer_id}', 
                 f'Attention Softmax\\nL{layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, seq={seq_len}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, seq={seq_len}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'attn_out_l{layer_id}', 
                 f'Attention Output\\nL{layer_id} Row-Parallel\\nInput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, d_k={hidden_size//attention_heads}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        # All-Reduce for Attention (Tensor Parallel Communication)
        dot.node(f'attn_allreduce_l{layer_id}', 
                 f'All-Reduce\\nAttention L{layer_id}\\nTP Group\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                 shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # MLP Gate (Expert Parallel - Routing)
        dot.node(f'mlp_gate_l{layer_id}', 
                 f'MLP Gate\\nL{layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, experts=4]',
                 shape='parallelogram', style='filled', fillcolor='yellow')
        
        # Expert processing (Expert Parallel)
        for expert_id in range(4):  # Show 4 experts per layer
            dot.node(f'expert{expert_id}_l{layer_id}', 
                     f'Expert {expert_id}\\nL{layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                     shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Expert Aggregation
        dot.node(f'expert_agg_l{layer_id}', 
                 f'Expert Aggregation\\nL{layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, experts=4, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                 shape='parallelogram', style='filled', fillcolor='yellow')
        
        # MLP components (Tensor Parallel)
        dot.node(f'mlp_fc1_l{layer_id}', 
                 f'MLP FC1\\nL{layer_id} Col-Parallel\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, ffn={4096//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'mlp_gelu_l{layer_id}', 
                 f'MLP GELU\\nL{layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, ffn={4096//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, ffn={4096//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'mlp_fc2_l{layer_id}', 
                 f'MLP FC2\\nL{layer_id} Row-Parallel\\nInput: [batch={batch_size}, seq={seq_len}, ffn={4096//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        # All-Reduce for MLP (Tensor Parallel Communication)
        dot.node(f'mlp_allreduce_l{layer_id}', 
                 f'All-Reduce\\nMLP L{layer_id}\\nTP Group\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                 shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Add remaining layers indicator
        if layer_id == 3:
            dot.node(f'more_layers', 
                     f'...\\n12 More Layers\\n(Similar Pattern)',
                     shape='rectangle', style='dashed', fillcolor='white')
    
    # Output node
    dot.node('output', 
             f'Output\\nDP Replica\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
             shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Connect the nodes
    dot.edge('input', 'layernorm_l0')
    
    for layer_id in range(min(4, total_layers)):
        # Attention path
        dot.edge(f'layernorm_l{layer_id}', f'attn_q_l{layer_id}')
        dot.edge(f'layernorm_l{layer_id}', f'attn_k_l{layer_id}')
        dot.edge(f'layernorm_l{layer_id}', f'attn_v_l{layer_id}')
        
        dot.edge(f'attn_q_l{layer_id}', f'attn_score_l{layer_id}')
        dot.edge(f'attn_k_l{layer_id}', f'attn_score_l{layer_id}')
        dot.edge(f'attn_score_l{layer_id}', f'attn_softmax_l{layer_id}')
        dot.edge(f'attn_softmax_l{layer_id}', f'attn_out_l{layer_id}')
        dot.edge(f'attn_v_l{layer_id}', f'attn_out_l{layer_id}')
        dot.edge(f'attn_out_l{layer_id}', f'attn_allreduce_l{layer_id}')
        
        # MLP path
        dot.edge(f'attn_allreduce_l{layer_id}', f'mlp_gate_l{layer_id}')
        
        # Expert routing (dashed lines)
        for expert_id in range(4):
            dot.edge(f'mlp_gate_l{layer_id}', f'expert{expert_id}_l{layer_id}', style='dashed')
            dot.edge(f'expert{expert_id}_l{layer_id}', f'expert_agg_l{layer_id}')
        
        dot.edge(f'expert_agg_l{layer_id}', f'mlp_fc1_l{layer_id}')
        dot.edge(f'mlp_fc1_l{layer_id}', f'mlp_gelu_l{layer_id}')
        dot.edge(f'mlp_gelu_l{layer_id}', f'mlp_fc2_l{layer_id}')
        dot.edge(f'mlp_fc2_l{layer_id}', f'mlp_allreduce_l{layer_id}')
        
        # Connect to next layer or output
        if layer_id < 3:
            dot.edge(f'mlp_allreduce_l{layer_id}', f'layernorm_l{layer_id+1}')
        elif layer_id == 3:
            dot.edge(f'mlp_allreduce_l{layer_id}', 'more_layers')
            dot.edge('more_layers', 'output')
        else:
            dot.edge(f'mlp_allreduce_l{layer_id}', 'output')
    
    # Add communication annotations
    dot.node('tp_comm', 'Tensor Parallel\\nAll-Reduce Communications', 
             shape='ellipse', style='filled', fillcolor='lightgreen')
    dot.node('ep_comm', 'Expert Parallel\\nRouting Communications', 
             shape='ellipse', style='filled', fillcolor='lightgreen')
    dot.node('dp_comm', 'Data Parallel\\nGradient Sync', 
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    return dot

def create_detailed_layer_view():
    # Create a detailed view of a single layer showing all parallel dimensions
    dot = Digraph(comment='LLM Layer Detail - All Parallel Dimensions')
    dot.attr(rankdir='LR', size='15,10', dpi='300')
    dot.attr('node', fontsize='10', margin='0.1,0.05')
    
    # Model specs
    hidden_size = 1024
    attention_heads = 16
    batch_size = "?"
    seq_len = "?"
    tp_size = 8
    
    # Input to layer
    dot.node('layer_input', 
             f'Layer Input\\n[batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
             shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Tensor Parallel Split
    for tp_rank in range(tp_size):
        dot.node(f'tp_split_{tp_rank}', 
                 f'TP Split {tp_rank}\\n[batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                 shape='parallelogram', style='filled', fillcolor='yellow')
        
        # Attention components for this TP rank
        dot.node(f'attn_q_tp{tp_rank}', 
                 f'Query\\nTP{tp_rank}\\n[batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'attn_k_tp{tp_rank}', 
                 f'Key\\nTP{tp_rank}\\n[batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'attn_v_tp{tp_rank}', 
                 f'Value\\nTP{tp_rank}\\n[batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'attn_out_tp{tp_rank}', 
                 f'Attention Output\\nTP{tp_rank}\\n[batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        # MLP components for this TP rank
        dot.node(f'mlp_fc1_tp{tp_rank}', 
                 f'MLP FC1\\nTP{tp_rank}\\n[batch={batch_size}, seq={seq_len}, ffn={4096//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'mlp_fc2_tp{tp_rank}', 
                 f'MLP FC2\\nTP{tp_rank}\\n[batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                 shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Expert Parallel Gate and All-Reduce
    dot.node('expert_gate', 
             f'Expert Gate\\n[batch={batch_size}, seq={seq_len}, experts=64]',
             shape='parallelogram', style='filled', fillcolor='yellow')
    
    dot.node('allreduce_attn', 
             f'All-Reduce\\nAttention\\n[batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    dot.node('allreduce_mlp', 
             f'All-Reduce\\nMLP\\n[batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Output
    dot.node('layer_output', 
             f'Layer Output\\n[batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
             shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Connect the nodes
    for tp_rank in range(tp_size):
        dot.edge('layer_input', f'tp_split_{tp_rank}')
        dot.edge(f'tp_split_{tp_rank}', f'attn_q_tp{tp_rank}')
        dot.edge(f'tp_split_{tp_rank}', f'attn_k_tp{tp_rank}')
        dot.edge(f'tp_split_{tp_rank}', f'attn_v_tp{tp_rank}')
        
        dot.edge(f'attn_q_tp{tp_rank}', 'expert_gate')
        dot.edge(f'attn_k_tp{tp_rank}', 'expert_gate')
        dot.edge(f'attn_v_tp{tp_rank}', 'expert_gate')
        
        dot.edge('expert_gate', f'attn_out_tp{tp_rank}')
        dot.edge(f'attn_out_tp{tp_rank}', 'allreduce_attn')
        dot.edge('allreduce_attn', f'mlp_fc1_tp{tp_rank}')
        dot.edge(f'mlp_fc1_tp{tp_rank}', f'mlp_fc2_tp{tp_rank}')
        dot.edge(f'mlp_fc2_tp{tp_rank}', 'allreduce_mlp')
        dot.edge('allreduce_mlp', 'layer_output')
    
    return dot

def main():
    # Model specifications
    total_layers = 16
    total_experts = 64
    hidden_size = 1024
    attention_heads = 16
    batch_size = "?"
    seq_len = "?"
    
    # Parallel configuration
    tp_size = 8   # Tensor Parallel
    pp_size = 4   # Pipeline Parallel
    ep_size = 16  # Expert Parallel
    dp_size = 4   # Data Parallel
    
    # Create the simplified DAG
    dag_simplified = create_simplified_llm_deployment_dag()
    
    # Create the detailed layer view
    dag_detailed = create_detailed_layer_view()
    
    # Save files
    files_created = []
    
    # Simplified DAG
    dot_file_path = '../outputs/2025-12-04-16-15-16/llm_deployment_dag_simplified.dot'
    dag_simplified.save(dot_file_path)
    files_created.append(dot_file_path)
    
    svg_file_path = '../outputs/2025-12-04-16-15-16/llm_deployment_dag_simplified.svg'
    try:
        dag_simplified.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
        files_created.append(svg_file_path)
    except Exception as e:
        print(f"SVG rendering failed for simplified DAG: {e}")
    
    # Detailed layer view
    dot_detailed_path = '../outputs/2025-12-04-16-15-16/llm_layer_detail.dot'
    dag_detailed.save(dot_detailed_path)
    files_created.append(dot_detailed_path)
    
    svg_detailed_path = '../outputs/2025-12-04-16-15-16/llm_layer_detail.svg'
    try:
        dag_detailed.render(svg_detailed_path.replace('.svg', ''), format='svg', cleanup=True)
        files_created.append(svg_detailed_path)
    except Exception as e:
        print(f"SVG rendering failed for detailed DAG: {e}")
    
    # Also create a text summary of the DAG structure
    summary_path = '../outputs/2025-12-04-16-15-16/dag_structure_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("LLM Deployment DAG Structure Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("Model Configuration:\n")
        f.write(f"- Total Layers: {total_layers}\n")
        f.write(f"- Hidden Size: {hidden_size}\n")
        f.write(f"- Attention Heads: {attention_heads}\n")
        f.write(f"- Total Experts: {total_experts}\n\n")
        f.write("Parallel Configuration:\n")
        f.write(f"- Tensor Parallel (TP): {tp_size} GPUs per group\n")
        f.write(f"- Pipeline Parallel (PP): {pp_size} stages\n")
        f.write(f"- Expert Parallel (EP): {ep_size} groups\n")
        f.write(f"- Data Parallel (DP): {dp_size} replicas\n\n")
        f.write("Node Types:\n")
        f.write("- Rectangle: Computation operations\n")
        f.write("- Ellipse: Communication operations\n")
        f.write("- Parallelogram: Routing/Aggregation operations\n\n")
        f.write("Communication Types:\n")
        f.write("- Solid lines: Data flow within GPU\n")
        f.write("- Dashed lines: Inter-GPU communication\n")
        f.write("- Red dashed: Tensor Parallel All-Reduce\n")
        f.write("- Blue dashed: Expert Parallel Routing\n")
        f.write("- Green dashed: Data Parallel Gradient Sync\n")
    
    files_created.append(summary_path)
    
    print(f"DAG files created successfully!")
    for file_path in files_created:
        print(f"Created: {file_path}")
    
    return files_created

if __name__ == "__main__":
    result = main()
    print(result)