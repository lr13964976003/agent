#!/usr/bin/env python3
"""
LLM Deployment DAG Generator for 30B MoE Model
Generates a comprehensive DAG showing all parallel strategies with GPU boundaries
"""

import graphviz
import json

def main():
    """Main function to generate the DAG"""
    print("Generating LLM Deployment DAG...")
    
    # Create the main graph
    dot = graphviz.Digraph(comment='LLM 30B MoE Deployment DAG')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fontsize='9')
    dot.attr('edge', fontsize='8')
    
    # Model configuration
    model_config = {
        'total_parameters': '30B',
        'layers': 16,
        'experts_per_layer': 64,
        'hidden_size': 1024,
        'ffn_hidden_size': 2048,
        'num_heads': 16,
        'head_dim': 64,
        'batch_size': 128,
        'sequence_length': 1024,
        'precision': 'FP16'
    }
    
    # Parallel configuration
    parallel_config = {
        'tensor_parallel_size': 8,
        'pipeline_parallel_size': 4,
        'expert_parallel_size': 8,
        'data_parallel_size': 2,
        'total_gpus': 512
    }
    
    # Create a comprehensive DAG showing all key components
    
    # Input node
    dot.node('input', 'Input\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='ellipse', style='filled', fillcolor='lightpink')
    
    # Token embedding (computation)
    dot.node('token_embedding', 'Token Embedding\\nVocab:32K→Hidden:1024\\n[batch_size=128, seq_len=1024, heads=1, d_k=1]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('input', 'token_embedding')
    
    # Position embedding (computation)
    dot.node('position_embedding', 'Position Embedding\\nSeq:1024→Hidden:1024\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('token_embedding', 'position_embedding')
    
    # Layer normalization (computation)
    dot.node('layer_norm_1', 'Layer Norm\\nHidden:1024\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('position_embedding', 'layer_norm_1')
    
    # Multi-head attention components (tensor parallel)
    dot.node('q_projection', 'Q Projection\\nTP: Hidden:1024→Heads:2×64\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=2, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('layer_norm_1', 'q_projection')
    
    dot.node('k_projection', 'K Projection\\nTP: Hidden:1024→Heads:2×64\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=2, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('layer_norm_1', 'k_projection')
    
    dot.node('v_projection', 'V Projection\\nTP: Hidden:1024→Heads:2×64\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=2, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('layer_norm_1', 'v_projection')
    
    # Attention computation (computation)
    dot.node('attention', 'Attention\\nQK^T V: Softmax\\n[batch_size=128, seq_len=1024, heads=2, d_k=64]\\n[batch_size=128, seq_len=1024, heads=2, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('q_projection', 'attention')
    dot.edge('k_projection', 'attention')
    dot.edge('v_projection', 'attention')
    
    # Attention output projection (tensor parallel)
    dot.node('attention_output', 'Attention Output\\nTP: Heads:2×64→Hidden:1024\\n[batch_size=128, seq_len=1024, heads=2, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('attention', 'attention_output')
    
    # Add attention all-reduce across tensor groups (communication)
    dot.node('attention_allreduce', 'All-Reduce\\nAttention Output\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='ellipse', style='dashed', fillcolor='lightyellow')
    dot.edge('attention_output', 'attention_allreduce')
    
    # Residual connection (computation)
    dot.node('residual_1', 'Residual Add\\nHidden:1024\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('attention_allreduce', 'residual_1')
    dot.edge('position_embedding', 'residual_1')  # Skip connection
    
    # Layer normalization 2 (computation)
    dot.node('layer_norm_2', 'Layer Norm\\nHidden:1024\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('residual_1', 'layer_norm_2')
    
    # Expert routing (routing)
    dot.node('expert_router', 'Expert Router\\nTop-K:2 Selection\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='parallelogram', style='filled', fillcolor='lightgreen')
    dot.edge('layer_norm_2', 'expert_router')
    
    # Expert computation (computation)
    for expert_idx in range(4):  # Show 4 experts
        dot.node(f'expert_{expert_idx}', f'Expert {expert_idx}\\nMLP: 1024→2048→1024\\n[batch_size=32, seq_len=1024, heads=16, d_k=64]\\n[batch_size=32, seq_len=1024, heads=16, d_k=64]', 
                 shape='rectangle', style='filled', fillcolor='lightblue')
        dot.edge('expert_router', f'expert_{expert_idx}')
        
        # Expert all-to-all communication (communication)
        dot.node(f'expert_all2all_{expert_idx}', f'All-to-All\\nExpert {expert_idx}\\n[batch_size=32, seq_len=1024, heads=16, d_k=64]\\n[batch_size=32, seq_len=1024, heads=16, d_k=64]', 
                 shape='ellipse', style='dashed', fillcolor='lightyellow')
        dot.edge(f'expert_{expert_idx}', f'expert_all2all_{expert_idx}')
    
    # Expert aggregation (routing)
    dot.node('expert_aggregation', 'Expert Aggregation\\nWeighted Sum\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Connect all expert all-to-all to aggregation
    for expert_idx in range(4):
        dot.edge(f'expert_all2all_{expert_idx}', 'expert_aggregation')
    
    # MLP output projection (tensor parallel)
    dot.node('mlp_output', 'MLP Output\\nTP: Hidden:1024→Hidden:1024\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('expert_aggregation', 'mlp_output')
    
    # Add MLP all-reduce across tensor groups (communication)
    dot.node('mlp_allreduce', 'All-Reduce\\nMLP Output\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='ellipse', style='dashed', fillcolor='lightyellow')
    dot.edge('mlp_output', 'mlp_allreduce')
    
    # Residual connection 2 (computation)
    dot.node('residual_2', 'Residual Add\\nHidden:1024\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('mlp_allreduce', 'residual_2')
    dot.edge('residual_1', 'residual_2')  # Skip connection
    
    # Final layer norm (computation)
    dot.node('final_layer_norm', 'Final Layer Norm\\nHidden:1024\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('residual_2', 'final_layer_norm')
    
    # Output projection (computation)
    dot.node('output_projection', 'Output Projection\\nHidden:1024→Vocab:32K\\n[batch_size=128, seq_len=1024, heads=16, d_k=64]\\n[batch_size=128, seq_len=1024, heads=1, d_k=32000]', 
             shape='rectangle', style='filled', fillcolor='lightblue')
    dot.edge('final_layer_norm', 'output_projection')
    
    # Final output (output)
    dot.node('output', 'Output\\nLogits:32K\\n[batch_size=128, seq_len=1024, heads=1, d_k=32000]\\n[batch_size=128, seq_len=1024, heads=1, d_k=32000]', 
             shape='ellipse', style='filled', fillcolor='orange')
    dot.edge('output_projection', 'output')
    
    # Save DAG in both DOT and SVG formats
    output_dir = "../outputs/2025-12-04-14-18-19"
    
    # Save as DOT file
    dot_file = f"{output_dir}/llm_deployment_dag.dot"
    with open(dot_file, 'w') as f:
        f.write(dot.source)
    
    # Save as SVG file
    svg_file = f"{output_dir}/llm_deployment_dag.svg"
    dot.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")
    
    # Save submission paths
    submission_paths = {
        "dag_dot_file": dot_file,
        "dag_svg_file": svg_file,
        "generated_at": "2025-12-04 14:18:19",
        "model_configuration": model_config,
        "parallel_configuration": parallel_config
    }
    
    with open(f"{output_dir}/final_dag_submission_paths.json", 'w') as f:
        json.dump(submission_paths, f, indent=2)
    
    return dot_file, svg_file

if __name__ == "__main__":
    main()