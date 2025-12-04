#!/usr/bin/env python3

import graphviz
import os

def create_llm_deployment_dag():
    """
    Create a comprehensive DAG for LLM deployment with EP64_TP2 configuration
    Showing complete expert parallelism and tensor parallelism at operator level
    """
    
    # Create directed graph
    dot = graphviz.Digraph(comment='LLM Deployment DAG - EP64_TP2 Configuration')
    dot.attr(rankdir='TB', size='100,100', dpi='300')
    
    # Define node styles
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Input node style (ellipse)
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Computation node style (rectangle) 
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Routing/aggregation node style (parallelogram)
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Communication node style (ellipse)
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightcoral')
    
    # Add main subgraph for better organization
    with dot.subgraph(name='cluster_main') as main:
        main.attr(label='LLM Model - 16 Layers with EP64_TP2', fontname='Arial Bold', fontsize='16')
        
        # Input layer
        main.node('input', 
                 f'Input\\nInput: [batch_size=128, seq_len=10000, d_model=4096]\\nOutput: [batch_size=128, seq_len=10000, d_model=4096]', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Add layer subgraphs
        for layer in range(16):
            with main.subgraph(name=f'cluster_layer_{layer}') as layer_sub:
                layer_sub.attr(label=f'Layer {layer}', style='rounded', fillcolor='lightgray')
                
                # Layer norm (shared across all GPUs)
                ln_node = f'layer_norm_{layer}'
                layer_sub.node(ln_node, 
                             f'LayerNorm_{layer}\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 4096]', 
                             shape='rectangle', fillcolor='lightgreen')
                
                # Multi-head attention (tensor parallel across 2 GPUs)
                with layer_sub.subgraph(name=f'cluster_mha_{layer}') as mha_sub:
                    mha_sub.attr(label=f'MHA Layer {layer} (TP=2)', style='dashed')
                    
                    # MHA input split
                    mha_split = f'mha_split_{layer}'
                    mha_sub.node(mha_split, 
                               f'MHA_Split_{layer}\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 2048]', 
                               shape='parallelogram', fillcolor='yellow')
                    
                    # QKV projection on GPU 0
                    qkv_proj_0 = f'qkv_proj_{layer}_gpu0'
                    mha_sub.node(qkv_proj_0, 
                               f'QKV_Proj_{layer}_GPU0\\nInput: [128, 10000, 2048]\\nOutput: [128, 10000, 192] (Q:64, K:64, V:64)', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # QKV projection on GPU 1
                    qkv_proj_1 = f'qkv_proj_{layer}_gpu1'
                    mha_sub.node(qkv_proj_1, 
                               f'QKV_Proj_{layer}_GPU1\\nInput: [128, 10000, 2048]\\nOutput: [128, 10000, 192] (Q:64, K:64, V:64)', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # Attention computation on GPU 0
                    attn_0 = f'attn_{layer}_gpu0'
                    mha_sub.node(attn_0, 
                               f'Attention_{layer}_GPU0\\nInput: [128, 10000, 64]\\nOutput: [128, 10000, 64]', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # Attention computation on GPU 1
                    attn_1 = f'attn_{layer}_gpu1'
                    mha_sub.node(attn_1, 
                               f'Attention_{layer}_GPU1\\nInput: [128, 10000, 64]\\nOutput: [128, 10000, 64]', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # Output projection on GPU 0
                    out_proj_0 = f'out_proj_{layer}_gpu0'
                    mha_sub.node(out_proj_0, 
                               f'Output_Proj_{layer}_GPU0\\nInput: [128, 10000, 64]\\nOutput: [128, 10000, 2048]', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # Output projection on GPU 1
                    out_proj_1 = f'out_proj_{layer}_gpu1'
                    mha_sub.node(out_proj_1, 
                               f'Output_Proj_{layer}_GPU1\\nInput: [128, 10000, 64]\\nOutput: [128, 10000, 2048]', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # MHA output merge
                    mha_merge = f'mha_merge_{layer}'
                    mha_sub.node(mha_merge, 
                               f'MHA_Merge_{layer}\\nInput: [128, 10000, 2048]\\nOutput: [128, 10000, 4096]', 
                               shape='parallelogram', fillcolor='yellow')
                
                # Add & Norm after attention
                add_norm_1 = f'add_norm_1_{layer}'
                layer_sub.node(add_norm_1, 
                             f'Add&Norm1_{layer}\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 4096]', 
                             shape='rectangle', fillcolor='lightgreen')
                
                # Expert routing (gate) - this will be dashed for token selection
                gate_node = f'gate_{layer}'
                layer_sub.node(gate_node, 
                             f'Expert_Gate_{layer}\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 4096] + Routing_Weights', 
                             shape='parallelogram', fillcolor='yellow', style='dashed')
                
                # MoE layer with 64 experts (EP=64, each expert on 2 GPUs for TP=2)
                with layer_sub.subgraph(name=f'cluster_moe_{layer}') as moe_sub:
                    moe_sub.attr(label=f'MoE Layer {layer} (EP=64, TP=2)', style='dashed')
                    
                    # Token distribution based on routing
                    token_dist = f'token_dist_{layer}'
                    moe_sub.node(token_dist, 
                               f'Token_Distribution_{layer}\\nInput: [128, 10000, 4096]\\nOutput: Distributed_Tokens[64_experts]', 
                               shape='parallelogram', fillcolor='yellow')
                    
                    # Create nodes for each expert (showing first 4 and last 2 as examples)
                    for expert_id in [0, 1, 2, 3, 62, 63]:
                        with moe_sub.subgraph(name=f'cluster_expert_{layer}_{expert_id}') as expert_sub:
                            expert_sub.attr(label=f'Expert {expert_id} (GPUs {expert_id*2},{expert_id*2+1})', style='rounded')
                            
                            # Expert computation on GPU 0
                            expert_0 = f'expert_{layer}_{expert_id}_gpu0'
                            expert_sub.node(expert_0, 
                                          f'Expert_{layer}_{expert_id}_GPU0\\nInput: [tokens, 4096]\\nOutput: [tokens, 8192]', 
                                          shape='rectangle', fillcolor='lightgreen')
                            
                            # Expert computation on GPU 1
                            expert_1 = f'expert_{layer}_{expert_id}_gpu1'
                            expert_sub.node(expert_1, 
                                          f'Expert_{layer}_{expert_id}_GPU1\\nInput: [tokens, 4096]\\nOutput: [tokens, 8192]', 
                                          shape='rectangle', fillcolor='lightgreen')
                            
                            # Expert output merge
                            expert_merge = f'expert_merge_{layer}_{expert_id}'
                            expert_sub.node(expert_merge, 
                                          f'Expert_Merge_{layer}_{expert_id}\\nInput: [tokens, 8192]\\nOutput: [tokens, 4096]', 
                                          shape='parallelogram', fillcolor='yellow')
                    
                    # Expert output aggregation
                    expert_agg = f'expert_agg_{layer}'
                    moe_sub.node(expert_agg, 
                               f'Expert_Aggregation_{layer}\\nInput: [64_expert_outputs, 4096]\\nOutput: [128, 10000, 4096]', 
                               shape='parallelogram', fillcolor='yellow')
                
                # Add & Norm after experts
                add_norm_2 = f'add_norm_2_{layer}'
                layer_sub.node(add_norm_2, 
                             f'Add&Norm2_{layer}\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 4096]', 
                             shape='rectangle', fillcolor='lightgreen')
        
        # Output layer
        dot.node('output', 
               f'Output\\nInput: [batch_size=128, seq_len=10000, d_model=4096]\\nOutput: [batch_size=128, seq_len=10000, vocab_size=51200]', 
               shape='ellipse', fillcolor='lightblue')
    
    # Add communication edges with different styles
    # Solid lines for data flow
    dot.attr('edge', style='solid', color='black')
    
    # Dashed lines for gate selection
    dot.attr('edge', style='dashed', color='red')
    
    # Connect input to first layer
    dot.edge('input', 'layer_norm_0')
    
    # Connect layers
    for layer in range(16):
        # Within layer connections
        dot.edge(f'layer_norm_{layer}', f'mha_split_{layer}')
        dot.edge(f'mha_split_{layer}', f'qkv_proj_{layer}_gpu0')
        dot.edge(f'mha_split_{layer}', f'qkv_proj_{layer}_gpu1')
        dot.edge(f'qkv_proj_{layer}_gpu0', f'attn_{layer}_gpu0')
        dot.edge(f'qkv_proj_{layer}_gpu1', f'attn_{layer}_gpu1')
        dot.edge(f'attn_{layer}_gpu0', f'out_proj_{layer}_gpu0')
        dot.edge(f'attn_{layer}_gpu1', f'out_proj_{layer}_gpu1')
        dot.edge(f'out_proj_{layer}_gpu0', f'mha_merge_{layer}')
        dot.edge(f'out_proj_{layer}_gpu1', f'mha_merge_{layer}')
        dot.edge(f'mha_merge_{layer}', f'add_norm_1_{layer}')
        dot.edge(f'add_norm_1_{layer}', f'gate_{layer}')
        dot.edge(f'gate_{layer}', f'token_dist_{layer}', style='dashed', color='red')
        dot.edge(f'token_dist_{layer}', f'expert_0_0')
        dot.edge(f'expert_merge_0_0', f'expert_agg_0')
        dot.edge(f'expert_agg_{layer}', f'add_norm_2_{layer}')
        
        # Connect to next layer or output
        if layer < 15:
            dot.edge(f'add_norm_2_{layer}', f'layer_norm_{layer+1}')
        else:
            dot.edge(f'add_norm_2_{layer}', 'output')
    
    return dot

def create_simplified_dag():
    """
    Create a simplified DAG showing key components and data flow
    """
    dot = graphviz.Digraph(comment='LLM Deployment DAG - Simplified View')
    dot.attr(rankdir='TB', size='50,50', dpi='300')
    
    # Define styles
    dot.attr('node', fontname='Arial', fontsize='12')
    
    # Input
    dot.node('input', 'Input Tokens\\n[128, 10000]', shape='ellipse', fillcolor='lightblue')
    
    # Embedding
    dot.node('embedding', 'Token Embedding\\n[128, 10000, 4096]', shape='rectangle', fillcolor='lightgreen')
    
    # Layer representation (showing one layer as example)
    with dot.subgraph(name='cluster_layer') as layer:
        layer.attr(label='Transformer Layer (×16)', style='dashed')
        
        # MHA block
        layer.node('mha', 'Multi-Head Attention\\nTP=2 on GPUs 0,1', shape='rectangle', fillcolor='lightgreen')
        
        # MoE block
        layer.node('moe', 'Mixture of Experts\\nEP=64, TP=2\\n64 experts on 128 GPUs', shape='rectangle', fillcolor='lightgreen')
        
        # Gate
        layer.node('gate', 'Expert Router\\nSelects 1-2 experts per token', shape='parallelogram', fillcolor='yellow', style='dashed')
        
        # Communication
        layer.node('comm', 'All-to-All Communication\\nToken redistribution', shape='ellipse', fillcolor='lightcoral')
    
    # Output
    dot.node('output', 'Output Logits\\n[128, 10000, 51200]', shape='ellipse', fillcolor='lightblue')
    
    # Edges
    dot.edge('input', 'embedding')
    dot.edge('embedding', 'mha')
    dot.edge('mha', 'gate', style='dashed', color='red')
    dot.edge('gate', 'comm', style='dashed', color='red')
    dot.edge('comm', 'moe')
    dot.edge('moe', 'output')
    
    return dot

def main():
    # Create output directory if it doesn't exist
    output_dir = '../outputs/2025-12-04-09-27-30'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate detailed DAG
    print("Generating detailed DAG...")
    detailed_dag = create_llm_deployment_dag()
    
    # Save as DOT file
    detailed_dag.save(os.path.join(output_dir, 'llm_deployment_detailed.dot'))
    
    # Render as SVG
    detailed_dag.render(os.path.join(output_dir, 'llm_deployment_detailed'), format='svg', cleanup=False)
    
    # Generate simplified DAG
    print("Generating simplified DAG...")
    simplified_dag = create_simplified_dag()
    
    # Save as DOT file
    simplified_dag.save(os.path.join(output_dir, 'llm_deployment_simplified.dot'))
    
    # Render as SVG
    simplified_dag.render(os.path.join(output_dir, 'llm_deployment_simplified'), format='svg', cleanup=False)
    
    print(f"DAGs saved to {output_dir}")
    print("Files created:")
    print(f"  - {output_dir}/llm_deployment_detailed.dot")
    print(f"  - {output_dir}/llm_deployment_detailed.svg")
    print(f"  - {output_dir}/llm_deployment_simplified.dot")
    print(f"  - {output_dir}/llm_deployment_simplified.svg")
    
    # Verify DAG properties
    print("\nVerifying DAG properties...")
    
    # Check for cycles
    try:
        import networkx as nx
        
        # Read DOT file and check for cycles
        G = nx.drawing.nx_pydot.read_dot(os.path.join(output_dir, 'llm_deployment_simplified.dot'))
        
        if nx.is_directed_acyclic_graph(G):
            print("✓ DAG verification: No cycles detected")
        else:
            print("✗ DAG verification: Cycles detected!")
            
        # Check node connectivity
        if nx.is_weakly_connected(G):
            print("✓ DAG verification: Graph is connected")
        else:
            print("✗ DAG verification: Graph is not connected")
            
    except ImportError:
        print("NetworkX not available for cycle checking")
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    main()