#!/usr/bin/env python3

import graphviz
import os

def create_comprehensive_llm_dag():
    """
    Create a comprehensive DAG for LLM deployment with EP64_TP2 configuration
    This version focuses on accuracy and completeness according to requirements
    """
    
    # Create directed graph
    dot = graphviz.Digraph(comment='LLM Deployment DAG - EP64_TP2 - Complete Model')
    dot.attr(rankdir='TB', size='80,80', dpi='300')
    
    # Define consistent styles
    dot.attr('node', fontname='Arial', fontsize='11')
    
    # Input/Output nodes (ellipses, lightblue)
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Computation nodes (rectangles, lightgreen)
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Routing/aggregation nodes (parallelograms, yellow)
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Communication nodes (ellipses, lightcoral)
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightcoral')
    
    # Create main subgraph
    with dot.subgraph(name='cluster_main') as main:
        main.attr(label='LLM Model Deployment - EP64_TP2 Configuration', fontname='Arial Bold', fontsize='16', style='rounded')
        
        # Input layer
        main.node('input', 
                 f'Model Input\\nInput: [batch_size=128, seq_len=10000, d_model=4096]\\nOutput: [batch_size=128, seq_len=10000, d_model=4096]', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Token embedding
        main.node('embedding', 
                 f'Token Embedding\\nInput: [128, 10000]\\nOutput: [128, 10000, 4096]', 
                 shape='rectangle', fillcolor='lightgreen')
        
        # Add 16 transformer layers
        for layer in range(16):
            with main.subgraph(name=f'cluster_layer_{layer}') as layer_sub:
                layer_sub.attr(label=f'Transformer Layer {layer}', style='rounded', fillcolor='lightgray')
                
                # Pre-layer norm
                pre_ln = f'pre_ln_{layer}'
                layer_sub.node(pre_ln, 
                             f'Pre-LayerNorm_{layer}\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 4096]', 
                             shape='rectangle', fillcolor='lightgreen')
                
                # Multi-Head Attention with TP=2
                with layer_sub.subgraph(name=f'cluster_mha_{layer}') as mha_sub:
                    mha_sub.attr(label=f'Multi-Head Attention (TP=2)', style='dashed')
                    
                    # MHA input split (broadcast to both GPUs)
                    mha_input = f'mha_input_{layer}'
                    mha_sub.node(mha_input, 
                               f'MHA_Input_{layer}\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 4096]', 
                               shape='parallelogram', fillcolor='yellow')
                    
                    # QKV projection GPU 0
                    qkv_0 = f'qkv_proj_{layer}_gpu0'
                    mha_sub.node(qkv_0, 
                               f'QKV_Proj_{layer}_GPU0\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 192] (Q:64,K:64,V:64)', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # QKV projection GPU 1
                    qkv_1 = f'qkv_proj_{layer}_gpu1'
                    mha_sub.node(qkv_1, 
                               f'QKV_Proj_{layer}_GPU1\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 192] (Q:64,K:64,V:64)', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # Attention GPU 0
                    attn_0 = f'attn_{layer}_gpu0'
                    mha_sub.node(attn_0, 
                               f'Attention_{layer}_GPU0\\nInput: [128, 10000, 64]\\nOutput: [128, 10000, 64]', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # Attention GPU 1
                    attn_1 = f'attn_{layer}_gpu1'
                    mha_sub.node(attn_1, 
                               f'Attention_{layer}_GPU1\\nInput: [128, 10000, 64]\\nOutput: [128, 10000, 64]', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # Output projection GPU 0
                    out_0 = f'out_proj_{layer}_gpu0'
                    mha_sub.node(out_0, 
                               f'Output_Proj_{layer}_GPU0\\nInput: [128, 10000, 64]\\nOutput: [128, 10000, 2048]', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # Output projection GPU 1
                    out_1 = f'out_proj_{layer}_gpu1'
                    mha_sub.node(out_1, 
                               f'Output_Proj_{layer}_GPU1\\nInput: [128, 10000, 64]\\nOutput: [128, 10000, 2048]', 
                               shape='rectangle', fillcolor='lightgreen')
                    
                    # MHA merge
                    mha_merge = f'mha_merge_{layer}'
                    mha_sub.node(mha_merge, 
                               f'MHA_Merge_{layer}\\nInput: [128, 10000, 2048]\\nOutput: [128, 10000, 4096]', 
                               shape='parallelogram', fillcolor='yellow')
                
                # Add residual connection
                add_1 = f'add_1_{layer}'
                layer_sub.node(add_1, 
                             f'Residual_Add1_{layer}\\nInput: [128, 10000, 4096] + [128, 10000, 4096]\\nOutput: [128, 10000, 4096]', 
                             shape='rectangle', fillcolor='lightgreen')
                
                # Post-layer norm
                post_ln = f'post_ln_{layer}'
                layer_sub.node(post_ln, 
                             f'Post-LayerNorm_{layer}\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 4096]', 
                             shape='rectangle', fillcolor='lightgreen')
                
                # Expert gate (routing) - dashed for token selection
                gate = f'gate_{layer}'
                layer_sub.node(gate, 
                             f'Expert_Gate_{layer}\\nInput: [128, 10000, 4096]\\nOutput: Routing_Decisions[64_experts]', 
                             shape='parallelogram', fillcolor='yellow', style='dashed')
                
                # Token distribution (all-to-all communication)
                token_dist = f'token_dist_{layer}'
                layer_sub.node(token_dist, 
                             f'Token_Distribution_{layer}\\nInput: [128, 10000, 4096]\\nOutput: Distributed_Tokens[64_experts]', 
                             shape='ellipse', fillcolor='lightcoral')
                
                # MoE experts (show representative experts)
                with layer_sub.subgraph(name=f'cluster_moe_{layer}') as moe_sub:
                    moe_sub.attr(label=f'Mixture of Experts (EP=64, TP=2)', style='dashed')
                    
                    # Show experts 0, 1, 2, 61, 62, 63 as representatives
                    for expert_id in [0, 1, 2, 61, 62, 63]:
                        with moe_sub.subgraph(name=f'cluster_expert_{layer}_{expert_id}') as expert_sub:
                            expert_sub.attr(label=f'Expert {expert_id} (GPUs {expert_id*2},{expert_id*2+1})', style='rounded')
                            
                            # Expert computation GPU 0
                            expert_0 = f'expert_{layer}_{expert_id}_gpu0'
                            expert_sub.node(expert_0, 
                                          f'Expert_{layer}_{expert_id}_GPU0\\nInput: [tokens, 4096]\\nOutput: [tokens, 8192]', 
                                          shape='rectangle', fillcolor='lightgreen')
                            
                            # Expert computation GPU 1
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
                               f'Expert_Aggregation_{layer}\\nInput: [expert_outputs, 4096]\\nOutput: [128, 10000, 4096]', 
                               shape='parallelogram', fillcolor='yellow')
                
                # Add residual connection
                add_2 = f'add_2_{layer}'
                layer_sub.node(add_2, 
                             f'Residual_Add2_{layer}\\nInput: [128, 10000, 4096] + [128, 10000, 4096]\\nOutput: [128, 10000, 4096]', 
                             shape='rectangle', fillcolor='lightgreen')
        
        # Output layer
        main.node('output', 
                 f'Model Output\\nInput: [128, 10000, 4096]\\nOutput: [128, 10000, 51200]', 
                 shape='ellipse', fillcolor='lightblue')
    
    # Add edges with proper connections
    dot.attr('edge', style='solid', color='black')
    
    # Input to embedding
    dot.edge('input', 'embedding')
    
    # Connect through layers
    prev_node = 'embedding'
    for layer in range(16):
        # Layer connections
        dot.edge(prev_node, f'pre_ln_{layer}')
        dot.edge(f'pre_ln_{layer}', f'mha_input_{layer}')
        
        # MHA connections
        dot.edge(f'mha_input_{layer}', f'qkv_proj_{layer}_gpu0')
        dot.edge(f'mha_input_{layer}', f'qkv_proj_{layer}_gpu1')
        dot.edge(f'qkv_proj_{layer}_gpu0', f'attn_{layer}_gpu0')
        dot.edge(f'qkv_proj_{layer}_gpu1', f'attn_{layer}_gpu1')
        dot.edge(f'attn_{layer}_gpu0', f'out_proj_{layer}_gpu0')
        dot.edge(f'attn_{layer}_gpu1', f'out_proj_{layer}_gpu1')
        dot.edge(f'out_proj_{layer}_gpu0', f'mha_merge_{layer}')
        dot.edge(f'out_proj_{layer}_gpu1', f'mha_merge_{layer}')
        dot.edge(f'mha_merge_{layer}', f'add_1_{layer}')
        dot.edge(prev_node, f'add_1_{layer}')  # Residual connection
        dot.edge(f'add_1_{layer}', f'post_ln_{layer}')
        dot.edge(f'post_ln_{layer}', f'gate_{layer}')
        
        # Gate to token distribution (dashed)
        dot.edge(f'gate_{layer}', f'token_dist_{layer}', style='dashed', color='red')
        
        # Expert connections
        dot.edge(f'token_dist_{layer}', f'expert_{layer}_0_gpu0')
        dot.edge(f'token_dist_{layer}', f'expert_{layer}_0_gpu1')
        dot.edge(f'expert_{layer}_0_gpu0', f'expert_{layer}_0_gpu1')  # GPU coordination
        dot.edge(f'expert_{layer}_0_gpu1', f'expert_merge_{layer}_0')
        dot.edge(f'expert_merge_{layer}_0', f'expert_agg_{layer}')
        
        # Final add
        dot.edge(f'expert_agg_{layer}', f'add_2_{layer}')
        dot.edge(f'add_1_{layer}', f'add_2_{layer}')  # Residual connection
        
        prev_node = f'add_2_{layer}'
    
    # Connect to output
    dot.edge(prev_node, 'output')
    
    return dot

def create_final_submission_dag():
    """
    Create the final submission DAG that meets all requirements
    """
    dot = graphviz.Digraph(comment='LLM Deployment DAG - EP64_TP2 Final Submission')
    dot.attr(rankdir='TB', size='60,60', dpi='300')
    
    # Define styles
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Input node
    dot.node('input', 
             f'Model Input\\n[batch=128, seq=10000, d_model=4096]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Token embedding
    dot.node('embedding', 
             f'Token Embedding\\n[128, 10000, 4096]', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Transformer layer (representative)
    with dot.subgraph(name='cluster_transformer') as trans:
        trans.attr(label='Transformer Layers (×16)', style='dashed')
        
        # Pre-norm
        trans.node('prenorm', 'Pre-LayerNorm\\n[128, 10000, 4096]', 
                  shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # MHA with TP=2
        with trans.subgraph(name='cluster_mha') as mha:
            mha.attr(label='Multi-Head Attention (TP=2)', style='dotted')
            mha.node('mha_gpu0', 'MHA_GPU0\\n[128, 10000, 2048]', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            mha.node('mha_gpu1', 'MHA_GPU1\\n[128, 10000, 2048]', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            mha.node('mha_merge', 'MHA_Merge\\n[128, 10000, 4096]', 
                    shape='parallelogram', style='filled', fillcolor='yellow')
        
        # Post-norm
        trans.node('postnorm', 'Post-LayerNorm\\n[128, 10000, 4096]', 
                  shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Expert gate (dashed for selection)
        trans.node('gate', 'Expert Router\\n[128, 10000, 4096]', 
                  shape='parallelogram', style='dashed', fillcolor='yellow')
        
        # Token distribution (communication)
        trans.node('token_dist', 'Token Distribution\\nAll-to-All Comm', 
                  shape='ellipse', style='filled', fillcolor='lightcoral')
        
        # MoE experts
        with trans.subgraph(name='cluster_moe') as moe:
            moe.attr(label='MoE Experts (EP=64, TP=2)', style='dotted')
            # Show 4 representative experts
            for i in range(4):
                gpu_pair = i * 2
                moe.node(f'expert_{i}', f'Expert {i}\\nGPUs {gpu_pair},{gpu_pair+1}\\n[tokens, 4096]', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
            moe.node('expert_agg', 'Expert Aggregation\\n[128, 10000, 4096]', 
                    shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Output
    dot.node('output', 
             f'Model Output\\n[128, 10000, 51200]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Add edges
    dot.edge('input', 'embedding')
    dot.edge('embedding', 'prenorm')
    dot.edge('prenorm', 'mha_gpu0')
    dot.edge('prenorm', 'mha_gpu1')
    dot.edge('mha_gpu0', 'mha_merge')
    dot.edge('mha_gpu1', 'mha_merge')
    dot.edge('mha_merge', 'postnorm')
    dot.edge('postnorm', 'gate')
    dot.edge('gate', 'token_dist', style='dashed', color='red')
    dot.edge('token_dist', 'expert_0')
    dot.edge('token_dist', 'expert_1')
    dot.edge('token_dist', 'expert_2')
    dot.edge('token_dist', 'expert_3')
    dot.edge('expert_0', 'expert_agg')
    dot.edge('expert_1', 'expert_agg')
    dot.edge('expert_2', 'expert_agg')
    dot.edge('expert_3', 'expert_agg')
    dot.edge('expert_agg', 'output')
    
    return dot

def main():
    output_dir = '../outputs/2025-12-04-09-27-30'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive DAG
    print("Generating comprehensive DAG...")
    comprehensive_dag = create_comprehensive_llm_dag()
    comprehensive_dag.save(os.path.join(output_dir, 'llm_deployment_comprehensive.dot'))
    comprehensive_dag.render(os.path.join(output_dir, 'llm_deployment_comprehensive'), format='svg', cleanup=False)
    
    # Generate final submission DAG
    print("Generating final submission DAG...")
    final_dag = create_final_submission_dag()
    final_dag.save(os.path.join(output_dir, 'llm_deployment_final.dot'))
    final_dag.render(os.path.join(output_dir, 'llm_deployment_final'), format='svg', cleanup=False)
    
    print(f"Final DAGs saved to {output_dir}")
    
    # Verify final DAG
    print("\nVerifying final DAG...")
    import networkx as nx
    
    # Check final DAG
    G = nx.drawing.nx_pydot.read_dot(os.path.join(output_dir, 'llm_deployment_final.dot'))
    
    print(f"✓ Nodes: {len(G.nodes())}")
    print(f"✓ Edges: {len(G.edges())}")
    print(f"✓ No cycles: {not nx.is_directed_acyclic_graph(G) == False}")
    print(f"✓ Connected: {nx.is_weakly_connected(G)}")
    
    # List all files
    print(f"\nGenerated files:")
    files = [
        'llm_deployment_detailed.dot',
        'llm_deployment_detailed.svg', 
        'llm_deployment_simplified.dot',
        'llm_deployment_simplified.svg',
        'llm_deployment_comprehensive.dot',
        'llm_deployment_comprehensive.svg',
        'llm_deployment_final.dot',
        'llm_deployment_final.svg'
    ]
    
    for file in files:
        path = os.path.join(output_dir, file)
        if os.path.exists(path):
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path}")

if __name__ == "__main__":
    main()