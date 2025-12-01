#!/usr/bin/env python3
"""
Complete LLM Deployment DAG with full implementation details
Shows all 16 layers, 64 experts, 4 pipeline stages, and 2-way tensor parallelism
"""

import graphviz
from graphviz import Digraph

def create_complete_deployment_dag():
    """Create the most comprehensive DAG showing the complete deployment"""
    dot = Digraph(comment='Complete LLM Deployment DAG - 64 GPUs with Hybrid Parallelism')
    dot.attr(rankdir='TB', size='100,100', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='8')
    dot.attr('edge', fontname='Arial', fontsize='6')
    
    # Define colors for different components
    colors = {
        'input_output': 'white',
        'computation': 'lightgreen',
        'communication': 'lightblue',
        'routing': 'lightyellow',
        'pipeline': 'orange',
        'expert_group': 'lightcyan',
        'layer': 'lightgray'
    }
    
    # Input node
    dot.node('input', 'Model Input\\n\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='box', style='filled', fillcolor=colors['input_output'], 
             fontname='Arial Bold', fontsize='12')
    
    # Create all 4 pipeline stages
    for pipeline_stage in range(4):
        with dot.subgraph(name=f'cluster_pipeline_stage_{pipeline_stage}') as stage_cluster:
            stage_cluster.attr(label=f'Pipeline Stage {pipeline_stage} (Layers {pipeline_stage*4}-{(pipeline_stage+1)*4-1})', 
                              style='rounded,filled', fillcolor=colors['pipeline'], 
                              fontname='Arial Bold', fontsize='10')
            
            # Create all 16 layers (4 per pipeline stage)
            for layer in range(pipeline_stage*4, (pipeline_stage+1)*4):
                with stage_cluster.subgraph(name=f'cluster_layer_{layer}') as layer_cluster:
                    layer_cluster.attr(label=f'Layer {layer}', style='rounded,filled', 
                                      fillcolor=colors['layer'], fontname='Arial Bold')
                    
                    # Create all 8 expert groups (8-way expert parallelism)
                    for expert_group in range(8):
                        gpu_id = (pipeline_stage * 16) + (expert_group * 2)  # Approximate GPU mapping
                        
                        with layer_cluster.subgraph(name=f'cluster_layer{layer}_expert{expert_group}') as expert_cluster:
                            expert_cluster.attr(label=f'Expert Group {expert_group} (GPU {gpu_id})', 
                                               style='rounded,filled', fillcolor=colors['expert_group'])
                            
                            # Expert routing and selection
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_route', 
                                               f'Expert Routing\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
                                               shape='parallelogram', style='filled', fillcolor=colors['routing'])
                            
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_gate', 
                                               f'Expert Gate\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, num_experts=8]', 
                                               shape='parallelogram', style='filled', fillcolor=colors['routing'])
                            
                            # All-to-all communication for expert assignment
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_a2a', 
                                               f'All-to-All Comm\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
                                               shape='ellipse', style='filled', fillcolor=colors['communication'])
                            
                            # Multi-head attention with tensor parallelism (2-way)
                            # Split attention heads
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_attn_split', 
                                               f'Attention Head Split\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
                                               shape='parallelogram', style='filled', fillcolor=colors['routing'])
                            
                            # Q, K, V projections (column parallel)
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_q_proj', 
                                               f'Q Projection (Col-Par)\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
                                               shape='rectangle', style='filled', fillcolor=colors['computation'])
                            
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_k_proj', 
                                               f'K Projection (Col-Par)\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
                                               shape='rectangle', style='filled', fillcolor=colors['computation'])
                            
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_v_proj', 
                                               f'V Projection (Col-Par)\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
                                               shape='rectangle', style='filled', fillcolor=colors['computation'])
                            
                            # Attention computation
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_attn', 
                                               f'Multi-Head Attention\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
                                               shape='rectangle', style='filled', fillcolor=colors['computation'])
                            
                            # Attention output (row parallel)
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_attn_out', 
                                               f'Attention Output (Row-Par)\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
                                               shape='rectangle', style='filled', fillcolor=colors['computation'])
                            
                            # Attention all-reduce
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_attn_ar', 
                                               f'Attention All-Reduce\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
                                               shape='ellipse', style='filled', fillcolor=colors['communication'])
                            
                            # Attention residual and layer norm
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_attn_res', 
                                               f'Attention + Residual\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
                                               shape='parallelogram', style='filled', fillcolor=colors['routing'])
                            
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_ln1', 
                                               f'Layer Norm 1\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
                                               shape='rectangle', style='filled', fillcolor=colors['computation'])
                            
                            # MLP with tensor parallelism
                            # First linear (column parallel)
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_mlp1', 
                                               f'MLP Linear 1 (Col-Par)\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, ffn_hidden=1024]', 
                                               shape='rectangle', style='filled', fillcolor=colors['computation'])
                            
                            # GELU activation
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_gelu', 
                                               f'GELU Activation\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, ffn_hidden=1024]', 
                                               shape='rectangle', style='filled', fillcolor=colors['computation'])
                            
                            # Second linear (row parallel)
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_mlp2', 
                                               f'MLP Linear 2 (Row-Par)\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
                                               shape='rectangle', style='filled', fillcolor=colors['computation'])
                            
                            # MLP all-reduce
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_mlp_ar', 
                                               f'MLP All-Reduce\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
                                               shape='ellipse', style='filled', fillcolor=colors['communication'])
                            
                            # MLP residual and layer norm
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_mlp_res', 
                                               f'MLP + Residual\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
                                               shape='parallelogram', style='filled', fillcolor=colors['routing'])
                            
                            expert_cluster.node(f'layer{layer}_expert{expert_group}_ln2', 
                                               f'Layer Norm 2\\nL{layer}_E{expert_group}\\nGPU:{gpu_id}\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
                                               shape='rectangle', style='filled', fillcolor=colors['computation'])
    
    # Output node
    dot.node('output', 'Model Output\\n\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='box', style='filled', fillcolor=colors['input_output'], 
             fontname='Arial Bold', fontsize='12')
    
    # Add connections (simplified for readability)
    # Input to first layer
    dot.edge('input', 'layer0_expert0_route')
    
    # Connect within first expert group of first layer
    dot.edge('layer0_expert0_route', 'layer0_expert0_gate')
    dot.edge('layer0_expert0_gate', 'layer0_expert0_a2a')
    dot.edge('layer0_expert0_a2a', 'layer0_expert0_attn_split')
    dot.edge('layer0_expert0_attn_split', 'layer0_expert0_q_proj')
    dot.edge('layer0_expert0_attn_split', 'layer0_expert0_k_proj')
    dot.edge('layer0_expert0_attn_split', 'layer0_expert0_v_proj')
    dot.edge('layer0_expert0_q_proj', 'layer0_expert0_attn')
    dot.edge('layer0_expert0_k_proj', 'layer0_expert0_attn')
    dot.edge('layer0_expert0_v_proj', 'layer0_expert0_attn')
    dot.edge('layer0_expert0_attn', 'layer0_expert0_attn_out')
    dot.edge('layer0_expert0_attn_out', 'layer0_expert0_attn_ar')
    dot.edge('layer0_expert0_attn_ar', 'layer0_expert0_attn_res')
    dot.edge('layer0_expert0_attn_res', 'layer0_expert0_ln1')
    dot.edge('layer0_expert0_ln1', 'layer0_expert0_mlp1')
    dot.edge('layer0_expert0_mlp1', 'layer0_expert0_gelu')
    dot.edge('layer0_expert0_gelu', 'layer0_expert0_mlp2')
    dot.edge('layer0_expert0_mlp2', 'layer0_expert0_mlp_ar')
    dot.edge('layer0_expert0_mlp_ar', 'layer0_expert0_mlp_res')
    dot.edge('layer0_expert0_mlp_res', 'layer0_expert0_ln2')
    dot.edge('layer0_expert0_ln2', 'output')
    
    return dot

def create_gpu_mapping_dag():
    """Create DAG showing GPU mapping and load balancing"""
    dot = Digraph(comment='GPU Mapping and Load Balancing DAG')
    dot.attr(rankdir='LR', size='40,30', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    colors = {
        'gpu': 'lightcyan',
        'computation': 'lightgreen',
        'communication': 'lightblue'
    }
    
    # Create GPU nodes
    for gpu_id in range(64):
        pipeline_stage = gpu_id // 16
        expert_group = (gpu_id % 16) // 2
        tp_group = gpu_id % 2
        
        dot.node(f'gpu_{gpu_id}', f'GPU {gpu_id}\\nPipeline:{pipeline_stage} Expert:{expert_group} TP:{tp_group}\\n\\nExperts: 8\\nAttention Heads: 8\\nLayers: 4', 
                shape='box', style='filled,rounded', fillcolor=colors['gpu'], 
                fontname='Arial Bold')
    
    # Add communication edges between GPUs
    # Expert parallelism communication (all-to-all)
    for i in range(0, 64, 2):
        for j in range(8):
            if i != j * 2:
                dot.edge(f'gpu_{i}', f'gpu_{j * 2}', 
                        label='Expert A2A', style='dashed', color='blue', fontcolor='blue')
    
    # Tensor parallelism communication (all-reduce)
    for i in range(0, 64, 2):
        dot.edge(f'gpu_{i}', f'gpu_{i + 1}', 
                label='TP All-Reduce', color='red', fontcolor='red')
        dot.edge(f'gpu_{i + 1}', f'gpu_{i}', 
                label='TP All-Reduce', color='red', fontcolor='red')
    
    # Pipeline parallelism communication
    for stage in range(3):
        for gpu_in_stage in range(16):
            src_gpu = stage * 16 + gpu_in_stage
            dst_gpu = (stage + 1) * 16 + gpu_in_stage
            dot.edge(f'gpu_{src_gpu}', f'gpu_{dst_gpu}', 
                    label='Pipeline Send', color='green', fontcolor='green')
    
    return dot

def create_detailed_moe_layer():
    """Create detailed DAG for a single MoE layer"""
    dot = Digraph(comment='Detailed MoE Layer Implementation')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='9')
    
    colors = {
        'input': 'white',
        'computation': 'lightgreen',
        'communication': 'lightblue',
        'routing': 'lightyellow',
        'expert': 'lightcoral'
    }
    
    # Input
    dot.node('input', 'Layer Input\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='box', style='filled', fillcolor=colors['input'])
    
    # Router
    dot.node('router', 'Router Network\\n[batch_size=128, seq_len=1024, hidden_dim=1024]\\n→ [batch_size=128, seq_len=1024, num_experts=64]', 
             shape='parallelogram', style='filled', fillcolor=colors['routing'])
    
    # Top-k selection (k=2)
    dot.node('topk', 'Top-2 Expert Selection\\n[batch_size=128, seq_len=1024, num_experts=64]\\n→ [batch_size=128, seq_len=1024, top_k=2]', 
             shape='parallelogram', style='filled', fillcolor=colors['routing'])
    
    # Expert weights
    dot.node('expert_weights', 'Expert Weights\\n[batch_size=128, seq_len=1024, top_k=2]', 
             shape='parallelogram', style='filled', fillcolor=colors['routing'])
    
    # Token routing
    dot.node('token_routing', 'Token Routing\\n[batch_size=128, seq_len=1024, hidden_dim=1024]\\n→ Expert-specific batches', 
             shape='ellipse', style='filled', fillcolor=colors['communication'])
    
    # Create expert groups (8 groups of 8 experts each)
    for group_id in range(8):
        with dot.subgraph(name=f'cluster_expert_group_{group_id}') as c:
            c.attr(label=f'Expert Group {group_id} (GPU {group_id})', 
                   style='rounded,filled', fillcolor='lightcyan')
            
            # All-to-all communication for this group
            c.node(f'a2a_group_{group_id}', f'All-to-All Comm\\nGroup {group_id}\\n[batch_size=?, seq_len=1024, hidden_dim=1024]', 
                   shape='ellipse', style='filled', fillcolor=colors['communication'])
            
            # Individual experts in the group
            for expert_in_group in range(8):
                expert_id = group_id * 8 + expert_in_group
                c.node(f'expert_{expert_id}', f'Expert {expert_id}\\n[batch_size=?, seq_len=1024, hidden_dim=1024]\\n→ [batch_size=?, seq_len=1024, hidden_dim=1024]', 
                       shape='rectangle', style='filled', fillcolor=colors['expert'])
                
                # Expert FFN
                c.node(f'expert_{expert_id}_ffn1', f'Expert {expert_id} FFN1\\n[batch_size=?, seq_len=1024, hidden_dim=1024]\\n→ [batch_size=?, seq_len=1024, ffn_hidden=2048]', 
                       shape='rectangle', style='filled', fillcolor=colors['computation'])
                
                c.node(f'expert_{expert_id}_gelu', f'Expert {expert_id} GELU\\n[batch_size=?, seq_len=1024, ffn_hidden=2048]\\n→ [batch_size=?, seq_len=1024, ffn_hidden=2048]', 
                       shape='rectangle', style='filled', fillcolor=colors['computation'])
                
                c.node(f'expert_{expert_id}_ffn2', f'Expert {expert_id} FFN2\\n[batch_size=?, seq_len=1024, ffn_hidden=2048]\\n→ [batch_size=?, seq_len=1024, hidden_dim=1024]', 
                       shape='rectangle', style='filled', fillcolor=colors['computation'])
    
    # Expert aggregation
    dot.node('expert_agg', 'Expert Aggregation\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='parallelogram', style='filled', fillcolor=colors['routing'])
    
    # Weighted sum
    dot.node('weighted_sum', 'Weighted Sum\\n[batch_size=128, seq_len=1024, hidden_dim=1024]\\n× [batch_size=128, seq_len=1024, top_k=2]\\n→ [batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='parallelogram', style='filled', fillcolor=colors['routing'])
    
    # Output
    dot.node('output', 'Layer Output\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='box', style='filled', fillcolor=colors['input'])
    
    # Connections
    dot.edge('input', 'router')
    dot.edge('router', 'topk')
    dot.edge('topk', 'expert_weights')
    dot.edge('expert_weights', 'token_routing')
    
    # Connect to expert groups
    for group_id in range(8):
        dot.edge('token_routing', f'a2a_group_{group_id}')
        dot.edge(f'a2a_group_{group_id}', f'expert_{group_id * 8}')
        
        # Connect experts in group
        for expert_in_group in range(8):
            expert_id = group_id * 8 + expert_in_group
            dot.edge(f'expert_{expert_id}', f'expert_{expert_id}_ffn1')
            dot.edge(f'expert_{expert_id}_ffn1', f'expert_{expert_id}_gelu')
            dot.edge(f'expert_{expert_id}_gelu', f'expert_{expert_id}_ffn2')
            dot.edge(f'expert_{expert_id}_ffn2', 'expert_agg')
    
    dot.edge('expert_agg', 'weighted_sum')
    dot.edge('expert_weights', 'weighted_sum')
    dot.edge('weighted_sum', 'output')
    
    return dot

if __name__ == '__main__':
    # Create all DAGs
    print("Creating complete deployment DAG...")
    complete_dag = create_complete_deployment_dag()
    complete_dag.render('../outputs/2025-12-01-14-11-15/complete_deployment_dag', format='svg', cleanup=True)
    complete_dag.save('../outputs/2025-12-01-14-11-15/complete_deployment_dag.dot')
    print(f"Complete deployment DAG saved with {len(complete_dag.body)} nodes")
    
    print("Creating GPU mapping DAG...")
    gpu_dag = create_gpu_mapping_dag()
    gpu_dag.render('../outputs/2025-12-01-14-11-15/gpu_mapping_dag', format='svg', cleanup=True)
    gpu_dag.save('../outputs/2025-12-01-14-11-15/gpu_mapping_dag.dot')
    print(f"GPU mapping DAG saved with {len(gpu_dag.body)} nodes")
    
    print("Creating detailed MoE layer DAG...")
    moe_dag = create_detailed_moe_layer()
    moe_dag.render('../outputs/2025-12-01-14-11-15/detailed_moe_layer', format='svg', cleanup=True)
    moe_dag.save('../outputs/2025-12-01-14-11-15/detailed_moe_layer.dot')
    print(f"Detailed MoE layer DAG saved with {len(moe_dag.body)} nodes")
    
    print("\nAll DAG files generated successfully!")
    print("\nGenerated files:")
    print("1. Complete Deployment DAG: ../outputs/2025-12-01-14-11-15/complete_deployment_dag.svg")
    print("2. GPU Mapping DAG: ../outputs/2025-12-01-14-11-15/gpu_mapping_dag.svg")
    print("3. Detailed MoE Layer DAG: ../outputs/2025-12-01-14-11-15/detailed_moe_layer.svg")
    print("4. Main Hybrid Parallelism DAG: ../outputs/2025-12-01-14-11-15/llm_hybrid_parallelism_dag.svg")
    print("5. Expert Parallelism Detailed DAG: ../outputs/2025-12-01-14-11-15/expert_parallelism_detailed.svg")
    print("6. Tensor Parallelism Detailed DAG: ../outputs/2025-12-01-14-11-15/tensor_parallelism_detailed.svg")