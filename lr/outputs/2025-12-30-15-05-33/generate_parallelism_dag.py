#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_parallelism_dag():
    # Create a new directed graph
    dot = Digraph(comment='LLM Parallelism Strategy Deployment DAG')
    dot.attr(rankdir='TB', size='50,50', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define colors for different parallelism types
    colors = {
        'input': 'lightblue',
        'computation': 'lightgreen', 
        'communication': 'lightyellow',
        'routing': 'lightcoral',
        'output': 'lightpink'
    }
    
    # Input node
    dot.node('input', 
             f'Input\\n[batch_size=128, seq_len=128-10240, heads=16, d_k=32]',
             shape='ellipse', style='filled', fillcolor=colors['input'])
    
    # Data Parallel groups (8 groups)
    for dp_group in range(8):
        with dot.subgraph(name=f'cluster_dp_{dp_group}') as dp:
            dp.attr(label=f'DP Group {dp_group}', style='rounded', bgcolor='lightgray')
            
            # Pipeline stages within each DP group
            for pp_stage in range(2):
                stage_name = f'PP_Stage_{pp_stage+1}_DP_{dp_group}'
                with dp.subgraph(name=f'cluster_{stage_name}') as stage:
                    stage.attr(label=f'PP Stage {pp_stage+1} (Layers {pp_stage*8+1}-{pp_stage*8+8})', 
                              style='dashed', bgcolor='white')
                    
                    # Expert Parallel groups within each stage
                    for ep_group in range(16):
                        ep_name = f'EP_{ep_group}_PP_{pp_stage}_DP_{dp_group}'
                        with stage.subgraph(name=f'cluster_{ep_name}') as ep:
                            ep.attr(label=f'EP {ep_group} (Expert {ep_group})', 
                                   style='filled', fillcolor='lightblue', bgcolor='lightblue')
                            
                            # Tensor Parallel groups (4 GPUs per EP)
                            for tp_gpu in range(4):
                                gpu_id = dp_group * 32 + pp_stage * 16 + ep_group * 4 + tp_gpu
                                gpu_name = f'GPU_{gpu_id}'
                                
                                # Layer processing nodes for each GPU
                                for layer in range(pp_stage*8+1, pp_stage*8+9):
                                    # Attention QKV computation
                                    qkv_node = f'QKV_L{layer}_GPU{gpu_id}'
                                    dot.node(qkv_node,
                                            f'QKV Computation\\nL{layer} GPU{gpu_id}\\nInput: [128, seq_len, 4, 32]\\nOutput: [128, seq_len, 4, 32]',
                                            shape='rectangle', style='filled', fillcolor=colors['computation'])
                                    
                                    # Attention communication
                                    attn_comm = f'AttnComm_L{layer}_GPU{gpu_id}'
                                    dot.node(attn_comm,
                                            f'Attention All-Reduce\\nL{layer} GPU{gpu_id}\\nInput: [128, seq_len, 4, 32]\\nOutput: [128, seq_len, 4, 32]',
                                            shape='ellipse', style='filled', fillcolor=colors['communication'])
                                    
                                    # Attention output
                                    attn_out = f'AttnOut_L{layer}_GPU{gpu_id}'
                                    dot.node(attn_out,
                                            f'Attention Output\\nL{layer} GPU{gpu_id}\\nInput: [128, seq_len, 4, 32]\\nOutput: [128, seq_len, 4, 32]',
                                            shape='rectangle', style='filled', fillcolor=colors['computation'])
                                    
                                    # MOE Gate (routing)
                                    gate_node = f'Gate_L{layer}_GPU{gpu_id}'
                                    dot.node(gate_node,
                                            f'MOE Gate\\nL{layer} GPU{gpu_id}\\nInput: [128, seq_len, 512]\\nOutput: [128, seq_len, 2]',
                                            shape='parallelogram', style='filled', fillcolor=colors['routing'])
                                    
                                    # Expert computation (this GPU's expert)
                                    expert_node = f'Expert_L{layer}_GPU{gpu_id}'
                                    dot.node(expert_node,
                                            f'Expert {ep_group}\\nL{layer} GPU{gpu_id}\\nInput: [128, seq_len, 512]\\nOutput: [128, seq_len, 1024]',
                                            shape='rectangle', style='filled', fillcolor=colors['computation'])
                                    
                                    # Expert communication (if needed)
                                    if ep_group < 15:  # Not the last expert
                                        expert_comm = f'ExpertComm_L{layer}_GPU{gpu_id}'
                                        dot.node(expert_comm,
                                                f'Expert Communication\\nL{layer} GPU{gpu_id}→GPU{gpu_id+4}\\nInput: [128, seq_len, 1024]\\nOutput: [128, seq_len, 1024]',
                                                shape='ellipse', style='filled', fillcolor=colors['communication'])
                                    
                                    # MOE aggregation
                                    moe_agg = f'MOEAgg_L{layer}_GPU{gpu_id}'
                                    dot.node(moe_agg,
                                            f'MOE Aggregation\\nL{layer} GPU{gpu_id}\\nInput: [128, seq_len, 1024]\\nOutput: [128, seq_len, 512]',
                                            shape='parallelogram', style='filled', fillcolor=colors['routing'])
                                    
                                    # Connect nodes within layer
                                    dot.edge(qkv_node, attn_comm)
                                    dot.edge(attn_comm, attn_out)
                                    
                                    # MOE connections with gate routing (dashed line)
                                    dot.edge(attn_out, gate_node, style='dashed')
                                    dot.edge(gate_node, expert_node, style='dashed')
                                    dot.edge(expert_node, moe_agg)
                                    
                                    # Expert communication if needed
                                    if ep_group < 15:
                                        dot.edge(expert_node, expert_comm)
    
    # Connect layers across pipeline stages
    for dp_group in range(8):
        for ep_group in range(16):
            for tp_gpu in range(4):
                gpu_id = dp_group * 32 + ep_group * 4 + tp_gpu
                
                # Connect PP Stage 1 to Stage 2
                last_layer_s1 = f'MOEAgg_L8_GPU{gpu_id}'
                first_layer_s2 = f'QKV_L9_GPU{gpu_id + 128}'  # Stage 2 starts at GPU 128
                dot.edge(last_layer_s1, first_layer_s2, 
                        label='PP Communication', style='bold')
    
    # Output node
    final_gpu = 127  # Last GPU
    final_layer = f'MOEAgg_L16_GPU{final_gpu}'
    dot.node('output', 
             f'Output\\n[batch_size=128, seq_len=128-10240, heads=16, d_k=32]',
             shape='ellipse', style='filled', fillcolor=colors['output'])
    dot.edge(final_layer, 'output', label='Final Output', style='bold')
    
    return dot

def create_simplified_dag():
    # Create a simplified version focusing on key parallelism patterns
    dot = Digraph(comment='LLM Parallelism Strategy - Simplified View')
    dot.attr(rankdir='TB', size='30,30', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='12')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    colors = {
        'input': 'lightblue',
        'computation': 'lightgreen', 
        'communication': 'lightyellow',
        'routing': 'lightcoral',
        'output': 'lightpink'
    }
    
    # Input
    dot.node('input', 'Input Batch\\n[128, seq_len, 512]', 
             shape='ellipse', style='filled', fillcolor=colors['input'])
    
    # Data Parallel groups
    for dp in range(8):
        with dot.subgraph(name=f'cluster_dp_{dp}') as c:
            c.attr(label=f'DP Group {dp}')
            
            # Pipeline stages
            for pp in range(2):
                stage_name = f'PP{pp+1}_DP{dp}'
                
                # Show representative GPUs for each parallelism dimension
                gpu_cluster = f'GPU_Cluster_PP{pp+1}_DP{dp}'
                c.node(gpu_cluster, 
                      f'PP Stage {pp+1}\\n16 EP × 4 TP = 64 GPUs\\nLayers {pp*8+1}-{pp*8+8}',
                      shape='rectangle', style='filled', fillcolor=colors['computation'])
                
                # Expert Parallel representation
                ep_cluster = f'EP_Cluster_PP{pp+1}_DP{dp}'
                c.node(ep_cluster,
                      f'Expert Parallel\\n16 Experts × 1 GPU each\\nMOE Computation',
                      shape='rectangle', style='filled', fillcolor=colors['computation'])
                
                # Tensor Parallel communication
                tp_comm = f'TP_Comm_PP{pp+1}_DP{dp}'
                c.node(tp_comm,
                      f'Tensor Parallel\\n4-way All-Reduce\\nAttention Heads',
                      shape='ellipse', style='filled', fillcolor=colors['communication'])
                
                # MOE Gate routing
                gate_node = f'Gate_PP{pp+1}_DP{dp}'
                c.node(gate_node,
                      f'MOE Gate\\nExpert Selection\\nTop-2 Routing',
                      shape='parallelogram', style='filled', fillcolor=colors['routing'])
                
                # Connect within stage
                c.edge(gpu_cluster, tp_comm)
                c.edge(tp_comm, ep_cluster)
                c.edge(gate_node, ep_cluster, style='dashed')
            
            # Connect pipeline stages
            pp1_final = f'EP_Cluster_PP1_DP{dp}'
            pp2_start = f'GPU_Cluster_PP2_DP{dp}'
            c.edge(pp1_final, pp2_start, label='PP Communication', style='bold')
    
    # Output
    dot.node('output', 'Output Batch\\n[128, seq_len, 512]',
             shape='ellipse', style='filled', fillcolor=colors['output'])
    
    # Connect final nodes to output
    for dp in range(8):
        final_node = f'EP_Cluster_PP2_DP{dp}'
        dot.edge(final_node, 'output')
    
    return dot

if __name__ == '__main__':
    # Generate detailed DAG
    print("Generating detailed parallelism DAG...")
    detailed_dag = create_parallelism_dag()
    
    # Save DOT file
    dot_file_path = './outputs/2025-12-30-15-05-33/parallelism_strategy_detailed.dot'
    with open(dot_file_path, 'w') as f:
        f.write(detailed_dag.source)
    print(f"Detailed DOT file saved to: {dot_file_path}")
    
    # Generate simplified DAG
    print("Generating simplified parallelism DAG...")
    simplified_dag = create_simplified_dag()
    
    # Save DOT file
    simplified_dot_path = './outputs/2025-12-30-15-05-33/parallelism_strategy_simplified.dot'
    with open(simplified_dot_path, 'w') as f:
        f.write(simplified_dag.source)
    print(f"Simplified DOT file saved to: {simplified_dot_path}")
    
    # Try to render SVG (if graphviz is available)
    try:
        detailed_dag.render('./outputs/2025-12-30-15-05-33/parallelism_strategy_detailed', format='svg', cleanup=True)
        print("Detailed SVG rendered successfully")
        
        simplified_dag.render('./outputs/2025-12-30-15-05-33/parallelism_strategy_simplified', format='svg', cleanup=True)
        print("Simplified SVG rendered successfully")
    except Exception as e:
        print(f"SVG rendering failed (graphviz may not be installed): {e}")
        print("DOT files are available for manual rendering")
    
    print("DAG generation complete!")