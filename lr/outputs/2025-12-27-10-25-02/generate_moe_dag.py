#!/usr/bin/env python3

import graphviz

def create_moe_parallelism_dag():
    """Create a comprehensive DAG for MOE model with all parallelism strategies"""
    
    # Create directed graph
    dot = graphviz.Digraph(comment='MOE Parallel Strategy Deployment DAG')
    dot.attr(rankdir='TB', size='20,20')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\n[batch_size=128, seq_len=10240, token_dim=512]', 
             shape='rectangle', fillcolor='lightgray')
    
    # Data Parallel split - 4 groups
    for dp in range(4):
        # Each data parallel group processes batch_size/4 = 32 samples
        batch_size_dp = 32
        
        # Pipeline Stage 1 (layers 0-7)
        for stage in range(2):
            if stage == 0:  # Stage 1: layers 0-7
                layer_range = "0-7"
            else:  # Stage 2: layers 8-15
                layer_range = "8-15"
                
            # Expert Parallel groups - 8 groups per pipeline stage
            for ep in range(8):
                # Each expert parallel group handles 2 experts (16 total experts / 8 groups)
                experts_per_group = 2
                
                # Tensor Parallel split - 2-way within each expert
                for tp in range(2):
                    
                    # Create unique GPU ID
                    gpu_id = f"DP{dp}_P{stage}_EP{ep}_TP{tp}"
                    gpu_num = dp * 2 * 8 * 2 + stage * 8 * 2 + ep * 2 + tp
                    
                    # Layer processing nodes
                    for layer in range(8):
                        actual_layer = stage * 8 + layer
                        
                        # MHA computation
                        mha_node = f"mha_{gpu_id}_L{actual_layer}"
                        dot.node(mha_node, 
                                f"MHA GPU{gpu_num}\\nLayer {actual_layer}\\nInput: [batch={batch_size_dp}, seq=10240, heads=16, d_k=32]\\nOutput: [batch={batch_size_dp}, seq=10240, heads=16, d_k=32]",
                                shape='rectangle', fillcolor='lightgreen')
                        
                        # MHA to MOE routing
                        route_node = f"route_{gpu_id}_L{actual_layer}"
                        dot.node(route_node,
                                f"Gate Router GPU{gpu_num}\\nLayer {actual_layer}\\nInput: [batch={batch_size_dp}, seq=10240, dim=512]\\nOutput: [batch={batch_size_dp}, seq=10240, experts=2]",
                                shape='parallelogram', fillcolor='lightyellow')
                        
                        # Expert computation - 2 experts per GPU group
                        for expert in range(experts_per_group):
                            expert_id = ep * experts_per_group + expert
                            expert_node = f"expert_{gpu_id}_L{actual_layer}_E{expert_id}"
                            
                            # Tensor parallel split within expert
                            if tp == 0:
                                # First half of tensor dimensions
                                dot.node(expert_node,
                                        f"Expert {expert_id} GPU{gpu_num}\\nLayer {actual_layer}\\nInput: [batch={batch_size_dp}, seq=10240, dim=256]\\nOutput: [batch={batch_size_dp}, seq=10240, dim=256]",
                                        shape='rectangle', fillcolor='lightgreen')
                            else:
                                # Second half of tensor dimensions  
                                dot.node(expert_node,
                                        f"Expert {expert_id} GPU{gpu_num}\\nLayer {actual_layer}\\nInput: [batch={batch_size_dp}, seq=10240, dim=256]\\nOutput: [batch={batch_size_dp}, seq=10240, dim=256]",
                                        shape='rectangle', fillcolor='lightgreen')
                        
                        # Expert aggregation
                        agg_node = f"agg_{gpu_id}_L{actual_layer}"
                        dot.node(agg_node,
                                f"Expert Agg GPU{gpu_num}\\nLayer {actual_layer}\\nInput: [batch={batch_size_dp}, seq=10240, experts=2, dim=512]\\nOutput: [batch={batch_size_dp}, seq=10240, dim=512]",
                                shape='parallelogram', fillcolor='lightyellow')
    
    # Add communication edges between experts (dashed lines for gate selection)
    for dp in range(4):
        for stage in range(2):
            for ep in range(8):
                for tp in range(2):
                    gpu_id = f"DP{dp}_P{stage}_EP{ep}_TP{tp}"
                    gpu_num = dp * 2 * 8 * 2 + stage * 8 * 2 + ep * 2 + tp
                    
                    for layer in range(8):
                        actual_layer = stage * 8 + layer
                        
                        # Gate routing to experts (dashed lines)
                        route_node = f"route_{gpu_id}_L{actual_layer}"
                        for expert in range(2):
                            expert_id = ep * 2 + expert
                            expert_node = f"expert_{gpu_id}_L{actual_layer}_E{expert_id}"
                            dot.edge(route_node, expert_node, style='dashed', label=f'select E{expert_id}')
                        
                        # Expert to aggregation
                        for expert in range(2):
                            expert_id = ep * 2 + expert
                            expert_node = f"expert_{gpu_id}_L{actual_layer}_E{expert_id}"
                            agg_node = f"agg_{gpu_id}_L{actual_layer}"
                            dot.edge(expert_node, agg_node)
    
    # Add pipeline communication between stages
    for dp in range(4):
        for ep in range(8):
            for tp in range(2):
                # Communication from stage 0 to stage 1
                gpu_id_0 = f"DP{dp}_P0_EP{ep}_TP{tp}"
                gpu_id_1 = f"DP{dp}_P1_EP{ep}_TP{tp}"
                gpu_num_0 = dp * 2 * 8 * 2 + 0 * 8 * 2 + ep * 2 + tp
                gpu_num_1 = dp * 2 * 8 * 2 + 1 * 8 * 2 + ep * 2 + tp
                
                # Last layer of stage 0 to first layer of stage 1
                last_layer_s0 = f"agg_{gpu_id_0}_L7"
                first_layer_s1 = f"mha_{gpu_id_1}_L8"
                
                dot.node(f"pipe_comm_{dp}_{ep}_{tp}", 
                        f"Pipeline Comm\\nGPU{gpu_num_0} -> GPU{gpu_num_1}\\n[batch=32, seq=10240, dim=512]",
                        shape='ellipse', fillcolor='lightblue')
                
                dot.edge(last_layer_s0, f"pipe_comm_{dp}_{ep}_{tp}")
                dot.edge(f"pipe_comm_{dp}_{ep}_{tp}", first_layer_s1)
    
    # Add data parallel communication for gradient synchronization
    for stage in range(2):
        for ep in range(8):
            for tp in range(2):
                # Create gradient sync nodes
                grad_sync = f"grad_sync_P{stage}_EP{ep}_TP{tp}"
                dot.node(grad_sync,
                        f"Gradient Sync\\nStage {stage} EP {ep} TP {tp}\\n[batch=32, seq=10240, dim=512]",
                        shape='ellipse', fillcolor='lightblue')
                
                # Connect all data parallel groups to gradient sync
                for dp in range(4):
                    gpu_id = f"DP{dp}_P{stage}_EP{ep}_TP{tp}"
                    # Connect last layer output to gradient sync
                    last_agg = f"agg_{gpu_id}_L{15 if stage == 1 else 7}"
                    dot.edge(last_agg, grad_sync)
    
    # Output node
    dot.node('output', 'Output\\n[batch_size=128, seq_len=10240, token_dim=512]', 
             shape='rectangle', fillcolor='lightgray')
    
    # Connect final outputs to output node
    for dp in range(4):
        for ep in range(8):
            for tp in range(2):
                gpu_id = f"DP{dp}_P1_EP{ep}_TP{tp}"
                last_agg = f"agg_{gpu_id}_L15"
                dot.edge(last_agg, 'output')
    
    return dot

if __name__ == "__main__":
    # Generate the DAG
    dag = create_moe_parallelism_dag()
    
    # Save as DOT file
    dot_file = "./outputs/2025-12-27-10-25-02/moe_parallelism_dag.dot"
    with open(dot_file, 'w') as f:
        f.write(dag.source)
    
    # Save as SVG image
    svg_file = "./outputs/2025-12-27-10-25-02/moe_parallelism_dag.svg"
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {_dot_file}")
    print(f"SVG file: {svg_file}")