import graphviz
import os

def create_baseline_dag():
    dot = graphviz.Digraph(comment='Baseline Method - TP=8, PP=2, 8 Experts/GPU')
    dot.attr(rankdir='TB', splines='ortho')
    
    # Overall input
    dot.node('input', 'Model Input\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
             shape='box', style='filled', fillcolor='lightblue')
    
    # Pipeline stages - 2 stages (0-7 layers, 8-15 layers)
    for stage in [0, 1]:
        stage_start = stage * 8
        stage_end = stage_start + 8
        stage_label = f"stage{stage}"
        
        # Create subgraph for pipeline stage
        with dot.subgraph(name=f'cluster_stage{stage}') as stage_subgraph:
            stage_subgraph.attr(label=f'Pipeline Stage {stage} (Layers {stage_start}-{stage_end-1})')
            stage_subgraph.attr(style='dotted')
            
            # For each layer in this stage
            for layer_idx in [stage_start, stage_start+7]:  # Show first and last in each stage
                layer_name = f"layer{layer_idx}"
                
                # MHA for this layer (with TP=8)
                mha_name = f"mha{layer_idx}"
                dot.node(mha_name, f'MHA Layer {layer_idx}\\nTP=8 across 8 GPUs\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                        shape='rectangle', style='filled', fillcolor='lightcyan')
                
                if stage == 0 and layer_idx == 0:
                    dot.edge('input', mha_name)
                elif layer_idx > 0:
                    dot.edge(f'moe{layer_idx-1}', mha_name)
                
                # Layernorm and residual
                add1 = f"add1_{layer_idx}"
                dot.node(add1, f'MHA Residual Add {layer_idx}\\nInput1: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nInput2: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                        shape='rectangle', style='filled', fillcolor='yellow')
                
                dot.edge(mha_name, add1)
                
                # Gate computation
                gate = f"gate{layer_idx}"
                dot.node(gate, f'Gate Layer {layer_idx}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, num_experts=16]', 
                        shape='parallelogram', style='filled', fillcolor='lightgreen')
                
                dot.edge(add1, gate)
                
                # Expert groups - 2 GPUs per group, 8 experts per GPU
                for gpu_group in range(2):  # 2 GPU groups per stage
                    gpu_start = gpu_group * 8
                    
                    # Each GPU has 8 experts
                    for gpu in range(4):  # 4 GPUs per group, 8 total per stage
                        gpu_id = stage * 8 + gpu_group * 4 + gpu
                        
                        # Expert cluster on this GPU
                        with dot.subgraph(name=f'cluster_gpu{gpu_id}') as gpu_subgraph:
                            gpu_subgraph.attr(label=f'GPU {gpu_id} (8 experts)')
                            gpu_subgraph.attr(style='dashed')
                            
                            # Expert routing to this GPU
                            route = f"route{layer_idx}_gpu{gpu_id}"
                            dot.node(route, f'Route to GPU {gpu_id}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [tokens_per_gpu, hidden_dim=4096]', 
                                    shape='ellipse', style='dashed', fillcolor='orange')
                            dot.edge(gate, route)
                            
                            # Process 8 experts on this GPU sequentially
                            expert_cluster = f"experts_gpu{gpu_id}_layer{layer_idx}"
                            dot.node(expert_cluster, f'8 Experts GPU {gpu_id}\\nInput: [tokens_per_gpu, hidden_dim=4096]\\nOutput: [tokens_per_gpu, hidden_dim=4096]\\n(Sequential processing)', 
                                    shape='rectangle', style='filled', fillcolor='lightcoral')
                            dot.edge(route, expert_cluster)
                            
                            # Aggregation from this GPU
                            aggregate = f"agg{layer_idx}_gpu{gpu_id}"
                            dot.node(aggregate, f'Aggregate GPU {gpu_id}\\nInput: [tokens_per_gpu, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                                    shape='ellipse', style='filled', fillcolor='lightpink')
                            dot.edge(expert_cluster, aggregate)
                
                # Final aggregation across all GPUs
                moe_agg = f"moe_agg{layer_idx}"
                dot.node(moe_agg, f'MoE Final Aggregation {layer_idx}\\nInput: 16×[batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                        shape='ellipse', style='filled', fillcolor='gold')
                
                # Connect all GPU aggregations to final aggregation
                for gpu_id in range(16):
                    dot.edge(f"agg{layer_idx}_gpu{gpu_id}", moe_agg)
                
                # MoE residual add
                moe_add = f"moe{layer_idx}"
                dot.node(moe_add, f'MoE Residual Add {layer_idx}\\nInput1: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nInput2: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                        shape='rectangle', style='filled', fillcolor='yellow')
                
                dot.edge(moe_agg, moe_add)
                dot.edge(add1, moe_add)  # Residual connection
    
    # Pipeline communication between stages
    dot.edge('moe7', 'mha8', label='Pipeline Communication\\nStage 0 → Stage 1')
    
    # Final output
    dot.node('output', 'Model Output\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
             shape='box', style='filled', fillcolor='lightblue')
    
    dot.edge('moe15', 'output')
    
    # Save the DAG
    output_dir = "../outputs/2025-11-24-14-31-20"
    os.makedirs(output_dir, exist_ok=True)
    
    dot.render(os.path.join(output_dir, 'baseline_dag'), format='svg', cleanup=False)
    
    # Save DOT file
    with open(os.path.join(output_dir, 'baseline_dag.dot'), 'w') as f:
        f.write(dot.source)
    
    return dot.source

if __name__ == "__main__":
    dag_source = create_baseline_dag()
    print("Baseline DAG generated successfully!")