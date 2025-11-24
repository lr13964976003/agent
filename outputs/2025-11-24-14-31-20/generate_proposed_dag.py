import graphviz
import os

def create_proposed_ep16_dag():
    dot = graphviz.Digraph(comment='Large EP Proposed Method - EP=16, 1 Expert/GPU')
    dot.attr(rankdir='TB', splines='ortho')
    
    # Overall input
    dot.node('input', 'Model Input\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
             shape='box', style='filled', fillcolor='lightblue')
    
    # Layer grouping - we'll show 3 representative layers (1st, middle, last) and indicate others
    for layer_idx in [0, 7, 15]:
        layer_name = f"layer{layer_idx}"
        
        # MHA for this layer
        mha_name = f"mha{layer_idx}"
        dot.node(mha_name, f'MHA Layer {layer_idx}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                shape='rectangle', style='filled', fillcolor='lightcyan')
        
        if layer_idx == 0:
            dot.edge('input', mha_name)
        else:
            prev_layer = layer_idx - 1
            dot.edge(f'moe{prev_layer}', mha_name)
        
        # Add layernorm before MHA (if needed)
        layernorm1 = f"ln1_{layer_idx}"
        dot.node(layernorm1, f'LayerNorm {layer_idx}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                shape='rectangle')
        
        # MHA residual add
        add1 = f"add1_{layer_idx}"
        dot.node(add1, f'MHA Residual Add {layer_idx}\\nInput1: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nInput2: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                shape='rectangle', style='filled', fillcolor='yellow')
        
        dot.edge(mha_name, add1)
        
        # Gate computation
        gate = f"gate{layer_idx}"
        dot.node(gate, f'Gate Layer {layer_idx}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, num_experts=16]', 
                shape='parallelogram', style='filled', fillcolor='lightgreen')
        
        dot.edge(add1, gate)
        
        # Expert routing and communication
        for expert_id in range(16):
            # Token routing to expert
            route = f"route{layer_idx}_exp{expert_id}"
            gpu_id = expert_id  # Each expert on separate GPU
            dot.node(route, f'Route to Expert {expert_id}\\nGPU {gpu_id}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [tokens_per_expert, hidden_dim=4096]', 
                    shape='ellipse', style='dashed', fillcolor='orange')
            dot.edge(gate, route)
            
            # Expert computation
            expert = f"expert{layer_idx}_exp{expert_id}"
            dot.node(expert, f'MLP Expert {expert_id}\\nGPU {gpu_id}\\nInput: [tokens_per_expert, hidden_dim=4096]\\nOutput: [tokens_per_expert, hidden_dim=4096]', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
            dot.edge(route, expert)
            
            # Expert output aggregation
            aggregate = f"agg{layer_idx}_exp{expert_id}"
            dot.node(aggregate, f'Aggregate from Expert {expert_id}\\nGPU {gpu_id}\\nInput: [tokens_per_expert, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                    shape='ellipse', style='filled', fillcolor='lightpink')
            dot.edge(expert, aggregate)
        
        # Final MoE aggregation (all experts combined)
        moe_agg = f"moe_agg{layer_idx}"
        dot.node(moe_agg, f'MoE Final Aggregation {layer_idx}\\nInput: 16×[batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                shape='ellipse', style='filled', fillcolor='gold')
        
        # Connect all expert aggregations to final aggregation
        for expert_id in range(16):
            dot.edge(f"agg{layer_idx}_exp{expert_id}", moe_agg)
        
        # MoE residual add
        moe_add = f"moe{layer_idx}"
        dot.node(moe_add, f'MoE Residual Add {layer_idx}\\nInput1: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nInput2: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
                shape='rectangle', style='filled', fillcolor='yellow')
        
        dot.edge(moe_agg, moe_add)
        dot.edge(add1, moe_add)  # Residual connection
    
    # Final output
    dot.node('output', 'Model Output\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]', 
             shape='box', style='filled', fillcolor='lightblue')
    
    dot.edge('moe15', 'output')
    
    # Save the DAG
    output_dir = "../outputs/2025-11-24-14-31-20"
    os.makedirs(output_dir, exist_ok=True)
    
    dot.render(os.path.join(output_dir, 'proposed_ep16_dag'), format='svg', cleanup=False)
    
    # Save DOT file
    with open(os.path.join(output_dir, 'proposed_ep16_dag.dot'), 'w') as f:
        f.write(dot.source)
    
    return dot.source

if __name__ == "__main__":
    dag_source = create_proposed_ep16_dag()
    print("Proposed EP=16 DAG generated successfully!")