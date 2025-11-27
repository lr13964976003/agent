#!/usr/bin/env python3

import graphviz
import os

def create_complete_deployment_dag():
    """
    Create a complete deployment DAG showing the integration of:
    - Expert Parallelism (EP >= 64)
    - Data Parallelism (DP)
    - Tensor Parallelism (TP) when needed
    - Pipeline scheduling
    - One expert per GPU strategy
    """
    
    # Model configuration
    token_dim = 7168
    num_heads = 128
    head_dim = 128
    mlp_hidden = 2048
    batch_size = 4
    seq_len = 2048
    experts_per_layer = 64
    dp_degree = 2  # Data parallelism across replicas
    total_gpus = 128
    
    dag = graphviz.Digraph('Complete_MoE_Deployment',
                          filename='complete_moe_deployment_dag',
                          format='svg',
                          graph_attr={
                              'rankdir': 'TB',
                              'bgcolor': 'white',
                              'fontname': 'Arial',
                              'fontsize': '12',
                              'label': 'Complete Large-Scale Cross-Node MoE Deployment\nEP=64, DP=2, One Expert per GPU, 128 GPUs Total',
                              'labelloc': 't',
                              'ranksep': '1.5',
                              'nodesep': '0.8'
                          })
    
    # Global node attributes
    dag.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen', fontsize='10')
    
    # Input processing - Data Parallelism
    for dp_id in range(dp_degree):
        dp_prefix = f'dp_{dp_id}'
        
        dag.node(f'{dp_prefix}_input',
                 f'DP Replica {dp_id} Input\nGPU: DP_{dp_id}_GPUs\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]',
                 fillcolor='lightcyan')
        
        # Dense layers (first 3)
        for layer_idx in range(3):
            # MHA decomposition
            dag.node(f'{dp_prefix}_dense_{layer_idx}_q',
                     f'Dense L{layer_idx} Q Proj\nGPU: GPU_{dp_id*16}-{dp_id*16+15}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                     fillcolor='lightgreen')
            
            dag.node(f'{dp_prefix}_dense_{layer_idx}_k',
                     f'Dense L{layer_idx} K Proj\nGPU: GPU_{dp_id*16+16}-{dp_id*16+31}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                     fillcolor='lightgreen')
            
            dag.node(f'{dp_prefix}_dense_{layer_idx}_v',
                     f'Dense L{layer_idx} V Proj\nGPU: GPU_{dp_id*16+32}-{dp_id*16+47}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                     fillcolor='lightgreen')
            
            dag.node(f'{dp_prefix}_dense_{layer_idx}_attn',
                     f'Dense L{layer_idx} Attention\nGPU: GPU_{dp_id*16}-{dp_id*16+63}\nInput: Q,K,V [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                     fillcolor='lightgreen')
            
            dag.node(f'{dp_prefix}_dense_{layer_idx}_out',
                     f'Dense L{layer_idx} Output Proj\nGPU: GPU_{dp_id*16}-{dp_id*16+63}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]',
                     fillcolor='lightgreen')
            
            dag.node(f'{dp_prefix}_dense_{layer_idx}_mlp1',
                     f'Dense L{layer_idx} MLP1\nGPU: GPU_{dp_id*16}-{dp_id*16+63}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, hidden={mlp_hidden}]',
                     fillcolor='lightgreen')
            
            dag.node(f'{dp_prefix}_dense_{layer_idx}_mlp2',
                     f'Dense L{layer_idx} MLP2\nGPU: GPU_{dp_id*16}-{dp_id*16+63}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, hidden={mlp_hidden}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]',
                     fillcolor='lightgreen')
        
        # MoE layers (remaining 58 layers)
        for layer_idx in range(3, 61):
            # MHA (shared across experts)
            dag.node(f'{dp_prefix}_moe_{layer_idx}_q',
                     f'MoE L{layer_idx} Q Proj\nGPU: GPU_{dp_id*16}-{dp_id*16+15}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                     fillcolor='lightgreen')
            
            dag.node(f'{dp_prefix}_moe_{layer_idx}_k',
                     f'MoE L{layer_idx} K Proj\nGPU: GPU_{dp_id*16+16}-{dp_id*16+31}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                     fillcolor='lightgreen')
            
            dag.node(f'{dp_prefix}_moe_{layer_idx}_v',
                     f'MoE L{layer_idx} V Proj\nGPU: GPU_{dp_id*16+32}-{dp_id*16+47}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                     fillcolor='lightgreen')
            
            dag.node(f'{dp_prefix}_moe_{layer_idx}_attn',
                     f'MoE L{layer_idx} Attention\nGPU: GPU_{dp_id*16}-{dp_id*16+63}\nInput: Q,K,V [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                     fillcolor='lightgreen')
            
            dag.node(f'{dp_prefix}_moe_{layer_idx}_out',
                     f'MoE L{layer_idx} Output Proj\nGPU: GPU_{dp_id*16}-{dp_id*16+63}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]',
                     fillcolor='lightgreen')
            
            # Gating network
            dag.node(f'{dp_prefix}_moe_{layer_idx}_gating',
                     f'MoE L{layer_idx} Gating\nGPU: GPU_{dp_id*16}-{dp_id*16+15}\nInput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: Expert assignments',
                     shape='parallelogram', fillcolor='yellow')
            
            # Expert processing - 64 experts across 128 GPUs
            for expert_id in range(experts_per_layer):
                gpu_id = dp_id * 64 + expert_id  # Each expert gets its own GPU
                
                dag.node(f'{dp_prefix}_expert_{layer_idx}_{expert_id}_mlp1',
                         f'Expert {expert_id} MLP1\nGPU: GPU_{gpu_id}\nInput: [routed_tokens, token_dim={token_dim}]\nOutput: [routed_tokens, hidden={mlp_hidden}]',
                         fillcolor='lightgreen')
                
                dag.node(f'{dp_prefix}_expert_{layer_idx}_{expert_id}_activation',
                         f'Expert {expert_id} GELU\nGPU: GPU_{gpu_id}\nInput: [routed_tokens, hidden={mlp_hidden}]\nOutput: [routed_tokens, hidden={mlp_hidden}]',
                         fillcolor='lightgreen')
                
                dag.node(f'{dp_prefix}_expert_{layer_idx}_{expert_id}_mlp2',
                         f'Expert {expert_id} MLP2\nGPU: GPU_{gpu_id}\nInput: [routed_tokens, hidden={mlp_hidden}]\nOutput: [routed_tokens, token_dim={token_dim}]',
                         fillcolor='lightgreen')
            
            # Expert aggregation
            dag.node(f'{dp_prefix}_moe_{layer_idx}_agg',
                     f'MoE L{layer_idx} Aggregation\nGPU: GPU_{dp_id*16}-{dp_id*16+63}\nInput: [expert_outputs, token_dim={token_dim}]\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]',
                     shape='parallelogram', fillcolor='yellow')
        
        # Final output
        dag.node(f'{dp_prefix}_output',
                 f'DP Replica {dp_id} Output\nGPU: DP_{dp_id}_GPUs\nOutput: [batch_size={batch_size//dp_degree}, seq_len={seq_len}, token_dim={token_dim}]',
                 fillcolor='lightcyan')
    
    # Add communication edges
    for dp_id in range(dp_degree):
        dp_prefix = f'dp_{dp_id}'
        
        # Connect dense layers
        for layer_idx in range(3):
            dag.edge(f'{dp_prefix}_input', f'{dp_prefix}_dense_0_q')
            dag.edge(f'{dp_prefix}_dense_{layer_idx}_mlp2', 
                    f'{dp_prefix}_dense_{layer_idx+1}_q' if layer_idx < 2 else f'{dp_prefix}_moe_3_q')
        
        # Connect MoE layers
        for layer_idx in range(3, 61):
            dag.edge(f'{dp_prefix}_moe_{layer_idx}_out', f'{dp_prefix}_moe_{layer_idx}_gating', style='dashed')
            
            # Connect gating to all experts
            for expert_id in range(experts_per_layer):
                dag.edge(f'{dp_prefix}_moe_{layer_idx}_gating', 
                        f'{dp_prefix}_expert_{layer_idx}_{expert_id}_mlp1', 
                        style='dashed', label='token routing')
                
                dag.edge(f'{dp_prefix}_expert_{layer_idx}_{expert_id}_mlp2', 
                        f'{dp_prefix}_moe_{layer_idx}_agg', 
                        style='dashed')
            
            dag.edge(f'{dp_prefix}_moe_{layer_idx}_agg', 
                    f'{dp_prefix}_moe_{layer_idx+1}_q' if layer_idx < 60 else f'{dp_prefix}_output')
    
    return dag

if __name__ == '__main__':
    os.makedirs('../outputs/2025-11-26-17-03-32', exist_ok=True)
    
    print("Generating complete deployment DAG...")
    deployment_dag = create_complete_deployment_dag()
    deployment_dag.render(directory='../outputs/2025-11-26-17-03-32', cleanup=True)
    
    # Save DOT file
    with open('../outputs/2025-11-26-17-03-32/complete_deployment_dag.dot', 'w') as f:
        f.write(deployment_dag.source)
    
    print("Complete deployment DAG generated!")