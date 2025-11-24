import graphviz
import os

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2, 16 GPUs total"""
    dot = graphviz.Digraph('baseline_moe', format='svg')
    dot.attr(rankdir='TB', size='15,20')
    
    # Color scheme for different GPU groups
    gpu_colors = {
        'gpu_0_7': 'lightblue',
        'gpu_8_15': 'lightgreen',
        'all_gpus': 'lightgray'
    }
    
    # Input node
    dot.node('input', 'Input\\nBatch: 128\\nSeq Len: 10,000\\nDim: 4096', 
             shape='ellipse', style='filled', fillcolor='yellow')
    
    # Pipeline Stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_pipeline_0') as c0:
        c0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='dashed', color='blue')
        
        # For each layer in pipeline stage 0
        for layer in range(1, 9):  # 8 layers per stage
            # Input split for tensor parallelism
            c0.node(f'split_l{layer}_p0', f'Split Layer {layer}\\nInput: [128,10000,4096]\\nOutput: [128,10000,512]', 
                   shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            # Multi-Head Attention
            for gpu in range(8):
                c0.node(f'mha_l{layer}_gpu{gpu}', 
                       f'MHA Layer {layer}\\nGPU {gpu}\\nInput: [128,10000,512]\\nOutput: [128,10000,512]', 
                       shape='rectangle', style='filled', fillcolor=gpu_colors['gpu_0_7'])
            
            # MLP with experts (8 experts per GPU)
            for gpu in range(8):
                # Gate computation
                c0.node(f'gate_l{layer}_gpu{gpu}', 
                       f'Gate Layer {layer}\\nGPU {gpu}\\nInput: [128,10000,512]\\nOutput: [128,10000,8]', 
                       shape='parallelogram', style='filled', fillcolor=gpu_colors['gpu_0_7'])
                
                # Expert computation
                for expert in range(8):
                    c0.node(f'exp_l{layer}_gpu{gpu}_exp{expert}', 
                           f'Expert {expert}\\nLayer {layer}\\nGPU {gpu}\\nInput: [128,?,512]\\nOutput: [128,?,512]', 
                           shape='rectangle', style='filled', fillcolor=gpu_colors['gpu_0_7'])
                
                # Expert aggregation
                c0.node(f'agg_l{layer}_gpu{gpu}', 
                       f'Aggregate\\nLayer {layer}\\nGPU {gpu}\\nInput: [128,10000,512]\\nOutput: [128,10000,512]', 
                       shape='parallelogram', style='filled', fillcolor=gpu_colors['gpu_0_7'])
            
            # Residual connections and all-reduce
            c0.node(f'residual_l{layer}_p0', f'Residual Add\\nLayer {layer}\\nGPUs 0-7\\nInput: [128,10000,512]\\nOutput: [128,10000,4096]', 
                   shape='rectangle', style='filled', fillcolor=gpu_colors['all_gpus'])
    
    # Pipeline Stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_pipeline_1') as c1:
        c1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='dashed', color='green')
        
        # Similar structure for pipeline stage 1
        for layer in range(9, 17):  # Layers 9-16
            c1.node(f'split_l{layer}_p1', f'Split Layer {layer}\\nInput: [128,10000,4096]\\nOutput: [128,10000,512]', 
                   shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            # Multi-Head Attention
            for gpu in range(8, 16):
                c1.node(f'mha_l{layer}_gpu{gpu}', 
                       f'MHA Layer {layer}\\nGPU {gpu}\\nInput: [128,10000,512]\\nOutput: [128,10000,512]', 
                       shape='rectangle', style='filled', fillcolor=gpu_colors['gpu_8_15'])
            
            # MLP with experts
            for gpu in range(8, 16):
                c1.node(f'gate_l{layer}_gpu{gpu}', 
                       f'Gate Layer {layer}\\nGPU {gpu}\\nInput: [128,10000,512]\\nOutput: [128,10000,8]', 
                       shape='parallelogram', style='filled', fillcolor=gpu_colors['gpu_8_15'])
                
                for expert in range(8):
                    c1.node(f'exp_l{layer}_gpu{gpu}_exp{expert}', 
                           f'Expert {expert}\\nLayer {layer}\\nGPU {gpu}\\nInput: [128,?,512]\\nOutput: [128,?,512]', 
                           shape='rectangle', style='filled', fillcolor=gpu_colors['gpu_8_15'])
                
                c1.node(f'agg_l{layer}_gpu{gpu}', 
                       f'Aggregate\\nLayer {layer}\\nGPU {gpu}\\nInput: [128,10000,512]\\nOutput: [128,10000,512]', 
                       shape='parallelogram', style='filled', fillcolor=gpu_colors['gpu_8_15'])
            
            c1.node(f'residual_l{layer}_p1', f'Residual Add\\nLayer {layer}\\nGPUs 8-15\\nInput: [128,10000,512]\\nOutput: [128,10000,4096]', 
                   shape='rectangle', style='filled', fillcolor=gpu_colors['all_gpus'])
    
    # Pipeline communication
    dot.node('pipeline_comm', 'Pipeline Communication\\nGPUs 7→8\\n[128,10000,4096]→[128,10000,4096]', 
             shape='ellipse', style='dashed', fillcolor='orange')
    
    # Output node
    dot.node('output', 'Output\\nBatch: 128\\nSeq Len: 10,000\\nDim: 4096', 
             shape='ellipse', style='filled', fillcolor='yellow')
    
    # Connections for baseline DAG
    # Simplified connections - representative pattern
    dot.edge('input', 'split_l1_p0')
    
    # MHA connections for layer 1
    for gpu in range(8):
        dot.edge(f'split_l1_p0', f'mha_l1_gpu{gpu}')
    
    # Gate connections
    for gpu in range(8):
        dot.edge(f'mha_l1_gpu{gpu}', f'gate_l1_gpu{gpu}')
        # Dashed connections from gate to experts
        for expert in range(8):
            dot.edge(f'gate_l1_gpu{gpu}', f'exp_l1_gpu{gpu}_exp{expert}', style='dashed')
    
    # Expert to aggregation
    for gpu in range(8):
        for expert in range(8):
            dot.edge(f'exp_l1_gpu{gpu}_exp{expert}', f'agg_l1_gpu{gpu}')
        dot.edge(f'agg_l1_gpu{gpu}', f'residual_l1_p0')
    
    # Pipeline communication
    dot.edge('residual_l8_p0', 'pipeline_comm')
    dot.edge('pipeline_comm', 'split_l9_p1')
    dot.edge('residual_l16_p1', 'output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with EP=16, 1 expert per GPU"""
    dot = graphviz.Digraph('proposed_moe', format='svg')
    dot.attr(rankdir='TB', size='20,25')
    
    # Color scheme for GPUs
    gpu_colors = [f'lightblue{i}' for i in range(16)]
    
    # Input node
    dot.node('input', 'Input\\nBatch: 128\\nSeq Len: 10,000\\nDim: 4096', 
             shape='ellipse', style='filled', fillcolor='yellow')
    
    # Global token distribution
    dot.node('token_dist', 'Token Distribution\\nAll GPUs\\n[128,10000,4096]→[128,10000,4096]', 
             shape='ellipse', style='filled', fillcolor='lightcoral')
    
    # For each layer (16 layers total)
    for layer in range(1, 17):
        with dot.subgraph(name=f'cluster_layer_{layer}') as c:
            c.attr(label=f'Layer {layer} (16 Experts on 16 GPUs)', style='dashed')
            
            # Layer input distribution
            c.node(f'dist_l{layer}', f'Distribute Tokens\\nLayer {layer}\\nAll GPUs\\n[128,10000,4096]', 
                   shape='ellipse', style='filled', fillcolor='lightyellow')
            
            # Multi-Head Attention - each GPU processes full sequence for its tokens
            for gpu in range(16):
                c.node(f'mha_l{layer}_gpu{gpu}', 
                       f'MHA\\nLayer {layer}\\nGPU {gpu}\\nInput: [128,?,4096]\\nOutput: [128,?,4096]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Gate computation - determines expert selection
            for gpu in range(16):
                c.node(f'gate_l{layer}_gpu{gpu}', 
                       f'Gate\\nLayer {layer}\\nGPU {gpu}\\nInput: [128,?,4096]\\nOutput: [128,?,16]', 
                       shape='parallelogram', style='filled', fillcolor='lightblue')
            
            # Single expert per GPU
            for gpu in range(16):
                c.node(f'expert_l{layer}_gpu{gpu}', 
                       f'Expert {gpu+1}\\nLayer {layer}\\nGPU {gpu}\\nInput: [128,?,4096]\\nOutput: [128,?,4096]', 
                       shape='rectangle', style='filled', fillcolor=f'light{gpu+1}')
            
            # Expert aggregation per GPU
            for gpu in range(16):
                c.node(f'agg_l{layer}_gpu{gpu}', 
                       f'Aggregate\\nLayer {layer}\\nGPU {gpu}\\nInput: [128,?,4096]\\nOutput: [128,?,4096]', 
                       shape='parallelogram', style='filled', fillcolor='orange')
            
            # Global aggregation across all experts
            c.node(f'global_agg_l{layer}', f'Global Aggregation\\nLayer {layer}\\nAll GPUs\\n[128,10000,4096]', 
                   shape='ellipse', style='filled', fillcolor='pink')
            
            # Residual connection
            c.node(f'residual_l{layer}', f'Residual Add\\nLayer {layer}\\nAll GPUs\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
    
    # Output node
    dot.node('output', 'Output\\nBatch: 128\\nSeq Len: 10,000\\nDim: 4096', 
             shape='ellipse', style='filled', fillcolor='yellow')
    
    # Connections for proposed DAG
    dot.edge('input', 'token_dist')
    
    for layer in range(1, 17):
        if layer == 1:
            dot.edge('token_dist', 'dist_l1')
        else:
            dot.edge(f'residual_l{layer-1}', f'dist_l{layer}')
        
        # Distribute to all GPUs
        for gpu in range(16):
            dot.edge(f'dist_l{layer}', f'mha_l{layer}_gpu{gpu}')
            dot.edge(f'mha_l{layer}_gpu{gpu}', f'gate_l{layer}_gpu{gpu}')
            
            # Gate to expert (dashed for selection)
            dot.edge(f'gate_l{layer}_gpu{gpu}', f'expert_l{layer}_gpu{gpu}', style='dashed')
            
            dot.edge(f'expert_l{layer}_gpu{gpu}', f'agg_l{layer}_gpu{gpu}')
            dot.edge(f'agg_l{layer}_gpu{gpu}', f'global_agg_l{layer}')
        
        dot.edge(f'global_agg_l{layer}', f'residual_l{layer}')
        
        # Skip connection from input
        if layer == 1:
            dot.edge('token_dist', f'residual_l1')
        else:
            dot.edge(f'residual_l{layer-1}', f'residual_l{layer}')
    
    dot.edge('residual_l16', 'output')
    
    return dot

def main():
    # Create output directory
    os.makedirs('../outputs/2025-11-24-11-31-17', exist_ok=True)
    
    # Generate baseline DAG
    baseline_dag = create_baseline_dag()
    baseline_dag.render('../outputs/2025-11-24-11-31-17/baseline_moe_dag', format='svg', cleanup=False)
    baseline_dag.render('../outputs/2025-11-24-11-31-17/baseline_moe_dag', format='dot', cleanup=False)
    
    # Generate proposed DAG
    proposed_dag = create_proposed_dag()
    proposed_dag.render('../outputs/2025-11-24-11-31-17/proposed_moe_dag', format='svg', cleanup=False)
    proposed_dag.render('../outputs/2025-11-24-11-31-17/proposed_moe_dag', format='dot', cleanup=False)
    
    print("Generated DAGs:")
    print("- baseline_moe_dag.svg")
    print("- baseline_moe_dag.dot")
    print("- proposed_moe_dag.svg")
    print("- proposed_moe_dag.dot")

if __name__ == "__main__":
    main()