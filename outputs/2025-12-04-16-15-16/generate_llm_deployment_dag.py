import graphviz
from graphviz import Digraph
import os

def create_llm_deployment_dag():
    # Create a new directed graph
    dot = Digraph(comment='LLM Deployment DAG - 512 GPUs with Hybrid Parallelism')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('node', fontsize='10', margin='0.1,0.05')
    
    # Define node shapes and styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow') # Routing/Aggregation
    
    # Model specifications
    total_layers = 16
    total_experts = 64
    hidden_size = 1024
    attention_heads = 16
    batch_size = "?"
    seq_len = "?"
    
    # Parallel configuration
    tp_size = 8   # Tensor Parallel
    pp_size = 4   # Pipeline Parallel
    ep_size = 16  # Expert Parallel
    dp_size = 4   # Data Parallel
    
    # GPU configuration
    total_gpus = 512
    gpus_per_node = 8
    total_nodes = 64
    
    # Create subgraphs for different GPU groups
    for node_id in range(total_nodes):
        with dot.subgraph(name=f'cluster_node_{node_id}') as node_cluster:
            node_cluster.attr(label=f'Node {node_id} (GPUs {node_id*8}-{(node_id+1)*8-1})', 
                            style='rounded,filled', fillcolor='lightgray', fontsize='12')
            
            # Within each node, create GPU clusters
            for gpu_local in range(gpus_per_node):
                gpu_global = node_id * gpus_per_node + gpu_local
                
                # Create pipeline stage subgraph
                pipeline_stage = gpu_global // (total_gpus // pp_size)
                
                with node_cluster.subgraph(name=f'cluster_gpu_{gpu_global}') as gpu_cluster:
                    gpu_cluster.attr(label=f'GPU {gpu_global} (PP Stage {pipeline_stage})', 
                                   style='dashed', fillcolor='white', fontsize='10')
                    
                    # Create tensor parallel group
                    tp_group = gpu_global // tp_size
                    tp_rank = gpu_global % tp_size
                    
                    # Create expert parallel group
                    ep_group = gpu_global // (total_gpus // ep_size)
                    ep_rank = gpu_global % (total_gpus // ep_size)
                    
                    # Create data parallel group
                    dp_group = gpu_global // (total_gpus // dp_size)
                    dp_rank = gpu_global % (total_gpus // dp_size)
                    
                    # Input Layer (Data Parallel)
                    input_node = f'input_gpu_{gpu_global}'
                    gpu_cluster.node(input_node, 
                                   f'Input\\nGPU {gpu_global}\\nDP Group {dp_group}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                   shape='rectangle', fillcolor='lightblue')
                    
                    # Layer processing for each of the 16 layers
                    for layer_id in range(total_layers):
                        # Experts assigned to this GPU (2 experts per GPU based on optimization)
                        expert_start = (gpu_global * 2) % total_experts
                        expert_end = (expert_start + 2) % total_experts
                        
                        # Layer norm (Tensor Parallel)
                        layernorm_node = f'layernorm_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(layernorm_node,
                                       f'LayerNorm L{layer_id}\\nTP Rank {tp_rank}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # Attention Query (Column Parallel)
                        attn_q_node = f'attn_q_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(attn_q_node,
                                       f'Attention Query\\nL{layer_id} Col-Parallel\\nTP Rank {tp_rank}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, d_k={hidden_size//attention_heads}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # Attention Key (Column Parallel)
                        attn_k_node = f'attn_k_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(attn_k_node,
                                       f'Attention Key\\nL{layer_id} Col-Parallel\\nTP Rank {tp_rank}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, d_k={hidden_size//attention_heads}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # Attention Value (Column Parallel)
                        attn_v_node = f'attn_v_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(attn_v_node,
                                       f'Attention Value\\nL{layer_id} Col-Parallel\\nTP Rank {tp_rank}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, d_k={hidden_size//attention_heads}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # Attention Score Computation
                        attn_score_node = f'attn_score_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(attn_score_node,
                                       f'Attention Score\\nL{layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, d_k={hidden_size//attention_heads}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, seq={seq_len}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # Attention Softmax
                        attn_softmax_node = f'attn_softmax_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(attn_softmax_node,
                                       f'Attention Softmax\\nL{layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, seq={seq_len}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, seq={seq_len}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # Attention Output (Row Parallel)
                        attn_out_node = f'attn_out_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(attn_out_node,
                                       f'Attention Output\\nL{layer_id} Row-Parallel\\nTP Rank {tp_rank}\\nInput: [batch={batch_size}, seq={seq_len}, heads={attention_heads//tp_size}, d_k={hidden_size//attention_heads}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # All-Reduce for Attention Output
                        attn_allreduce_node = f'attn_allreduce_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(attn_allreduce_node,
                                       f'All-Reduce\\nAttention L{layer_id}\\nTP Group {tp_group}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                       shape='ellipse', fillcolor='lightgreen')
                        
                        # MLP Gate (Expert Selection)
                        mlp_gate_node = f'mlp_gate_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(mlp_gate_node,
                                       f'MLP Gate\\nL{layer_id} EP Rank {ep_rank}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, experts=2]',
                                       shape='parallelogram', fillcolor='yellow')
                        
                        # Expert 1 Processing
                        expert1_node = f'expert1_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(expert1_node,
                                       f'Expert {expert_start}\\nL{layer_id} GPU {gpu_global}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # Expert 2 Processing
                        expert2_node = f'expert2_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(expert2_node,
                                       f'Expert {expert_start+1}\\nL{layer_id} GPU {gpu_global}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # Expert Aggregation
                        expert_agg_node = f'expert_agg_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(expert_agg_node,
                                       f'Expert Aggregation\\nL{layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, experts=2, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                       shape='parallelogram', fillcolor='yellow')
                        
                        # MLP Output (Column-Parallel First Layer)
                        mlp_fc1_node = f'mlp_fc1_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(mlp_fc1_node,
                                       f'MLP FC1\\nL{layer_id} Col-Parallel\\nTP Rank {tp_rank}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, ffn={4096//tp_size}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # MLP GELU Activation
                        mlp_gelu_node = f'mlp_gelu_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(mlp_gelu_node,
                                       f'MLP GELU\\nL{layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, ffn={4096//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, ffn={4096//tp_size}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # MLP Output (Row-Parallel Second Layer)
                        mlp_fc2_node = f'mlp_fc2_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(mlp_fc2_node,
                                       f'MLP FC2\\nL{layer_id} Row-Parallel\\nTP Rank {tp_rank}\\nInput: [batch={batch_size}, seq={seq_len}, ffn={4096//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]',
                                       shape='rectangle', fillcolor='lightblue')
                        
                        # All-Reduce for MLP Output
                        mlp_allreduce_node = f'mlp_allreduce_l{layer_id}_gpu_{gpu_global}'
                        gpu_cluster.node(mlp_allreduce_node,
                                       f'All-Reduce\\nMLP L{layer_id}\\nTP Group {tp_group}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//tp_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                       shape='ellipse', fillcolor='lightgreen')
                        
                        # Connect nodes within layer
                        if layer_id == 0:
                            gpu_cluster.edge(input_node, layernorm_node)
                        
                        gpu_cluster.edge(layernorm_node, attn_q_node)
                        gpu_cluster.edge(layernorm_node, attn_k_node)
                        gpu_cluster.edge(layernorm_node, attn_v_node)
                        
                        gpu_cluster.edge(attn_q_node, attn_score_node)
                        gpu_cluster.edge(attn_k_node, attn_score_node)
                        gpu_cluster.edge(attn_score_node, attn_softmax_node)
                        gpu_cluster.edge(attn_softmax_node, attn_v_node, attn_out_node)
                        gpu_cluster.edge(attn_out_node, attn_allreduce_node)
                        
                        gpu_cluster.edge(attn_allreduce_node, mlp_gate_node)
                        gpu_cluster.edge(mlp_gate_node, expert1_node, style='dashed')
                        gpu_cluster.edge(mlp_gate_node, expert2_node, style='dashed')
                        gpu_cluster.edge(expert1_node, expert_agg_node)
                        gpu_cluster.edge(expert2_node, expert_agg_node)
                        gpu_cluster.edge(expert_agg_node, mlp_fc1_node)
                        gpu_cluster.edge(mlp_fc1_node, mlp_gelu_node)
                        gpu_cluster.edge(mlp_gelu_node, mlp_fc2_node)
                        gpu_cluster.edge(mlp_fc2_node, mlp_allreduce_node)
                        
                        # Connect to next layer or output
                        if layer_id < total_layers - 1:
                            next_layernorm = f'layernorm_l{layer_id+1}_gpu_{gpu_global}'
                            gpu_cluster.edge(mlp_allreduce_node, next_layernorm)
                        else:
                            # Output layer
                            output_node = f'output_gpu_{gpu_global}'
                            gpu_cluster.node(output_node,
                                           f'Output\\nGPU {gpu_global}\\nDP Group {dp_group}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                           shape='rectangle', fillcolor='lightblue')
                            gpu_cluster.edge(mlp_allreduce_node, output_node)
    
    # Add inter-GPU communication edges
    # Tensor Parallelism communications
    for tp_group in range(total_gpus // tp_size):
        for tp_rank in range(tp_size):
            gpu_id = tp_group * tp_size + tp_rank
            for layer_id in range(total_layers):
                # All-reduce for attention
                attn_allreduce = f'attn_allreduce_l{layer_id}_gpu_{gpu_id}'
                for other_rank in range(tp_size):
                    if other_rank != tp_rank:
                        other_gpu = tp_group * tp_size + other_rank
                        other_allreduce = f'attn_allreduce_l{layer_id}_gpu_{other_gpu}'
                        dot.edge(attn_allreduce, other_allreduce, 
                               style='dashed', color='red', constraint='false')
                
                # All-reduce for MLP
                mlp_allreduce = f'mlp_allreduce_l{layer_id}_gpu_{gpu_id}'
                for other_rank in range(tp_size):
                    if other_rank != tp_rank:
                        other_gpu = tp_group * tp_size + other_rank
                        other_allreduce = f'mlp_allreduce_l{layer_id}_gpu_{other_gpu}'
                        dot.edge(mlp_allreduce, other_allreduce,
                               style='dashed', color='blue', constraint='false')
    
    # Expert Parallelism communications (gate routing)
    for ep_group in range(ep_size):
        for gpu_in_group in range(total_gpus // ep_size):
            gpu_id = ep_group * (total_gpus // ep_size) + gpu_in_group
            for layer_id in range(total_layers):
                gate_node = f'mlp_gate_l{layer_id}_gpu_{gpu_id}'
                # Connect to other experts in the EP group
                for other_gpu_in_group in range(total_gpus // ep_size):
                    if other_gpu_in_group != gpu_in_group:
                        other_gpu = ep_group * (total_gpus // ep_size) + other_gpu_in_group
                        other_expert1 = f'expert1_l{layer_id}_gpu_{other_gpu}'
                        other_expert2 = f'expert2_l{layer_id}_gpu_{other_gpu}'
                        dot.edge(gate_node, other_expert1, style='dashed', color='green', constraint='false')
                        dot.edge(gate_node, other_expert2, style='dashed', color='green', constraint='false')
    
    # Data Parallelism communications (gradient synchronization)
    for dp_group in range(dp_size):
        for gpu_in_group in range(total_gpus // dp_size):
            gpu_id = dp_group * (total_gpus // dp_size) + gpu_in_group
            output_node = f'output_gpu_{gpu_id}'
            # Connect to other GPUs in DP group for gradient sync
            for other_gpu_in_group in range(total_gpus // dp_size):
                if other_gpu_in_group != gpu_in_group:
                    other_gpu = dp_group * (total_gpus // dp_size) + other_gpu_in_group
                    other_output = f'output_gpu_{other_gpu}'
                    dot.edge(output_node, other_output, 
                           style='dashed', color='purple', constraint='false')
    
    return dot

def main():
    # Create the DAG
    dag = create_llm_deployment_dag()
    
    # Save the DOT file
    dot_file_path = '../outputs/2025-12-04-16-15-16/llm_deployment_dag.dot'
    dag.save(dot_file_path)
    
    # Render to SVG
    svg_file_path = '../outputs/2025-12-04-16-15-16/llm_deployment_dag.svg'
    dag.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    
    # Also render to PNG for better viewing
    png_file_path = '../outputs/2025-12-04-16-15-16/llm_deployment_dag.png'
    dag.render(png_file_path.replace('.png', ''), format='png', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")
    print(f"PNG file: {png_file_path}")
    
    return {
        "dot_file": dot_file_path,
        "svg_file": svg_file_path,
        "png_file": png_file_path
    }

if __name__ == "__main__":
    result = main()
    print(result)