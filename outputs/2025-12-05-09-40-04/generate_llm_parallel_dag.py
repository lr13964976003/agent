#!/usr/bin/env python3

import graphviz
import os

def create_llm_parallel_dag():
    """
    Create a detailed DAG for LLM parallel deployment with EP64_TP2_Hybrid_Optimized strategy
    """
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='LLM Parallel Deployment DAG - EP64_TP2_Hybrid_Optimized')
    dot.attr(rankdir='TB', splines='true', nodesep='0.5')
    
    # Define node shapes and styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('edge', style='solid', color='black')
    
    # Define colors for different GPU groups
    colors = {
        'embedding': 'lightblue',
        'expert': 'lightgreen', 
        'aggregation': 'lightyellow',
        'communication': 'lightcoral',
        'routing': 'lightpink',
        'split': 'lightgray',
        'merge': 'lightgray'
    }
    
    # === INPUT STAGE ===
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Stage', style='rounded', fillcolor='white', color='black')
        c.node('input', 'Input Layer\nGPU: All\nInput: [batch_size=128, seq_len=1024, hidden_size=1024]\nOutput: [batch_size=128, seq_len=1024, hidden_size=1024]', 
               shape='ellipse', fillcolor='white')
    
    # === EMBEDDING STAGE (Tensor Parallel) ===
    with dot.subgraph(name='cluster_embedding') as c:
        c.attr(label='Embedding Stage (TP-2)', style='rounded', fillcolor='lightblue', color='blue')
        
        # Input split for embedding
        c.node('split_embed', 'Split Input\nGPU: 0,1\nInput: [128,1024,1024]\nOutput: [128,1024,512]', 
               shape='parallelogram', fillcolor=colors['split'])
        
        # Column-parallel embedding on GPU 0
        c.node('embed_0', 'Embedding Layer 0\nGPU: 0\nInput: [128,1024,512]\nOutput: [128,1024,2048]\nWeight: [1024,2048]', 
               fillcolor=colors['embedding'])
        
        # Column-parallel embedding on GPU 1  
        c.node('embed_1', 'Embedding Layer 1\nGPU: 1\nInput: [128,1024,512]\nOutput: [128,1024,2048]\nWeight: [1024,2048]', 
               fillcolor=colors['embedding'])
        
        # All-reduce communication
        c.node('embed_allreduce', 'All-Reduce\nGPU: 0,1\nInput: [128,1024,2048]\nOutput: [128,1024,2048]', 
               shape='ellipse', fillcolor=colors['communication'])
        
        # Merge after all-reduce
        c.node('embed_merge', 'Merge Embeddings\nGPU: 0,1\nInput: [128,1024,2048]\nOutput: [128,1024,1024]', 
               shape='parallelogram', fillcolor=colors['merge'])
    
    # === EXPERT PARALLEL STAGE ===
    with dot.subgraph(name='cluster_expert') as c:
        c.attr(label='Expert Parallel Stage (EP-64)', style='rounded', fillcolor='lightgreen', color='green')
        
        # Broadcast input to all expert GPUs
        c.node('broadcast_expert', 'Broadcast to Experts\nGPU: 0,1 → 2-65\nInput: [128,1024,1024]\nOutput: [128,1024,1024]', 
               shape='ellipse', fillcolor=colors['communication'])
        
        # Gate computation (routing)
        c.node('gate_compute', 'Gate Computation\nGPU: 2-65\nInput: [128,1024,1024]\nOutput: [128,1024,64]\n(gate scores)', 
               shape='parallelogram', fillcolor=colors['routing'])
        
        # Expert selection (dashed line for gate routing)
        c.node('expert_select', 'Expert Selection\nGPU: 2-65\nInput: [128,1024,64]\nOutput: routing decisions', 
               shape='parallelogram', fillcolor=colors['routing'])
        
        # Individual experts (showing a few as examples)
        expert_gpus = list(range(2, 66))
        for i, gpu_id in enumerate(expert_gpus[:8]):  # Show first 8 experts
            c.node(f'expert_{gpu_id}', f'Expert {i}\nGPU: {gpu_id}\nInput: [128,1024,1024]\nOutput: [128,1024,2048]\nWeight: [1024,2048]', 
                   fillcolor=colors['expert'])
        
        # Show ellipsis for remaining experts
        c.node('experts_ellipsis', '... 56 more experts ...\nGPUs: 10-65', 
               shape='none', style='dashed')
        
        # All-to-All communication for expert routing
        c.node('alltoall_expert', 'All-to-All Communication\nGPU: 2-65\nRoutes tokens to experts', 
               shape='ellipse', fillcolor=colors['communication'])
        
        # Expert computation after routing
        for i, gpu_id in enumerate(expert_gpus[:4]):  # Show first 4 after routing
            c.node(f'expert_compute_{gpu_id}', f'Expert {i} Compute\nGPU: {gpu_id}\nInput: routed tokens\nOutput: [128,1024,2048]', 
                   fillcolor=colors['expert'])
        
        c.node('expert_compute_ellipsis', '... expert computations ...', 
               shape='none', style='dashed')
        
        # Expert output aggregation
        c.node('expert_agg', 'Expert Output Aggregation\nGPU: 2-65\nInput: [128,1024,2048]\nOutput: [128,1024,1024]', 
               shape='parallelogram', fillcolor=colors['merge'])
    
    # === AGGREGATION STAGE ===
    with dot.subgraph(name='cluster_aggregation') as c:
        c.attr(label='Aggregation Stage (TP-2)', style='rounded', fillcolor='lightyellow', color='orange')
        
        # Reduce-scatter from experts to aggregation GPUs
        c.node('reduce_scatter', 'Reduce-Scatter\nGPU: 2-65 → 66,67\nInput: [128,1024,1024]\nOutput: [128,1024,512]', 
               shape='ellipse', fillcolor=colors['communication'])
        
        # Row-parallel aggregation on GPU 66
        c.node('agg_0', 'Aggregation Layer 0\nGPU: 66\nInput: [128,1024,512]\nOutput: [128,1024,512]\nWeight: [512,1024]', 
               fillcolor=colors['aggregation'])
        
        # Row-parallel aggregation on GPU 67
        c.node('agg_1', 'Aggregation Layer 1\nGPU: 67\nInput: [128,1024,512]\nOutput: [128,1024,512]\nWeight: [512,1024]', 
               fillcolor=colors['aggregation'])
        
        # All-reduce for final aggregation
        c.node('agg_allreduce', 'All-Reduce\nGPU: 66,67\nInput: [128,1024,512]\nOutput: [128,1024,1024]', 
               shape='ellipse', fillcolor=colors['communication'])
        
        # Final merge
        c.node('final_merge', 'Final Output Merge\nGPU: 66,67\nInput: [128,1024,512]\nOutput: [128,1024,1024]', 
               shape='parallelogram', fillcolor=colors['merge'])
    
    # === OUTPUT STAGE ===
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Stage', style='rounded', fillcolor='white', color='black')
        c.node('output', 'Output Layer\nGPU: 66,67\nInput: [128,1024,1024]\nOutput: [128,1024,1024]\n(final hidden states)', 
               shape='ellipse', fillcolor='white')
    
    # Define edges (connections between nodes)
    # Input to embedding
    dot.edge('input', 'split_embed')
    dot.edge('split_embed', 'embed_0')
    dot.edge('split_embed', 'embed_1')
    dot.edge('embed_0', 'embed_allreduce')
    dot.edge('embed_1', 'embed_allreduce')
    dot.edge('embed_allreduce', 'embed_merge')
    
    # Embedding to expert stage
    dot.edge('embed_merge', 'broadcast_expert')
    dot.edge('broadcast_expert', 'gate_compute')
    dot.edge('gate_compute', 'expert_select')
    
    # Connect experts (showing pattern)
    for gpu_id in expert_gpus[:8]:
        dot.edge('expert_select', f'expert_{gpu_id}', style='dashed')  # Dashed for gate routing
    dot.edge('expert_select', 'experts_ellipsis', style='dashed')
    
    # Expert computation flow
    dot.edge('broadcast_expert', 'alltoall_expert')
    for gpu_id in expert_gpus[:4]:
        dot.edge('alltoall_expert', f'expert_compute_{gpu_id}')
    dot.edge('alltoall_expert', 'expert_compute_ellipsis')
    
    # Expert to aggregation
    dot.edge('expert_agg', 'reduce_scatter')
    dot.edge('reduce_scatter', 'agg_0')
    dot.edge('reduce_scatter', 'agg_1')
    dot.edge('agg_0', 'agg_allreduce')
    dot.edge('agg_1', 'agg_allreduce')
    dot.edge('agg_allreduce', 'final_merge')
    dot.edge('final_merge', 'output')
    
    return dot

def main():
    # Create the DAG
    dag = create_llm_parallel_dag()
    
    # Save as DOT file
    output_dir = "../outputs/2025-12-05-09-40-04"
    os.makedirs(output_dir, exist_ok=True)
    
    dot_file = os.path.join(output_dir, "llm_parallel_deployment_dag.dot")
    svg_file = os.path.join(output_dir, "llm_parallel_deployment_dag.svg")
    
    # Save DOT file
    dag.save(dot_file)
    
    # Render to SVG
    dag.render(dot_file.replace('.dot', ''), format='svg', cleanup=False)
    
    print(f"DAG files created:")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")
    
    return dot_file, svg_file

if __name__ == "__main__":
    main()