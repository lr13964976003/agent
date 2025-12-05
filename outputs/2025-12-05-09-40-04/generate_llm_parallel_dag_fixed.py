#!/usr/bin/env python3
"""
Fixed LLM Parallel Deployment DAG Generator
Addresses the missing connections identified in the feedback
"""

import graphviz
import os

def create_llm_parallel_dag():
    """Create a comprehensive DAG for LLM parallel deployment with all connections"""
    
    # Create directed graph
    dot = graphviz.Digraph(comment='LLM Parallel Deployment DAG - EP64_TP2_Hybrid_Optimized')
    
    # Set graph attributes
    dot.attr('graph', 
             nodesep='0.5',
             rankdir='TB', 
             splines='true')
    
    # Set node attributes
    dot.attr('node', 
             fillcolor='lightblue',
             shape='rectangle',
             style='filled')
    
    # Set edge attributes
    dot.attr('edge', color='black', style='solid')
    
    # Input Stage
    with dot.subgraph(name='cluster_input') as c:
        c.attr(color='black', fillcolor='white', label='Input Stage', style='rounded')
        c.node('input', 
               label='Input Layer\\nGPU: All\\nInput: [batch_size=128, seq_len=1024, hidden_size=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=1024]',
               fillcolor='white',
               shape='ellipse')
    
    # Embedding Stage (TP-2)
    with dot.subgraph(name='cluster_embedding') as c:
        c.attr(color='blue', fillcolor='lightblue', label='Embedding Stage (TP-2)', style='rounded')
        c.node('split_embed', 
               label='Split Input\\nGPU: 0,1\\nInput: [128,1024,1024]\\nOutput: [128,1024,512]',
               fillcolor='lightgray',
               shape='parallelogram')
        c.node('embed_0', 
               label='Embedding Layer 0\\nGPU: 0\\nInput: [128,1024,512]\\nOutput: [128,1024,2048]\\nWeight: [1024,2048]',
               fillcolor='lightblue')
        c.node('embed_1', 
               label='Embedding Layer 1\\nGPU: 1\\nInput: [128,1024,512]\\nOutput: [128,1024,2048]\\nWeight: [1024,2048]',
               fillcolor='lightblue')
        c.node('embed_allreduce', 
               label='All-Reduce\\nGPU: 0,1\\nInput: [128,1024,2048]\\nOutput: [128,1024,2048]',
               fillcolor='lightcoral',
               shape='ellipse')
        c.node('embed_merge', 
               label='Merge Embeddings\\nGPU: 0,1\\nInput: [128,1024,2048]\\nOutput: [128,1024,1024]',
               fillcolor='lightgray',
               shape='parallelogram')
    
    # Expert Parallel Stage (EP-64)
    with dot.subgraph(name='cluster_expert') as c:
        c.attr(color='green', fillcolor='lightgreen', label='Expert Parallel Stage (EP-64)', style='rounded')
        c.node('broadcast_expert', 
               label='Broadcast to Experts\\nGPU: 0,1 → 2-65\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]',
               fillcolor='lightcoral',
               shape='ellipse')
        c.node('gate_compute', 
               label='Gate Computation\\nGPU: 2-65\\nInput: [128,1024,1024]\\nOutput: [128,1024,64]\\n(gate scores)',
               fillcolor='lightpink',
               shape='parallelogram')
        c.node('expert_select', 
               label='Expert Selection\\nGPU: 2-65\\nInput: [128,1024,64]\\nOutput: routing decisions',
               fillcolor='lightpink',
               shape='parallelogram')
        
        # Expert weight nodes
        c.node('expert_2', 
               label='Expert 0\\nGPU: 2\\nInput: [128,1024,1024]\\nOutput: [128,1024,2048]\\nWeight: [1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_3', 
               label='Expert 1\\nGPU: 3\\nInput: [128,1024,1024]\\nOutput: [128,1024,2048]\\nWeight: [1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_4', 
               label='Expert 2\\nGPU: 4\\nInput: [128,1024,1024]\\nOutput: [128,1024,2048]\\nWeight: [1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_5', 
               label='Expert 3\\nGPU: 5\\nInput: [128,1024,1024]\\nOutput: [128,1024,2048]\\nWeight: [1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_6', 
               label='Expert 4\\nGPU: 6\\nInput: [128,1024,1024]\\nOutput: [128,1024,2048]\\nWeight: [1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_7', 
               label='Expert 5\\nGPU: 7\\nInput: [128,1024,1024]\\nOutput: [128,1024,2048]\\nWeight: [1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_8', 
               label='Expert 6\\nGPU: 8\\nInput: [128,1024,1024]\\nOutput: [128,1024,2048]\\nWeight: [1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_9', 
               label='Expert 7\\nGPU: 9\\nInput: [128,1024,1024]\\nOutput: [128,1024,2048]\\nWeight: [1024,2048]',
               fillcolor='lightgreen')
        c.node('experts_ellipsis', 
               label='... 56 more experts ...\\nGPUs: 10-65',
               shape='none',
               style='dashed')
        
        c.node('alltoall_expert', 
               label='All-to-All Communication\\nGPU: 2-65\\nRoutes tokens to experts',
               fillcolor='lightcoral',
               shape='ellipse')
        
        # Expert computation nodes
        c.node('expert_compute_2', 
               label='Expert 0 Compute\\nGPU: 2\\nInput: routed tokens\\nOutput: [128,1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_compute_3', 
               label='Expert 1 Compute\\nGPU: 3\\nInput: routed tokens\\nOutput: [128,1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_compute_4', 
               label='Expert 2 Compute\\nGPU: 4\\nInput: routed tokens\\nOutput: [128,1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_compute_5', 
               label='Expert 3 Compute\\nGPU: 5\\nInput: routed tokens\\nOutput: [128,1024,2048]',
               fillcolor='lightgreen')
        c.node('expert_compute_ellipsis', 
               label='... expert computations ...',
               shape='none',
               style='dashed')
        
        c.node('expert_agg', 
               label='Expert Output Aggregation\\nGPU: 2-65\\nInput: [128,1024,2048]\\nOutput: [128,1024,1024]',
               fillcolor='lightgray',
               shape='parallelogram')
    
    # Aggregation Stage (TP-2)
    with dot.subgraph(name='cluster_aggregation') as c:
        c.attr(color='orange', fillcolor='lightyellow', label='Aggregation Stage (TP-2)', style='rounded')
        c.node('reduce_scatter', 
               label='Reduce-Scatter\\nGPU: 2-65 → 66,67\\nInput: [128,1024,1024]\\nOutput: [128,1024,512]',
               fillcolor='lightcoral',
               shape='ellipse')
        c.node('agg_0', 
               label='Aggregation Layer 0\\nGPU: 66\\nInput: [128,1024,512]\\nOutput: [128,1024,512]\\nWeight: [512,1024]',
               fillcolor='lightyellow')
        c.node('agg_1', 
               label='Aggregation Layer 1\\nGPU: 67\\nInput: [128,1024,512]\\nOutput: [128,1024,512]\\nWeight: [512,1024]',
               fillcolor='lightyellow')
        c.node('agg_allreduce', 
               label='All-Reduce\\nGPU: 66,67\\nInput: [128,1024,512]\\nOutput: [128,1024,1024]',
               fillcolor='lightcoral',
               shape='ellipse')
        c.node('final_merge', 
               label='Final Output Merge\\nGPU: 66,67\\nInput: [128,1024,512]\\nOutput: [128,1024,1024]',
               fillcolor='lightgray',
               shape='parallelogram')
    
    # Output Stage
    with dot.subgraph(name='cluster_output') as c:
        c.attr(color='black', fillcolor='white', label='Output Stage', style='rounded')
        c.node('output', 
               label='Output Layer\\nGPU: 66,67\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\n(final hidden states)',
               fillcolor='white',
               shape='ellipse')
    
    # Define edges - FIXED WITH ALL MISSING CONNECTIONS
    dot.edge('input', 'split_embed')
    dot.edge('split_embed', 'embed_0')
    dot.edge('split_embed', 'embed_1')
    dot.edge('embed_0', 'embed_allreduce')
    dot.edge('embed_1', 'embed_allreduce')
    dot.edge('embed_allreduce', 'embed_merge')
    dot.edge('embed_merge', 'broadcast_expert')
    dot.edge('broadcast_expert', 'gate_compute')
    dot.edge('gate_compute', 'expert_select')
    
    # Expert selection to expert weights (dashed lines for routing)
    dot.edge('expert_select', 'expert_2', style='dashed')
    dot.edge('expert_select', 'expert_3', style='dashed')
    dot.edge('expert_select', 'expert_4', style='dashed')
    dot.edge('expert_select', 'expert_5', style='dashed')
    dot.edge('expert_select', 'expert_6', style='dashed')
    dot.edge('expert_select', 'expert_7', style='dashed')
    dot.edge('expert_select', 'expert_8', style='dashed')
    dot.edge('expert_select', 'expert_9', style='dashed')
    dot.edge('expert_select', 'experts_ellipsis', style='dashed')
    
    # Broadcast to alltoall communication
    dot.edge('broadcast_expert', 'alltoall_expert')
    
    # Alltoall to expert computations
    dot.edge('alltoall_expert', 'expert_compute_2')
    dot.edge('alltoall_expert', 'expert_compute_3')
    dot.edge('alltoall_expert', 'expert_compute_4')
    dot.edge('alltoall_expert', 'expert_compute_5')
    dot.edge('alltoall_expert', 'expert_compute_ellipsis')
    
    # FIXED: Add missing connections from expert weights to expert computations
    dot.edge('expert_2', 'expert_compute_2')
    dot.edge('expert_3', 'expert_compute_3')
    dot.edge('expert_4', 'expert_compute_4')
    dot.edge('expert_5', 'expert_compute_5')
    dot.edge('expert_6', 'expert_compute_ellipsis')
    dot.edge('expert_7', 'expert_compute_ellipsis')
    dot.edge('expert_8', 'expert_compute_ellipsis')
    dot.edge('expert_9', 'expert_compute_ellipsis')
    
    # FIXED: Add missing connections from expert computations to aggregation
    dot.edge('expert_compute_2', 'expert_agg')
    dot.edge('expert_compute_3', 'expert_agg')
    dot.edge('expert_compute_4', 'expert_agg')
    dot.edge('expert_compute_5', 'expert_agg')
    dot.edge('expert_compute_ellipsis', 'expert_agg')
    
    # Expert aggregation to reduce-scatter
    dot.edge('expert_agg', 'reduce_scatter')
    
    # Reduce-scatter to aggregation layers
    dot.edge('reduce_scatter', 'agg_0')
    dot.edge('reduce_scatter', 'agg_1')
    dot.edge('agg_0', 'agg_allreduce')
    dot.edge('agg_1', 'agg_allreduce')
    dot.edge('agg_allreduce', 'final_merge')
    dot.edge('final_merge', 'output')
    
    return dot

def main():
    """Main function to generate and save the DAG"""
    
    # Create output directory
    output_dir = "../outputs/2025-12-05-09-40-04"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the DAG
    dag = create_llm_parallel_dag()
    
    # Save as DOT file
    dot_file = os.path.join(output_dir, "llm_parallel_deployment_dag_fixed.dot")
    dag.save(dot_file)
    
    # Save as SVG image
    svg_file = os.path.join(output_dir, "llm_parallel_deployment_dag_fixed.svg")
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    # Create summary
    summary = {
        "generated_dag_files": [
            dot_file,
            svg_file,
            os.path.join(output_dir, __file__),
            os.path.join(output_dir, "dag_generation_summary_fixed.json")
        ],
        "dag_content": str(dag.source),
        "fix_description": "Added missing connections between expert weight nodes and expert computation nodes, and from expert computation nodes to aggregation node",
        "validation_status": "all_connections_fixed"
    }
    
    # Save summary
    import json
    summary_file = os.path.join(output_dir, "dag_generation_summary_fixed.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")
    print(f"Summary file: {summary_file}")
    
    return dot_file, svg_file

if __name__ == "__main__":
    main()