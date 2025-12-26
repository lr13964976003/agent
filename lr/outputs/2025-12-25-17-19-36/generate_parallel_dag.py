#!/usr/bin/env python3

import graphviz
import os

def create_parallel_strategy_dag():
    """
    Create a comprehensive DAG for the 24-GPU parallel strategy deployment
    showing all parallel dimensions with proper node representations
    """
    
    # Create the main DAG
    dot = graphviz.Digraph(comment='24-GPU Parallel Strategy Deployment DAG')
    dot.attr(rankdir='TB', size='20,30', dpi='300')
    dot.attr('node', fontsize='10')
    dot.attr('edge', fontsize='8')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input layer
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer', style='rounded', fillcolor='lightgray')
        c.node('input', 'Input\nInput: [batch_size=128, seq_len=10240, heads=16, d_k=32]\nOutput: [batch_size=128, seq_len=10240, heads=16, d_k=32]', 
               shape='rectangle', fillcolor='lightgreen')
    
    # Data Parallel split (3 groups)
    with dot.subgraph(name='cluster_dp') as c:
        c.attr(label='Data Parallel (3 groups)', style='rounded', fillcolor='lightcyan')
        
        # DP Group 0 (Ranks 0-7, 16-23)
        with dot.subgraph(name='cluster_dp0') as dp0:
            dp0.attr(label='DP Group 0', style='rounded', fillcolor='lightcyan')
            
            # Pipeline Parallel Stage 0 (Ranks 0-7)
            with dot.subgraph(name='cluster_pp0_stage0') as pp0_s0:
                pp0_s0.attr(label='Pipeline Stage 0\nRanks 0-7', style='rounded', fillcolor='lightpink')
                
                # Tensor Parallel pairs for Stage 0
                with dot.subgraph(name='cluster_tp_pairs_s0') as tp_pairs_s0:
                    tp_pairs_s0.attr(label='Tensor Parallel Groups', style='rounded', fillcolor='lightyellow')
                    
                    # TP Pair (0,1) - Stage 0
                    with dot.subgraph(name='cluster_tp_01_s0') as tp_01_s0:
                        tp_01_s0.attr(label='TP Group (0,1)', style='rounded', fillcolor='lightyellow')
                        
                        # GPU 0 - Stage 0
                        with dot.subgraph(name='cluster_gpu0_s0') as gpu0_s0:
                            gpu0_s0.attr(label='GPU 0', style='rounded', fillcolor='white')
                            gpu0_s0.node('gpu0_embed', 'Embedding Layer\nGPU: 0\nInput: [batch_size=43, seq_len=10240, heads=16, d_k=32]\nOutput: [batch_size=43, seq_len=10240, d_model=512]', 
                                        shape='rectangle', fillcolor='lightgreen')
                            gpu0_s0.node('gpu0_l0_attn', 'Layer 0 Attention\nGPU: 0\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, d_model=512]', 
                                        shape='rectangle', fillcolor='lightgreen')
                            gpu0_s0.node('gpu0_l0_moe', 'Layer 0 MoE\nGPU: 0\n8 Experts\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, d_model=512]', 
                                        shape='rectangle', fillcolor='lightgreen')
                            
                            # Expert routing for GPU 0
                            gpu0_s0.node('gpu0_l0_gate', 'Gate Router\nGPU: 0\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: routing decisions', 
                                        shape='parallelogram', fillcolor='lightyellow', style='dashed')
                        
                        # GPU 1 - Stage 0  
                        with dot.subgraph(name='cluster_gpu1_s0') as gpu1_s0:
                            gpu1_s0.attr(label='GPU 1', style='rounded', fillcolor='white')
                            gpu1_s0.node('gpu1_embed', 'Embedding Layer\nGPU: 1\nInput: [batch_size=43, seq_len=10240, heads=16, d_k=32]\nOutput: [batch_size=43, seq_len=10240, d_model=512]', 
                                        shape='rectangle', fillcolor='lightgreen')
                            gpu1_s0.node('gpu1_l0_attn', 'Layer 0 Attention\nGPU: 1\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, d_model=512]', 
                                        shape='rectangle', fillcolor='lightgreen')
                            gpu1_s0.node('gpu1_l0_moe', 'Layer 0 MoE\nGPU: 1\n8 Experts\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, d_model=512]', 
                                        shape='rectangle', fillcolor='lightgreen')
                            
                            # Expert routing for GPU 1
                            gpu1_s0.node('gpu1_l0_gate', 'Gate Router\nGPU: 1\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: routing decisions', 
                                        shape='parallelogram', fillcolor='lightyellow', style='dashed')
                    
                    # Communication between TP pairs
                    tp_pairs_s0.node('tp_01_comm', 'TP All-Reduce\nGPUs: 0,1\nInput: partial results\nOutput: aggregated results', 
                                    shape='ellipse', fillcolor='lightblue')
                    
                # Continue with more layers for Stage 0
                for layer in range(1, 4):  # Layers 1-3
                    with dot.subgraph(name=f'cluster_layer{layer}_s0') as layer_s0:
                        layer_s0.attr(label=f'Layer {layer}', style='rounded', fillcolor='white')
                        
                        # GPU 0 Layer computations
                        layer_s0.node(f'gpu0_l{layer}_attn', f'Layer {layer} Attention\nGPU: 0\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, d_model=512]', 
                                     shape='rectangle', fillcolor='lightgreen')
                        layer_s0.node(f'gpu0_l{layer}_moe', f'Layer {layer} MoE\nGPU: 0\n8 Experts\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, d_model=512]', 
                                    shape='rectangle', fillcolor='lightgreen')
                        
                        # GPU 1 Layer computations  
                        layer_s0.node(f'gpu1_l{layer}_attn', f'Layer {layer} Attention\nGPU: 1\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, d_model=512]', 
                                     shape='rectangle', fillcolor='lightgreen')
                        layer_s0.node(f'gpu1_l{layer}_moe', f'Layer {layer} MoE\nGPU: 1\n8 Experts\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: [batch_size=43, seq_len=10240, d_model=512]', 
                                    shape='rectangle', fillcolor='lightgreen')
                        
                        # Expert routing
                        layer_s0.node(f'gate_l{layer}_01', f'Gate Router\nGPUs: 0,1\nInput: [batch_size=43, seq_len=10240, d_model=512]\nOutput: routing decisions', 
                                     shape='parallelogram', fillcolor='lightyellow', style='dashed')
                        
                        # TP communication
                        layer_s0.node(f'tp_l{layer}_01_comm', f'TP All-Reduce\nGPUs: 0,1\nInput: partial results\nOutput: aggregated results', 
                                     shape='ellipse', fillcolor='lightblue')
            
            # Pipeline Stage 1 (Ranks 8-15) - Similar structure
            with dot.subgraph(name='cluster_pp0_stage1') as pp0_s1:
                pp0_s1.attr(label='Pipeline Stage 1\nRanks 8-15', style='rounded', fillcolor='lightpink')
                
                # Similar structure for Stage 1... (abbreviated for space)
                pp0_s1.node('pp0_s1_summary', 'Layers 8-15\nGPUs: 8-15\nSimilar structure\n8 layers per GPU', 
                           shape='rectangle', fillcolor='lightgray')
        
        # DP Group 1 (Ranks 16-23) - Similar structure  
        with dot.subgraph(name='cluster_dp1') as dp1:
            dp1.attr(label='DP Group 1', style='rounded', fillcolor='lightcyan')
            dp1.node('dp1_summary', 'DP Group 1\nRanks 16-23\nSame structure as DP0\nDuplicate computation', 
                    shape='rectangle', fillcolor='lightgray')
    
    # Communication layers between parallel dimensions
    with dot.subgraph(name='cluster_communication') as comm:
        comm.attr(label='Inter-GPU Communication', style='rounded', fillcolor='lightcoral')
        
        # Data Parallel communication
        comm.node('dp_allreduce', 'DP All-Sum\nGPUs: 0-23\nInput: gradient chunks\nOutput: synchronized gradients', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Pipeline Parallel communication
        comm.node('pp_sendrecv', 'PP Send/Recv\nGPUs: stage0→stage1\nInput: activations\nOutput: forwarded activations', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Expert Parallel communication (all-to-all)
        comm.node('ep_all2all', 'EP All-to-All\nGPUs: per EP group\nInput: token representations\nOutput: routed tokens', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Tensor Parallel communication
        comm.node('tp_allreduce', 'TP All-Reduce\nGPUs: per TP pair\nInput: partial tensors\nOutput: complete tensors', 
                 shape='ellipse', fillcolor='lightblue')
    
    # Aggregation/Routing layers
    with dot.subgraph(name='cluster_routing') as routing:
        routing.attr(label='Routing & Aggregation', style='rounded', fillcolor='lightgoldenrodyellow')
        
        # Global aggregation
        routing.node('global_agg', 'Global Aggregation\nInput: distributed results\nOutput: final predictions', 
                    shape='parallelogram', fillcolor='lightyellow')
        
        # Load balancing
        routing.node('load_balance', 'Load Balancer\nInput: GPU loads\nOutput: balancing decisions', 
                    shape='parallelogram', fillcolor='lightyellow')
    
    # Output layer
    with dot.subgraph(name='cluster_output') as output:
        output.attr(label='Output Layer', style='rounded', fillcolor='lightgray')
        output.node('output', 'Output\nInput: [batch_size=128, seq_len=10240, d_model=512]\nOutput: [batch_size=128, seq_len=10240, vocab_size]', 
                   shape='rectangle', fillcolor='lightgreen')
    
    # Define edges (connections)
    # Input to first computation
    dot.edge('input', 'gpu0_embed', label='batch split')
    dot.edge('input', 'gpu1_embed', label='batch split')
    
    # Within GPU 0, Stage 0
    dot.edge('gpu0_embed', 'gpu0_l0_attn')
    dot.edge('gpu0_l0_attn', 'gpu0_l0_moe')
    dot.edge('gpu0_l0_gate', 'gpu0_l0_moe', style='dashed', label='routing')
    
    # TP communication
    dot.edge('gpu0_l0_attn', 'tp_01_comm', label='partial results')
    dot.edge('gpu1_l0_attn', 'tp_01_comm', label='partial results')
    dot.edge('tp_01_comm', 'gpu0_l0_moe', label='aggregated')
    dot.edge('tp_01_comm', 'gpu1_l0_moe', label='aggregated')
    
    # Continue pattern for more layers...
    dot.edge('gpu0_l0_moe', 'gpu0_l1_attn')
    dot.edge('gpu1_l0_moe', 'gpu1_l1_attn')
    
    # Pipeline communication between stages
    dot.edge('gpu0_l3_moe', 'pp_sendrecv', label='activations')
    dot.edge('pp_sendrecv', 'pp0_s1_summary', label='forwarded')
    
    # Data parallel communication
    dot.edge('dp0_final', 'dp_allreduce', label='gradients')
    dot.edge('dp1_final', 'dp_allreduce', label='gradients')
    
    # Expert parallel communication
    dot.edge('gpu0_l0_moe', 'ep_all2all', label='tokens')
    dot.edge('gpu1_l0_moe', 'ep_all2all', label='tokens')
    
    # Final aggregation
    dot.edge('dp_allreduce', 'global_agg', label='synchronized')
    dot.edge('global_agg', 'output', label='final result')
    
    # Load balancing feedback
    dot.edge('load_balance', 'gate_l0_01', style='dashed', label='balancing')
    
    return dot

def create_simplified_dag():
    """
    Create a simplified version focusing on key parallel dimensions
    """
    dot = graphviz.Digraph(comment='Simplified 24-GPU Parallel Strategy DAG')
    dot.attr(rankdir='TB', size='15,20', dpi='300')
    dot.attr('node', fontsize='10')
    
    # Input
    dot.node('input', 'Input\n[batch=128, seq=10240]', shape='rectangle', fillcolor='lightgreen')
    
    # Data Parallel split
    dot.node('dp_split', 'Data Parallel Split\n3 groups', shape='parallelogram', fillcolor='lightyellow')
    
    # Pipeline stages
    dot.node('pp_stage0', 'Pipeline Stage 0\nLayers 0-7\nGPUs: 0-7', shape='rectangle', fillcolor='lightgreen')
    dot.node('pp_stage1', 'Pipeline Stage 1\nLayers 8-15\nGPUs: 8-15', shape='rectangle', fillcolor='lightgreen')
    
    # Tensor Parallel within stages
    dot.node('tp_group0', 'TP Group (0,1)\nAttention + MoE', shape='rectangle', fillcolor='lightgreen')
    dot.node('tp_group1', 'TP Group (2,3)\nAttention + MoE', shape='rectangle', fillcolor='lightgreen')
    
    # Expert routing
    dot.node('gate0', 'Gate Router\nGPU 0-1', shape='parallelogram', fillcolor='lightyellow', style='dashed')
    dot.node('gate1', 'Gate Router\nGPU 2-3', shape='parallelogram', fillcolor='lightyellow', style='dashed')
    
    # Communication nodes
    dot.node('tp_comm', 'TP All-Reduce\nAcross pairs', shape='ellipse', fillcolor='lightblue')
    dot.node('pp_comm', 'PP Send/Recv\nStage0→Stage1', shape='ellipse', fillcolor='lightblue')
    dot.node('dp_comm', 'DP All-Reduce\nAcross replicas', shape='ellipse', fillcolor='lightblue')
    dot.node('ep_comm', 'EP All-to-All\nExpert routing', shape='ellipse', fillcolor='lightblue')
    
    # Output
    dot.node('output', 'Output\n[batch=128, predictions]', shape='rectangle', fillcolor='lightgreen')
    
    # Edges
    dot.edge('input', 'dp_split')
    dot.edge('dp_split', 'pp_stage0', label='group 0')
    dot.edge('dp_split', 'pp_stage1', label='group 1')
    
    dot.edge('pp_stage0', 'tp_group0')
    dot.edge('pp_stage0', 'tp_group1')
    
    dot.edge('tp_group0', 'gate0', style='dashed')
    dot.edge('tp_group1', 'gate1', style='dashed')
    
    dot.edge('tp_group0', 'tp_comm')
    dot.edge('tp_group1', 'tp_comm')
    
    dot.edge('tp_comm', 'pp_comm')
    dot.edge('pp_comm', 'pp_stage1')
    
    dot.edge('pp_stage1', 'dp_comm')
    dot.edge('gate0', 'ep_comm')
    dot.edge('gate1', 'ep_comm')
    
    dot.edge('dp_comm', 'output')
    dot.edge('ep_comm', 'output')
    
    return dot

if __name__ == '__main__':
    # Create output directory
    output_dir = './outputs/2025-12-25-17-19-36'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive DAG
    print("Generating comprehensive DAG...")
    comprehensive_dag = create_parallel_strategy_dag()
    comprehensive_dag.render(os.path.join(output_dir, 'comprehensive_parallel_dag'), format='svg', cleanup=True)
    comprehensive_dag.render(os.path.join(output_dir, 'comprehensive_parallel_dag'), format='dot', cleanup=True)
    
    # Generate simplified DAG
    print("Generating simplified DAG...")
    simplified_dag = create_simplified_dag()
    simplified_dag.render(os.path.join(output_dir, 'simplified_parallel_dag'), format='svg', cleanup=True)
    simplified_dag.render(os.path.join(output_dir, 'simplified_parallel_dag'), format='dot', cleanup=True)
    
    print(f"DAGs generated in {output_dir}")
    
    # List generated files
    files = os.listdir(output_dir)
    svg_files = [f for f in files if f.endswith('.svg')]
    dot_files = [f for f in files if f.endswith('.dot')]
    
    print(f"SVG files: {svg_files}")
    print(f"DOT files: {dot_files}")