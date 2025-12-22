import graphviz
from graphviz import Digraph
import os

def create_corrected_llm_dag():
    """
    Create a corrected DAG representing the EP64-TP8-PP2-DP2 strategy for 30B MoE LLM inference.
    This version removes cycles and properly represents the parallel strategy.
    """
    
    dot = Digraph(comment='Corrected LLM EP64-TP8-PP2-DP2 DAG')
    dot.attr(rankdir='TB', bgcolor='white', fontname='Arial', size='20,30')
    
    # Define styles
    dot.attr('node', shape='rectangle', style='filled', fontname='Arial', fontsize='12')
    
    # Input
    dot.node('input', 'Input Layer\\n[128, seq_len, 1024]\\nBatch: 128, DP2', fillcolor='white')
    
    # Data parallel split (DP2)
    dot.node('dp_split', 'Data Parallel Split\\nDP2: 64 sequences each', shape='parallelogram', fillcolor='lightpinkion')
    
    # Stage 1 overview
    dot.node('stage1', 'Pipeline Stage 1\\nLayers 1-8\\nTP8-EP64\\n8 GPUs per TP group', fillcolor='lightblue')
    
    # TP synchronization within stage 1
    dot.node('tp_sync_stage1', 'TP All-Reduce Sync\\n16 operations\\nStage 1', shape='ellipse', fillcolor='lightyellow')
    
    # EP communication for stage 1
    dot.node('ep_comm_stage1', 'EP All-to-All\\n128 operations\\nExpert dispatch/combine\\nStage 1', shape='ellipse', fillcolor='lightyellow')
    
    # Routing for stage 1
    dot.node('routing_stage1', 'MoE Routing\\nGate Selection\\nStage 1', shape='parallelogram', fillcolor='lightcoral')
    
    # Experts for stage 1
    dot.node('experts_stage1', '64 Expert Computations\\nTP8 Parallel\\nStage 1', fillcolor='lightblue')
    
    # Pipeline communication
    dot.node('pp_comm', 'Pipeline Communication\\nPP Transfer\\nStage 1 -> Stage 2', shape='ellipse', fillcolor='lightyellow')
    
    # Stage 2 overview
    dot.node('stage2', 'Pipeline Stage 2\\nLayers 9-16\\nTP8-EP64\\n8 GPUs per TP group', fillcolor='lightgreen')
    
    # TP synchronization within stage 2
    dot.node('tp_sync_stage2', 'TP All-Reduce Sync\\n16 operations\\nStage 2', shape='ellipse', fillcolor='lightyellow')
    
    # EP communication for stage 2
    dot.node('ep_comm_stage2', 'EP All-to-All\\n128 operations\\nExpert dispatch/combine\\nStage 2', shape='ellipse', fillcolor='lightyellow')
    
    # Routing for stage 2
    dot.node('routing_stage2', 'MoE Routing\\nGate Selection\\nStage 2', shape='parallelogram', fillcolor='lightcoral')
    
    # Experts for stage 2
    dot.node('experts_stage2', '64 Expert Computations\\nTP8 Parallel\\nStage 2', fillcolor='lightgreen')
    
    # Data parallel merge
    dot.node('dp_merge', 'Data Parallel Merge\\nDP2: Combine results', shape='parallelogram', fillcolor='lightpinkion')
    
    # Output
    dot.node('output', 'Output Layer\\n[128, seq_len, vocab_size]\\nBatch: 128, DP2', fillcolor='white')
    
    # Connections - strict DAG with no cycles
    dot.edge('input', 'dp_split')
    dot.edge('dp_split', 'stage1')
    
    # Stage 1 operations
    dot.edge('stage1', 'routing_stage1', style='dashed')
    dot.edge('routing_stage1', 'ep_comm_stage1')
    dot.edge('ep_comm_stage1', 'experts_stage1')
    dot.edge('experts_stage1', 'ep_comm_stage1')
    dot.edge('stage1', 'tp_sync_stage1', style='dotted')
    dot.edge('tp_sync_stage1', 'stage1')
    
    # Pipeline transfer
    dot.edge('stage1', 'pp_comm')
    dot.edge('pp_comm', 'stage2')
    
    # Stage 2 operations
    dot.edge('stage2', 'routing_stage2', style='dashed')
    dot.edge('routing_stage2', 'ep_comm_stage2')
    dot.edge('ep_comm_stage2', 'experts_stage2')
    dot.edge('experts_stage2', 'ep_comm_stage2')
    dot.edge('stage2', 'tp_sync_stage2', style='dotted')
    dot.edge('tp_sync_stage2', 'stage2')
    
    # Final output
    dot.edge('stage2', 'dp_merge')
    dot.edge('dp_merge', 'output')
    
    return dot

def create_comprehensive_dag():
    """
    Create a comprehensive DAG showing the complete EP64-TP8-PP2-DP2 strategy
    with proper GPU assignments and communication patterns.
    """
    
    dot = Digraph(comment='Comprehensive LLM EP64-TP8-PP2-DP2 DAG')
    dot.attr(rankdir='LR', bgcolor='white', fontname='Arial', size='25,40')
    
    # Define GPU groups
    gpu_groups = {
        'stage1_tp': 'lightblue',
        'stage2_tp': 'lightgreen',
        'comm': 'lightyellow',
        'routing': 'lightcoral',
        'dp': 'lightpinkion'
    }
    
    # Create subgraph for each GPU configuration
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Processing\\nDP2 Split', style='rounded', fillcolor=gpu_groups['dp'])
        c.node('input', 'Input\\n[128, seq_len, 1024]', shape='ellipse', fillcolor='white')
        c.node('dp_split', 'DP Split\\n64 seq per GPU', shape='parallelogram', fillcolor=gpu_groups['dp'])
    
    # Stage 1: GPUs 0-1023 (PP1, EP64, TP8)
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Stage 1: GPUs 0-1023\\nPP1, EP64, TP8', style='rounded', fillcolor=gpu_groups['stage1_tp'])
        
        # Show TP8 groups within EP64
        for ep_group in range(8):  # 8 EP groups, each with 8 TP ranks
            with c.subgraph(name=f'cluster_ep{ep_group}_stage1') as ep:
                ep.attr(label=f'EP Group {ep_group}\\nTP8', style='dotted')
                
                # Token processing
                ep.node(f'token_proc_s1_ep{ep_group}', 
                       f'Token Processing\\nEP{ep_group}\\nInput: [64, seq_len, 1024]\\nOutput: [64, seq_len, 1024]', 
                       shape='rectangle', fillcolor=gpu_groups['stage1_tp'])
                
                # LayerNorm
                ep.node(f'ln_s1_ep{ep_group}', 
                       f'LayerNorm (TP8)\\nEP{ep_group}\\nInput: [64, seq_len, 1024]\\nOutput: [64, seq_len, 1024]', 
                       shape='rectangle', fillcolor=gpu_groups['stage1_tp'])
                
                # Attention QKV
                ep.node(f'qkv_s1_ep{ep_group}', 
                       f'QKV Proj (TP8)\\nEP{ep_group}\\nInput: [64, seq_len, 1024]\\nOutput: [64, seq_len, 16, 64]', 
                       shape='rectangle', fillcolor=gpu_groups['stage1_tp'])
                
                # Attention computation
                ep.node(f'attn_s1_ep{ep_group}', 
                       f'Self-Attention (TP8)\\nEP{ep_group}\\nInput: [64, seq_len, 16, 64]\\nOutput: [64, seq_len, 16, 64]', 
                       shape='rectangle', fillcolor=gpu_groups['stage1_tp'])
                
                # Attention output
                ep.node(f'attn_out_s1_ep{ep_group}', 
                       f'Attn Output (TP8)\\nEP{ep_group}\\nInput: [64, seq_len, 16, 64]\\nOutput: [64, seq_len, 1024]', 
                       shape='rectangle', fillcolor=gpu_groups['stage1_tp'])
                
                # MoE routing
                ep.node(f'route_s1_ep{ep_group}', 
                       f'MoE Routing\\nEP{ep_group}\\nInput: [64, seq_len, 1024]\\nOutput: [64, seq_len, 2048]', 
                       shape='parallelogram', fillcolor=gpu_groups['routing'])
                
                # Expert dispatch
                ep.node(f'dispatch_s1_ep{ep_group}', 
                       f'Expert Dispatch\\nEP{ep_group}\\nAll-to-All\\nInput: [64, seq_len, 2048]\\nOutput: [1, seq_len, 2048]', 
                       shape='ellipse', fillcolor=gpu_groups['comm'])
                
                # Expert computation
                for expert in range(8):  # Each EP group handles 8 experts
                    ep.node(f'expert_s1_ep{ep_group}_ex{expert}', 
                           f'Expert {ep_group*8+expert}\\nTP8\\nInput: [1, seq_len, 2048]\\nOutput: [1, seq_len, 2048]', 
                           shape='rectangle', fillcolor=gpu_groups['stage1_tp'])
                
                # Expert combine
                ep.node(f'combine_s1_ep{ep_group}', 
                       f'Expert Combine\\nEP{ep_group}\\nAll-to-All\\nInput: [1, seq_len, 2048]\\nOutput: [64, seq_len, 2048]', 
                       shape='ellipse', fillcolor=gpu_groups['comm'])
                
                # MoE output
                ep.node(f'moe_out_s1_ep{ep_group}', 
                       f'MoE Output (TP8)\\nEP{ep_group}\\nInput: [64, seq_len, 2048]\\nOutput: [64, seq_len, 1024]', 
                       shape='rectangle', fillcolor=gpu_groups['stage1_tp'])
    
    # Pipeline communication
    dot.node('pp_comm', 'Pipeline Transfer\\nStage 1 -> Stage 2\\n1024 GPUs to 1024 GPUs', 
             shape='ellipse', fillcolor=gpu_groups['comm'])
    
    # Stage 2: GPUs 1024-2047 (PP2, EP64, TP8)
    with dot.subgraph(name='cluster_stage2') as c:
        c.attr(label='Stage 2: GPUs 1024-2047\\nPP2, EP64, TP8', style='rounded', fillcolor=gpu_groups['stage2_tp'])
        
        # Similar structure to stage 1
        for ep_group in range(8):
            with c.subgraph(name=f'cluster_ep{ep_group}_stage2') as ep:
                ep.attr(label=f'EP Group {ep_group}\\nTP8', style='dotted')
                
                ep.node(f'token_proc_s2_ep{ep_group}', 
                       f'Token Processing\\nEP{ep_group}\\nInput: [64, seq_len, 1024]\\nOutput: [64, seq_len, 1024]', 
                       shape='rectangle', fillcolor=gpu_groups['stage2_tp'])
                
                ep.node(f'ln_s2_ep{ep_group}', 
                       f'LayerNorm (TP8)\\nEP{ep_group}\\nInput: [64, seq_len, 1024]\\nOutput: [64, seq_len, 1024]', 
                       shape='rectangle', fillcolor=gpu_groups['stage2_tp'])
                
                ep.node(f'qkv_s2_ep{ep_group}', 
                       f'QKV Proj (TP8)\\nEP{ep_group}\\nInput: [64, seq_len, 1024]\\nOutput: [64, seq_len, 16, 64]', 
                       shape='rectangle', fillcolor=gpu_groups['stage2_tp'])
                
                ep.node(f'attn_s2_ep{ep_group}', 
                       f'Self-Attention (TP8)\\nEP{ep_group}\\nInput: [64, seq_len, 16, 64]\\nOutput: [64, seq_len, 16, 64]', 
                       shape='rectangle', fillcolor=gpu_groups['stage2_tp'])
                
                ep.node(f'attn_out_s2_ep{ep_group}', 
                       f'Attn Output (TP8)\\nEP{ep_group}\\nInput: [64, seq_len, 16, 64]\\nOutput: [64, seq_len, 1024]', 
                       shape='rectangle', fillcolor=gpu_groups['stage2_tp'])
                
                ep.node(f'route_s2_ep{ep_group}', 
                       f'MoE Routing\\nEP{ep_group}\\nInput: [64, seq_len, 1024]\\nOutput: [64, seq_len, 2048]', 
                       shape='parallelogram', fillcolor=gpu_groups['routing'])
                
                ep.node(f'dispatch_s2_ep{ep_group}', 
                       f'Expert Dispatch\\nEP{ep_group}\\nAll-to-All\\nInput: [64, seq_len, 2048]\\nOutput: [1, seq_len, 2048]', 
                       shape='ellipse', fillcolor=gpu_groups['comm'])
                
                for expert in range(8):
                    ep.node(f'expert_s2_ep{ep_group}_ex{expert}', 
                           f'Expert {ep_group*8+expert+512}\\nTP8\\nInput: [1, seq_len, 2048]\\nOutput: [1, seq_len, 2048]',                            shape='rectangle', fillcolor=gpu_groups['stage2_tp'])
                
                ep.node(f'combine_s2_ep{ep_group}', 
                       f'Expert Combine\\nEP{ep_group}\\nAll-to-All\\nInput: [1, seq_len, 2048]\\nOutput: [64, seq_len, 2048]', 
                       shape='ellipse', fillcolor=gpu_groups['comm'])
                
                ep.node(f'moe_out_s2_ep{ep_group}', 
                       f'MoE Output (TP8)\\nEP{ep_group}\\nInput: [64, seq_len, 2048]\\nOutput: [64, seq_len, 1024]', 
                       shape='rectangle', fillcolor=gpu_groups['stage2_tp'])
    
    # Output processing
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Processing\\nDP2 Merge', style='rounded', fillcolor=gpu_groups['dp'])
        c.node('dp_merge', 'DP Merge\\n64 seq from each\\nTotal: 128 seq', shape='parallelogram', fillcolor=gpu_groups['dp'])
        c.node('output', 'Output\\n[128, seq_len, vocab_size]', shape='ellipse', fillcolor='white')
    
    # Connections
    dot.edge('input', 'dp_split')
    
    # Connect to all EP groups in stage 1
    for ep_group in range(8):
        dot.edge('dp_split', f'token_proc_s1_ep{ep_group}')
        dot.edge(f'token_proc_s1_ep{ep_group}', f'ln_s1_ep{ep_group}')
        dot.edge(f'ln_s1_ep{ep_group}', f'qkv_s1_ep{ep_group}')
        dot.edge(f'qkv_s1_ep{ep_group}', f'attn_s1_ep{ep_group}')
        dot.edge(f'attn_s1_ep{ep_group}', f'attn_out_s1_ep{ep_group}')
        dot.edge(f'attn_out_s1_ep{ep_group}', f'route_s1_ep{ep_group}')
        dot.edge(f'route_s1_ep{ep_group}', f'dispatch_s1_ep{ep_group}')
        
        # Connect to experts
        for expert in range(8):
            dot.edge(f'dispatch_s1_ep{ep_group}', f'expert_s1_ep{ep_group}_ex{expert}')
            dot.edge(f'expert_s1_ep{ep_group}_ex{expert}', f'combine_s1_ep{ep_group}')
        
        dot.edge(f'combine_s1_ep{ep_group}', f'moe_out_s1_ep{ep_group}')
        
        # Connect to pipeline transfer
        if ep_group == 0:
            dot.edge(f'moe_out_s1_ep{ep_group}', 'pp_comm')
    
    # Pipeline transfer to stage 2
    for ep_group in range(8):
        dot.edge('pp_comm', f'token_proc_s2_ep{ep_group}')
        dot.edge(f'token_proc_s2_ep{ep_group}', f'ln_s2_ep{ep_group}')
        dot.edge(f'ln_s2_ep{ep_group}', f'qkv_s2_ep{ep_group}')
        dot.edge(f'qkv_s2_ep{ep_group}', f'attn_s2_ep{ep_group}')
        dot.edge(f'attn_s2_ep{ep_group}', f'attn_out_s2_ep{ep_group}')
        dot.edge(f'attn_out_s2_ep{ep_group}', f'route_s2_ep{ep_group}')
        dot.edge(f'route_s2_ep{ep_group}', f'dispatch_s2_ep{ep_group}')
        
        # Connect to experts
        for expert in range(8):
            dot.edge(f'dispatch_s2_ep{ep_group}', f'expert_s2_ep{ep_group}_ex{expert}')
            dot.edge(f'expert_s2_ep{ep_group}_ex{expert}', f'combine_s2_ep{ep_group}')
        
        dot.edge(f'combine_s2_ep{ep_group}', f'moe_out_s2_ep{ep_group}')
        
        # Connect to DP merge
        if ep_group == 0:
            dot.edge(f'moe_out_s2_ep{ep_group}', 'dp_merge')
    
    dot.edge('dp_merge', 'output')
    
    return dot

if __name__ == '__main__':
    # Create comprehensive DAG
    comprehensive_dag = create_comprehensive_dag()
    
    # Save comprehensive DAG
    comprehensive_dag.render('../outputs/2025-12-22-14-17-26/llm_comprehensive_dag', format='dot', cleanup=False)
    comprehensive_dag.render('../outputs/2025-12-22-14-17-26/llm_comprehensive_dag', format='svg', cleanup=False)
    
    print("Comprehensive DAG files generated successfully!")
    print("Files saved:")
    print("- ../outputs/2025-12-22-14-17-26/llm_comprehensive_dag.dot")
    print("- ../outputs/2025-12-22-14-17-26/llm_comprehensive_dag.svg")