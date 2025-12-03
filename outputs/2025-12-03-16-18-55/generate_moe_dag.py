#!/usr/bin/env python3
"""
Generate complete DAG for EP16 + TP4 + PP2 MoE deployment strategy
This script creates a detailed DAG showing all operations, communications, and data flows
"""

import graphviz
import os

def create_moe_parallel_dag():
    """Create the complete DAG for MoE parallel deployment"""
    
    # Create directed graph
    dot = graphviz.Digraph(comment='MoE EP16+TP4+PP2 Parallel Strategy DAG')
    dot.attr(rankdir='TB', size='100,200')
    dot.attr('node', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Define GPU groups based on EP16+TP4+PP2 strategy
    # Total: 128 GPUs = 16 EP groups × 4 TP groups × 2 PP stages
    
    # Let's create a simplified but complete representation
    # We'll show one representative from each parallel dimension
    
    # Input node
    dot.node('input', 'Total Input\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgray')
    
    # PP Stage 0 (Layers 0-7) - focusing on one representative layer
    # We'll show detailed breakdown for Layer 0
    
    # Token embedding and initial processing
    dot.node('embed_0', 'Token Embedding PP0\\nInput: [batch_size=128, seq_len=10240]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    # Layer 0 - Multi-Head Attention with TP4
    # TP split: 1024 dimensions → 256 per GPU
    
    # QKV projection with tensor parallelism
    dot.node('qkv_tp_0', 'QKV Projection TP\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=256]', 
             shape='box', fillcolor='lightgreen')
    
    dot.node('qkv_comm_0', 'QKV All-Gather\\nInput: [batch_size=128, seq_len=10240, hidden=256]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Attention computation
    dot.node('attn_scale_0', 'Attention Scale\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    dot.node('attn_score_0', 'Attention Score\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, seq_len=10240]', 
             shape='box', fillcolor='lightgreen')
    
    dot.node('attn_softmax_0', 'Attention Softmax\\nInput: [batch_size=128, seq_len=10240, seq_len=10240]\\nOutput: [batch_size=128, seq_len=10240, seq_len=10240]', 
             shape='box', fillcolor='lightgreen')
    
    dot.node('attn_weight_0', 'Attention Weight\\nInput: [batch_size=128, seq_len=10240, seq_len=10240]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    # Attention output projection
    dot.node('attn_out_tp_0', 'Attention Output TP\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=256]', 
             shape='box', fillcolor='lightgreen')
    
    dot.node('attn_out_comm_0', 'Attention Out All-Reduce\\nInput: [batch_size=128, seq_len=10240, hidden=256]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Residual connection
    dot.node('attn_res_0', 'Attention Residual Add\\nInput: [batch_size=128, seq_len=10240, hidden=1024], [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Layer Norm
    dot.node('ln1_0', 'Layer Norm 1\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    # MoE Layer with EP16
    # Gate computation
    dot.node('gate_0', 'MoE Gate\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, experts=64]', 
             shape='box', fillcolor='lightgreen')
    
    # Expert routing (dashed line for selection)
    dot.node('route_0', 'Expert Routing\\nInput: [batch_size=128, seq_len=10240, experts=64]\\nOutput: [batch_size=128, seq_len=10240, top_k=2]', 
             shape='parallelogram', fillcolor='lightyellow')
    dot.edge('gate_0', 'route_0', style='dashed')
    
    # All-to-all communication for expert dispatch
    dot.node('dispatch_0', 'Expert Dispatch All-to-All\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Expert computation (4 experts per GPU in EP16)
    for expert_id in range(4):
        dot.node(f'expert_{expert_id}_0', f'Expert {expert_id}\\nInput: [batch_size=8, seq_len=640, hidden=1024]\\nOutput: [batch_size=8, seq_len=640, hidden=1024]', 
                 shape='box', fillcolor='lightgreen')
        dot.node(f'expert_{expert_id}_ffn1_0', f'Expert {expert_id} FFN1\\nInput: [batch_size=8, seq_len=640, hidden=1024]\\nOutput: [batch_size=8, seq_len=640, hidden=512]', 
                 shape='box', fillcolor='lightgreen')
        dot.node(f'expert_{expert_id}_gelu_0', f'Expert {expert_id} GELU\\nInput: [batch_size=8, seq_len=640, hidden=512]\\nOutput: [batch_size=8, seq_len=640, hidden=512]', 
                 shape='box', fillcolor='lightgreen')
        dot.node(f'expert_{expert_id}_ffn2_0', f'Expert {expert_id} FFN2\\nInput: [batch_size=8, seq_len=640, hidden=512]\\nOutput: [batch_size=8, seq_len=640, hidden=1024]', 
                 shape='box', fillcolor='lightgreen')
        
        # Connect expert components
        dot.edge(f'expert_{expert_id}_0', f'expert_{expert_id}_ffn1_0')
        dot.edge(f'expert_{expert_id}_ffn1_0', f'expert_{expert_id}_gelu_0')
        dot.edge(f'expert_{expert_id}_gelu_0', f'expert_{expert_id}_ffn2_0')
    
    # Expert aggregation
    dot.node('expert_agg_0', 'Expert Aggregation\\nInput: [batch_size=8, seq_len=640, hidden=1024] × 4\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # All-to-all communication for expert combine
    dot.node('combine_0', 'Expert Combine All-to-All\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # MoE output projection
    dot.node('moe_out_0', 'MoE Output Projection\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    # Residual connection for MoE
    dot.node('moe_res_0', 'MoE Residual Add\\nInput: [batch_size=128, seq_len=10240, hidden=1024], [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Layer Norm 2
    dot.node('ln2_0', 'Layer Norm 2\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    # Connect Layer 0 components
    connections = [
        ('input', 'embed_0'),
        ('embed_0', 'qkv_tp_0'),
        ('qkv_tp_0', 'qkv_comm_0'),
        ('qkv_comm_0', 'attn_scale_0'),
        ('attn_scale_0', 'attn_score_0'),
        ('attn_score_0', 'attn_softmax_0'),
        ('attn_softmax_0', 'attn_weight_0'),
        ('attn_weight_0', 'attn_out_tp_0'),
        ('attn_out_tp_0', 'attn_out_comm_0'),
        ('attn_out_comm_0', 'attn_res_0'),
        ('embed_0', 'attn_res_0'),  # Residual connection
        ('attn_res_0', 'ln1_0'),
        ('ln1_0', 'gate_0'),
        ('gate_0', 'route_0'),
        ('route_0', 'dispatch_0'),
        ('ln1_0', 'dispatch_0'),  # Data flow
        ('dispatch_0', 'expert_0_0'),
        ('dispatch_0', 'expert_1_0'),
        ('dispatch_0', 'expert_2_0'),
        ('dispatch_0', 'expert_3_0'),
        ('expert_0_ffn2_0', 'expert_agg_0'),
        ('expert_1_ffn2_0', 'expert_agg_0'),
        ('expert_2_ffn2_0', 'expert_agg_0'),
        ('expert_3_ffn2_0', 'expert_agg_0'),
        ('expert_agg_0', 'combine_0'),
        ('combine_0', 'moe_out_0'),
        ('moe_out_0', 'moe_res_0'),
        ('ln1_0', 'moe_res_0'),  # Residual connection
        ('moe_res_0', 'ln2_0')
    ]
    
    for src, dst in connections:
        dot.edge(src, dst)
    
    # PP Stage 1 (Layers 8-15) - similar structure but different GPU allocation
    # For brevity, showing the transition between stages
    
    dot.node('pp_transition', 'Pipeline Stage Transition\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Final output
    dot.node('output', 'Total Output\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgray')
    
    # Connect final components
    dot.edge('ln2_0', 'pp_transition')
    dot.edge('pp_transition', 'output')
    
    return dot

def create_complete_moe_dag():
    """Create a more complete DAG showing all 16 layers and GPU allocations"""
    
    dot = graphviz.Digraph(comment='Complete MoE EP16+TP4+PP2 Parallel Strategy')
    dot.attr(rankdir='TB', size='200,400')
    dot.attr('node', fontsize='8')
    
    # Define colors for different GPU groups
    colors = {
        'ep0_tp0_pp0': 'lightblue',
        'ep0_tp1_pp0': 'lightgreen', 
        'ep0_tp2_pp0': 'lightyellow',
        'ep0_tp3_pp0': 'lightpink',
        'ep0_tp0_pp1': 'lightcoral',
        'ep0_tp1_pp1': 'lightcyan',
        'ep0_tp2_pp1': 'lightgray',
        'ep0_tp3_pp1': 'lightsteelblue'
    }
    
    # Input
    dot.node('input', 'Model Input\\nGPU: Host\\n[batch=128, seq=10240, hidden=1024]', 
             shape='box', fillcolor='white', fontsize='10')
    
    # Create nodes for each layer with GPU annotations
    for layer in range(16):
        pp_stage = layer // 8  # 0 or 1
        layer_in_stage = layer % 8
        
        # Attention components for this layer
        gpu_base = pp_stage * 64  # PP0: 0-63, PP1: 64-127
        
        # Show TP4 split within each EP group
        for tp in range(4):
            gpu_id = gpu_base + tp * 16  # Simplified representation
            color = colors[f'ep0_tp{tp}_pp{pp_stage}']
            
            # Attention QKV split
            dot.node(f'layer{layer}_qkv_tp{tp}', 
                     f'Layer{layer} QKV TP{tp}\\nGPU:{gpu_id}-{gpu_id+15}\\n[batch=128, seq=10240, hidden=256]', 
                     shape='box', fillcolor=color, fontsize='8')
            
            # Attention output  
            dot.node(f'layer{layer}_attn_out_tp{tp}',
                     f'Layer{layer} AttnOut TP{tp}\\nGPU:{gpu_id}-{gpu_id+15}\\n[batch=128, seq=10240, hidden=256]', 
                     shape='box', fillcolor=color, fontsize='8')
            
            # MLP first layer (column parallel)
            dot.node(f'layer{layer}_mlp1_tp{tp}',
                     f'Layer{layer} MLP1 TP{tp}\\nGPU:{gpu_id}-{gpu_id+15}\\n[batch=128, seq=10240, hidden=512]', 
                     shape='box', fillcolor=color, fontsize='8')
            
            # MLP second layer (row parallel) 
            dot.node(f'layer{layer}_mlp2_tp{tp}',
                     f'Layer{layer} MLP2 TP{tp}\\nGPU:{gpu_id}-{gpu_id+15}\\n[batch=128, seq=10240, hidden=256]', 
                     shape='box', fillcolor=color, fontsize='8')
        
        # Communication nodes for this layer
        dot.node(f'layer{layer}_qkv_comm', 
                 f'Layer{layer} QKV AllGather\\n[batch=128, seq=10240, hidden=1024]', 
                 shape='ellipse', fillcolor='lightblue', fontsize='8')
        
        dot.node(f'layer{layer}_attn_out_comm',
                 f'Layer{layer} AttnOut AllReduce\\n[batch=128, seq=10240, hidden=1024]', 
                 shape='ellipse', fillcolor='lightblue', fontsize='8')
        
        dot.node(f'layer{layer}_mlp_comm',
                 f'Layer{layer} MLP AllReduce\\n[batch=128, seq=10240, hidden=1024]', 
                 shape='ellipse', fillcolor='lightblue', fontsize='8')
        
        # MoE components
        dot.node(f'layer{layer}_gate',
                 f'Layer{layer} MoE Gate\\nGPU:EP0-15\\n[batch=128, seq=10240, experts=64]', 
                 shape='box', fillcolor='lightgreen', fontsize='8')
        
        dot.node(f'layer{layer}_dispatch',
                 f'Layer{layer} Expert Dispatch\\nAll-to-All Comm', 
                 shape='ellipse', fillcolor='lightblue', fontsize='8')
        
        # Expert computation (4 experts per GPU in EP16)
        for ep in range(16):  # 16 EP groups
            gpu_base = pp_stage * 64 + ep * 4
            for expert in range(4):  # 4 experts per GPU
                gpu_id = gpu_base + expert
                dot.node(f'layer{layer}_expert{ep*4+expert}',
                         f'Layer{layer} Expert{ep*4+expert}\\nGPU:{gpu_id}\\n[batch=8, seq=640, hidden=1024]', 
                         shape='box', fillcolor='lightyellow', fontsize='7')
        
        dot.node(f'layer{layer}_combine',
                 f'Layer{layer} Expert Combine\\nAll-to-All Comm', 
                 shape='ellipse', fillcolor='lightblue', fontsize='8')
        
        # Residual and norm
        dot.node(f'layer{layer}_res1',
                 f'Layer{layer} Residual1\\n[batch=128, seq=10240, hidden=1024]', 
                 shape='parallelogram', fillcolor='lightcyan', fontsize='8')
        
        dot.node(f'layer{layer}_ln1',
                 f'Layer{layer} LayerNorm1\\n[batch=128, seq=10240, hidden=1024]', 
                 shape='box', fillcolor='lightgreen', fontsize='8')
        
        dot.node(f'layer{layer}_res2',
                 f'Layer{layer} Residual2\\n[batch=128, seq=10240, hidden=1024]', 
                 shape='parallelogram', fillcolor='lightcyan', fontsize='8')
        
        dot.node(f'layer{layer}_ln2',
                 f'Layer{layer} LayerNorm2\\n[batch=128, seq=10240, hidden=1024]', 
                 shape='box', fillcolor='lightgreen', fontsize='8')
    
    # Pipeline stage transitions
    dot.node('pp_stage0_to_1', 'Pipeline Stage0→Stage1\\nPP Comm', 
             shape='ellipse', fillcolor='red', fontsize='10')
    
    # Output
    dot.node('output', 'Model Output\\nGPU: Host\\n[batch=128, seq=10240, hidden=1024]', 
             shape='box', fillcolor='white', fontsize='10')
    
    # Connect all layers
    dot.edge('input', 'layer0_qkv_tp0')
    
    for layer in range(16):
        # Attention flow
        for tp in range(4):
            dot.edge(f'layer{layer}_qkv_tp{tp}', f'layer{layer}_qkv_comm')
        
        # Simplified connections - in reality all TP splits work together
        dot.edge(f'layer{layer}_qkv_comm', f'layer{layer}_attn_out_tp0')
        
        for tp in range(4):
            dot.edge(f'layer{layer}_attn_out_tp{tp}', f'layer{layer}_attn_out_comm')
        
        dot.edge(f'layer{layer}_attn_out_comm', f'layer{layer}_res1')
        dot.edge(f'layer{layer}_ln1' if layer > 0 else 'input', f'layer{layer}_res1')
        dot.edge(f'layer{layer}_res1', f'layer{layer}_ln1')
        
        # MoE flow
        dot.edge(f'layer{layer}_ln1', f'layer{layer}_gate')
        dot.edge(f'layer{layer}_gate', f'layer{layer}_dispatch')
        
        # Expert computations
        for ep in range(16):
            for expert in range(4):
                dot.edge(f'layer{layer}_dispatch', f'layer{layer}_expert{ep*4+expert}')
                dot.edge(f'layer{layer}_expert{ep*4+expert}', f'layer{layer}_combine')
        
        dot.edge(f'layer{layer}_combine', f'layer{layer}_res2')
        dot.edge(f'layer{layer}_ln1', f'layer{layer}_res2')  # Residual
        dot.edge(f'layer{layer}_res2', f'layer{layer}_ln2')
        
        # Connect to next layer
        if layer == 7:  # Transition between pipeline stages
            dot.edge(f'layer{layer}_ln2', 'pp_stage0_to_1')
            dot.edge('pp_stage0_to_1', f'layer{layer+1}_qkv_tp0')
        elif layer < 15:
            dot.edge(f'layer{layer}_ln2', f'layer{layer+1}_res1')
    
    dot.edge('layer15_ln2', 'output')
    
    return dot

if __name__ == '__main__':
    # Create output directory
    os.makedirs('../outputs/2025-12-03-16-18-55', exist_ok=True)
    
    # Generate the detailed DAG
    print("Generating MoE parallel strategy DAG...")
    dag = create_complete_moe_dag()
    
    # Save as DOT file
    dot_file = '../outputs/2025-12-03-16-18-55/moe_parallel_strategy.dot'
    with open(dot_file, 'w') as f:
        f.write(dag.source)
    print(f"Saved DOT file: {dot_file}")
    
    # Save as SVG
    svg_file = '../outputs/2025-12-03-16-18-55/moe_parallel_strategy.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    print(f"Saved SVG file: {svg_file}")
    
    # Also generate a simpler version for readability
    simple_dag = create_moe_parallel_dag()
    simple_dot_file = '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_simple.dot'
    with open(simple_dot_file, 'w') as f:
        f.write(simple_dag.source)
    print(f"Saved simple DOT file: {simple_dot_file}")
    
    simple_svg_file = '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_simple.svg'
    simple_dag.render(simple_svg_file.replace('.svg', ''), format='svg', cleanup=True)
    print(f"Saved simple SVG file: {simple_svg_file}")
    
    print("DAG generation complete!")
    print(f"Files saved to: ../outputs/2025-12-03-16-18-55/")
    
    # List all generated files
    files = [
        '../outputs/2025-12-03-16-18-55/moe_parallel_strategy.dot',
        '../outputs/2025-12-03-16-18-55/moe_parallel_strategy.svg',
        '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_simple.dot', 
        '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_simple.svg'
    ]
    
    print("\nGenerated files:")
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (not found)")