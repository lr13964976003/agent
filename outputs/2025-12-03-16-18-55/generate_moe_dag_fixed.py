#!/usr/bin/env python3
"""
Generate complete DAG for EP16 + TP4 + PP2 MoE deployment strategy
Fixed version without cycles
"""

import graphviz
import os

def create_complete_moe_dag_fixed():
    """Create the complete DAG for MoE parallel deployment without cycles"""
    
    dot = graphviz.Digraph(comment='Complete MoE EP16+TP4+PP2 Parallel Strategy - Fixed')
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
    prev_ln = 'input'  # Track the previous layer norm output
    
    for layer in range(16):
        pp_stage = layer // 8  # 0 or 1
        layer_in_stage = layer % 8
        
        # Attention components for this layer
        gpu_base = pp_stage * 64  # PP0: 0-63, PP1: 64-127
        
        # Show TP4 split within each EP group
        for tp in range(4):  # 4 TP groups
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
            gpu_base_exp = pp_stage * 64 + ep * 4
            for expert in range(4):  # 4 experts per GPU
                gpu_id = gpu_base_exp + expert
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
        
        # Connect components within the layer properly
        # Attention path
        for tp in range(4):
            dot.edge(f'layer{layer}_qkv_tp{tp}', f'layer{layer}_qkv_comm')
            dot.edge(f'layer{layer}_qkv_comm', f'layer{layer}_attn_out_tp{tp}')
            dot.edge(f'layer{layer}_attn_out_tp{tp}', f'layer{layer}_attn_out_comm')
        
        # Residual connection 1
        dot.edge(f'layer{layer}_attn_out_comm', f'layer{layer}_res1')
        if layer == 0:
            dot.edge('input', f'layer{layer}_res1')
        else:
            dot.edge(f'layer{layer-1}_ln2', f'layer{layer}_res1')
        
        dot.edge(f'layer{layer}_res1', f'layer{layer}_ln1')
        
        # MoE path
        dot.edge(f'layer{layer}_ln1', f'layer{layer}_gate')
        dot.edge(f'layer{layer}_gate', f'layer{layer}_dispatch')
        
        # Expert computations
        for ep in range(16):
            for expert in range(4):
                dot.edge(f'layer{layer}_dispatch', f'layer{layer}_expert{ep*4+expert}')
                dot.edge(f'layer{layer}_expert{ep*4+expert}', f'layer{layer}_combine')
        
        # Residual connection 2
        dot.edge(f'layer{layer}_combine', f'layer{layer}_res2')
        dot.edge(f'layer{layer}_ln1', f'layer{layer}_res2')  # Residual
        dot.edge(f'layer{layer}_res2', f'layer{layer}_ln2')
        
        prev_ln = f'layer{layer}_ln2'
    
    # Output
    dot.node('output', 'Model Output\\nGPU: Host\\n[batch=128, seq=10240, hidden=1024]', 
             shape='box', fillcolor='white', fontsize='10')
    
    dot.edge(prev_ln, 'output')
    
    return dot

def create_simple_moe_dag():
    """Create a simplified but complete DAG showing key operations"""
    
    dot = graphviz.Digraph(comment='MoE EP16+TP4+PP2 Parallel Strategy - Simplified')
    dot.attr(rankdir='TB', size='100,200')
    dot.attr('node', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Total Input\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgray')
    
    # Show one complete layer with all components
    layer = 0
    
    # Token embedding and initial processing
    dot.node('embed', 'Token Embedding\\nGPU: PP0-TP0-TP3\\nInput: [batch_size=128, seq_len=10240]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    # Multi-Head Attention with TP4
    # TP split: 1024 dimensions → 256 per GPU
    
    # QKV projection with tensor parallelism
    dot.node('qkv_tp', 'QKV Projection TP4\\nGPU: PP0-TP0-TP3\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=256]', 
             shape='box', fillcolor='lightgreen')
    
    dot.node('qkv_comm', 'QKV All-Gather\\nInput: [batch_size=128, seq_len=10240, hidden=256]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Attention computation
    dot.node('attn_score', 'Attention Score\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, seq_len=10240]', 
             shape='box', fillcolor='lightgreen')
    
    dot.node('attn_softmax', 'Attention Softmax\\nInput: [batch_size=128, seq_len=10240, seq_len=10240]\\nOutput: [batch_size=128, seq_len=10240, seq_len=10240]', 
             shape='box', fillcolor='lightgreen')
    
    dot.node('attn_weight', 'Attention Weight\\nInput: [batch_size=128, seq_len=10240, seq_len=10240]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    # Attention output projection
    dot.node('attn_out_tp', 'Attention Output TP4\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=256]', 
             shape='box', fillcolor='lightgreen')
    
    dot.node('attn_out_comm', 'Attention Out All-Reduce\\nInput: [batch_size=128, seq_len=10240, hidden=256]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Residual connection
    dot.node('attn_res', 'Attention Residual Add\\nInput: [batch_size=128, seq_len=10240, hidden=1024], [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Layer Norm
    dot.node('ln1', 'Layer Norm 1\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    # MoE Layer with EP16
    # Gate computation
    dot.node('gate', 'MoE Gate\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, experts=64]', 
             shape='box', fillcolor='lightgreen')
    
    # Expert routing (dashed line for selection)
    dot.node('route', 'Expert Routing\\nInput: [batch_size=128, seq_len=10240, experts=64]\\nOutput: [batch_size=128, seq_len=10240, top_k=2]', 
             shape='parallelogram', fillcolor='lightyellow')
    dot.edge('gate', 'route', style='dashed')
    
    # All-to-all communication for expert dispatch
    dot.node('dispatch', 'Expert Dispatch All-to-All\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Expert computation (4 experts per GPU in EP16)
    for expert_id in range(4):
        dot.node(f'expert_{expert_id}', f'Expert {expert_id}\\nGPU: {expert_id}\\nInput: [batch_size=8, seq_len=640, hidden=1024]\\nOutput: [batch_size=8, seq_len=640, hidden=1024]', 
                 shape='box', fillcolor='lightgreen')
    
    # Expert aggregation
    dot.node('expert_agg', 'Expert Aggregation\\nInput: [batch_size=8, seq_len=640, hidden=1024] × 4\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # All-to-all communication for expert combine
    dot.node('combine', 'Expert Combine All-to-All\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # MoE output projection
    dot.node('moe_out', 'MoE Output Projection\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    # Residual connection for MoE
    dot.node('moe_res', 'MoE Residual Add\\nInput: [batch_size=128, seq_len=10240, hidden=1024], [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Layer Norm 2
    dot.node('ln2', 'Layer Norm 2\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgreen')
    
    # Pipeline stage transitions
    dot.node('pp_transition', 'Pipeline Stage Transition\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Final output
    dot.node('output', 'Total Output\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='box', fillcolor='lightgray')
    
    # Connect components
    connections = [
        ('input', 'embed'),
        ('embed', 'qkv_tp'),
        ('qkv_tp', 'qkv_comm'),
        ('qkv_comm', 'attn_score'),
        ('attn_score', 'attn_softmax'),
        ('attn_softmax', 'attn_weight'),
        ('attn_weight', 'attn_out_tp'),
        ('attn_out_tp', 'attn_out_comm'),
        ('attn_out_comm', 'attn_res'),
        ('embed', 'attn_resres'),  # Residual connection
        ('attn_res', 'ln1'),
        ('ln1', 'gate'),
        ('gate', 'route'),
        ('route', 'dispatch'),
        ('ln1', 'dispatch'),  # Data flow
        ('dispatch', 'expert_0'),
        ('dispatch', 'expert_1'),
        ('dispatch', 'expert_2'),
        ('dispatch', 'expert_3'),
        ('expert_0', 'expert_agg'),
        ('expert_1', 'expert_agg'),
        ('expert_2', 'expert_agg'),
        ('expert_3', 'expert_agg'),
        ('expert_agg', 'combine'),
        ('combine', 'moe_out'),
        ('moe_out', 'moe_res'),
        ('ln1', 'moe_res'),  # Residual connection
        ('moe_res', 'ln2'),
        ('ln2', 'pp_transition'),
        ('pp_transition', 'output')
    ]
    
    for src, dst in connections:
        dot.edge(src, dst)
    
    return dot

if __name__ == '__main__':
    # Create output directory
    os.makedirs('../outputs/2025-12-03-16-18-55', exist_ok=True)
    
    # Generate the complete DAG
    print("Generating fixed MoE parallel strategy DAG...")
    dag = create_complete_moe_dag_fixed()
    
    # Save as DOT file
    dot_file = '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_fixed.dot'
    with open(dot_file, 'w') as f:
        f.write(dag.source)
    print(f"Saved fixed DOT file: {dot_file}")
    
    # Save as SVG
    svg_file = '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_fixed.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    print(f"Saved fixed SVG file: {svg_file}")
    
    # Also generate simplified version
    simple_dag = create_simple_moe_dag()
    simple_dot_file = '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_simple_fixed.dot'
    with open(simple_dot_file, 'w') as f:
        f.write(simple_dag.source)
    print(f"Saved simple fixed DOT file: {simple_dot_file}")
    
    simple_svg_file = '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_simple_fixed.svg'
    simple_dag.render(simple_svg_file.replace('.svg', ''), format='svg', cleanup=True)
    print(f"Saved simple fixed SVG file: {simple_svg_file}")
    
    print("Fixed DAG generation complete!")
    print(f"Files saved to: ../outputs/2025-12-03-16-18-55/")
    
    # List all generated files
    files = [
        '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_fixed.dot',
        '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_fixed.svg',
        '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_simple_fixed.dot', 
        '../outputs/2025-12-03-16-18-55/moe_parallel_strategy_simple_fixed.svg'
    ]
    
    print("\nGenerated fixed files:")
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (not found)")