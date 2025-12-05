#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_moe_dag():
    # Create a new directed graph
    dot = Digraph(comment='30B MoE Model Deployment DAG - EP16-TP8-PP4-DP4')
    dot.attr(rankdir='TB', splines='ortho', bgcolor='white')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    with dot.subgraph(name='cluster_input') as c:
        c.attr(style='rounded', fillcolor='lightgray', label='Input Layer')
        c.node('input', 'Input Embedding\nGPU: [0-511]\nInput: [batch_size=128, seq_len=1024, hidden=1024]\nOutput: [batch_size=128, seq_len=1024, hidden=1024]', 
               shape='rectangle', fillcolor='lightblue')
    
    # Data Parallel split
    dot.node('dp_split', 'DP Split\nGPU: [0-511]\nInput: [batch_size=128, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
             shape='parallelogram', fillcolor='lightyellow')
    dot.edge('input', 'dp_split')
    
    # Pipeline Stage 0 (Layers 0-3)
    with dot.subgraph(name='cluster_pp0') as c:
        c.attr(style='rounded', fillcolor='lightcoral', label='Pipeline Stage 0: Layers 0-3 (GPUs 0-127)')
        
        # First layer - attention
        c.node('pp0_layer0_attn', 'Layer 0: Attention\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='rectangle', fillcolor='lightblue')
        
        # Attention TP communication
        c.node('pp0_layer0_tp_comm', 'TP All-Reduce\nGPU: [0-7], [8-15], [16-23], [24-31]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='ellipse', fillcolor='lightgreen')
        
        # First layer - MoE
        c.node('pp0_layer0_moe', 'Layer 0: MoE Routing\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Expert selection (gate)
        c.node('pp0_layer0_gate', 'Expert Gate Selection\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions', 
               shape='parallelogram', fillcolor='lightyellow', style='dashed')
        
        # Expert computation (4 experts per GPU group)
        for expert_id in range(4):
            c.node(f'pp0_layer0_expert{expert_id}', f'Expert {expert_id}\nGPU: [{expert_id*8}-{expert_id*8+7}]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]', 
                   shape='rectangle', fillcolor='lightblue')
            
            # Expert TP communication
            c.node(f'pp0_layer0_expert{expert_id}_tp', f'TP All-Reduce\nGPU: [{expert_id*8}-{expert_id*8+7}]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]', 
                   shape='ellipse', fillcolor='lightgreen')
        
        # Expert aggregation
        c.node('pp0_layer0_agg', 'Expert Aggregation\nGPU: [0-31]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # EP communication (all-to-all)
        c.node('pp0_layer0_ep_comm', 'EP All-to-All\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='ellipse', fillcolor='lightgreen')
    
    # Connect input to PP0
    dot.edge('dp_split', 'pp0_layer0_attn')
    dot.edge('pp0_layer0_attn', 'pp0_layer0_tp_comm')
    dot.edge('pp0_layer0_tp_comm', 'pp0_layer0_moe')
    dot.edge('pp0_layer0_moe', 'pp0_layer0_gate')
    
    # Connect experts
    for expert_id in range(4):
        dot.edge('pp0_layer0_gate', f'pp0_layer0_expert{expert_id}', style='dashed')
        dot.edge(f'pp0_layer0_expert{expert_id}', f'pp0_layer0_expert{expert_id}_tp')
        dot.edge(f'pp0_layer0_expert{expert_id}_tp', 'pp0_layer0_agg')
    
    dot.edge('pp0_layer0_agg', 'pp0_layer0_ep_comm')
    
    # Pipeline Stage 1 (Layers 4-7)
    with dot.subgraph(name='cluster_pp1') as c:
        c.attr(style='rounded', fillcolor='lightsteelblue', label='Pipeline Stage 1: Layers 4-7 (GPUs 128-255)')
        
        c.node('pp1_layer4_attn', 'Layer 4: Attention\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='rectangle', fillcolor='lightblue')
        
        c.node('pp1_layer4_tp_comm', 'TP All-Reduce\nGPU: [128-135], [136-143], [144-151], [152-159]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='ellipse', fillcolor='lightgreen')
        
        c.node('pp1_layer4_moe', 'Layer 4: MoE Routing\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('pp1_layer4_gate', 'Expert Gate Selection\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions', 
               shape='parallelogram', fillcolor='lightyellow', style='dashed')
        
        for expert_id in range(4):
            c.node(f'pp1_layer4_expert{expert_id}', f'Expert {expert_id}\nGPU: [128+{expert_id*8}-128+{expert_id*8+7}]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]', 
                   shape='rectangle', fillcolor='lightblue')
            
            c.node(f'pp1_layer4_expert{expert_id}_tp', f'TP All-Reduce\nGPU: [128+{expert_id*8}-128+{expert_id*8+7}]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]', 
                   shape='ellipse', fillcolor='lightgreen')
        
        c.node('pp1_layer4_agg', 'Expert Aggregation\nGPU: [128-159]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('pp1_layer4_ep_comm', 'EP All-to-All\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 2 (Layers 8-11)
    with dot.subgraph(name='cluster_pp2') as c:
        c.attr(style='rounded', fillcolor='lightseagreen', label='Pipeline Stage 2: Layers 8-11 (GPUs 256-383)')
        
        c.node('pp2_layer8_attn', 'Layer 8: Attention\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='rectangle', fillcolor='lightblue')
        
        c.node('pp2_layer8_tp_comm', 'TP All-Reduce\nGPU: [256-263], [264-271], [272-279], [280-287]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='ellipse', fillcolor='lightgreen')
        
        c.node('pp2_layer8_moe', 'Layer 8: MoE Routing\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('pp2_layer8_gate', 'Expert Gate Selection\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions', 
               shape='parallelogram', fillcolor='lightyellow', style='dashed')
        
        for expert_id in range(4):
            c.node(f'pp2_layer8_expert{expert_id}', f'Expert {expert_id}\nGPU: [256+{expert_id*8}-256+{expert_id*8+7}]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]', 
                   shape='rectangle', fillcolor='lightblue')
            
            c.node(f'pp2_layer8_expert{expert_id}_tp', f'TP All-Reduce\nGPU: [256+{expert_id*8}-256+{expert_id*8+7}]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]', 
                   shape='ellipse', fillcolor='lightgreen')
        
        c.node('pp2_layer8_agg', 'Expert Aggregation\nGPU: [256-287]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('pp2_layer8_ep_comm', 'EP All-to-All\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 3 (Layers 12-15)
    with dot.subgraph(name='cluster_pp3') as c:
        c.attr(style='rounded', fillcolor='lightsalmon', label='Pipeline Stage 3: Layers 12-15 (GPUs 384-511)')
        
        c.node('pp3_layer12_attn', 'Layer 12: Attention\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='rectangle', fillcolor='lightblue')
        
        c.node('pp3_layer12_tp_comm', 'TP All-Reduce\nGPU: [384-391], [392-399], [400-407], [408-415]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='ellipse', fillcolor='lightgreen')
        
        c.node('pp3_layer12_moe', 'Layer 12: MoE Routing\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('pp3_layer12_gate', 'Expert Gate Selection\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions', 
               shape='parallelogram', fillcolor='lightyellow', style='dashed')
        
        for expert_id in range(4):
            c.node(f'pp3_layer12_expert{expert_id}', f'Expert {expert_id}\nGPU: [384+{expert_id*8}-384+{expert_id*8+7}]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]', 
                   shape='rectangle', fillcolor='lightblue')
            
            c.node(f'pp3_layer12_expert{expert_id}_tp', f'TP All-Reduce\nGPU: [384+{expert_id*8}-384+{expert_id*8+7}]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]', 
                   shape='ellipse', fillcolor='lightgreen')
        
        c.node('pp3_layer12_agg', 'Expert Aggregation\nGPU: [384-415]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('pp3_layer12_ep_comm', 'EP All-to-All\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
               shape='ellipse', fillcolor='lightgreen')
    
    # Output layer
    with dot.subgraph(name='cluster_output') as c:
        c.attr(style='rounded', fillcolor='lightgray', label='Output Layer')
        c.node('output', 'Output Layer\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, vocab_size=32000]', 
               shape='rectangle', fillcolor='lightblue')
        
        c.node('dp_agg', 'DP Aggregation\nGPU: [0-511]\nInput: [batch_size=32, seq_len=1024, vocab_size=32000]\nOutput: [batch_size=128, seq_len=1024, vocab_size=32000]', 
               shape='parallelogram', fillcolor='lightyellow')
    
    # Connect pipeline stages
    dot.edge('pp0_layer0_ep_comm', 'pp1_layer4_attn')
    dot.edge('pp1_layer4_ep_comm', 'pp2_layer8_attn')
    dot.edge('pp2_layer8_ep_comm', 'pp3_layer12_attn')
    dot.edge('pp3_layer12_ep_comm', 'output')
    dot.edge('output', 'dp_agg')
    
    # Add communication between EP groups for all-to-all
    for stage in range(4):
        stage_gpus = [f'pp{stage}_layer{stage*4}_ep_comm']
        for other_stage in range(4):
            if stage != other_stage:
                dot.edge(f'pp{stage}_layer{stage*4}_ep_comm', f'pp{other_stage}_layer{other_stage*4}_moe', 
                        style='dotted', constraint='false')
    
    return dot

def create_simplified_moe_dag():
    # Create a more focused DAG showing one complete layer
    dot = Digraph(comment='30B MoE Model - Single Layer Detail')
    dot.attr(rankdir='TB', splines='ortho', bgcolor='white')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input to layer
    dot.node('layer_input', 'Layer Input\nGPU: All 512 GPUs\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
             shape='rectangle', fillcolor='lightblue')
    
    # Attention computation
    dot.node('attention', 'Multi-Head Attention\nGPU: [0-511] (TP=8 per group)\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
             shape='rectangle', fillcolor='lightblue')
    
    dot.node('attn_tp_comm', 'TP All-Reduce\nGPU: 64 TP groups (8 GPUs each)\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # MoE routing
    dot.node('moe_route', 'MoE Router\nGPU: [0-511] (16 EP groups)\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing weights', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Expert gate selection
    dot.node('gate_select', 'Gate Selection\nGPU: [0-511]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: top-2 expert selection', 
             shape='parallelogram', fillcolor='lightyellow', style='dashed')
    
    # Expert computation (showing 4 experts per EP group)
    for ep_group in range(16):
        for expert_id in range(4):
            gpu_start = ep_group * 32 + expert_id * 8
            gpu_end = gpu_start + 7
            
            dot.node(f'expert_{ep_group}_{expert_id}', 
                     f'Expert {ep_group*4 + expert_id}\nGPU: [{gpu_start}-{gpu_end}]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]', 
                     shape='rectangle', fillcolor='lightblue')
            
            # Expert TP communication
            dot.node(f'expert_{ep_group}_{expert_id}_tp', 
                     f'Expert TP All-Reduce\nGPU: [{gpu_start}-{gpu_end}]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]', 
                     shape='ellipse', fillcolor='lightgreen')
    
    # Expert aggregation
    dot.node('expert_agg', 'Expert Aggregation\nGPU: [0-511] (16 EP groups)\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # EP communication
    dot.node('ep_comm', 'EP All-to-All\nGPU: [0-511] (16 EP groups)\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Layer output
    dot.node('layer_output', 'Layer Output\nGPU: All 512 GPUs\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]', 
             shape='rectangle', fillcolor='lightblue')
    
    # Connect nodes
    dot.edge('layer_input', 'attention')
    dot.edge('attention', 'attn_tp_comm')
    dot.edge('attn_tp_comm', 'moe_route')
    dot.edge('moe_route', 'gate_select')
    
    # Connect experts
    for ep_group in range(16):
        for expert_id in range(4):
            dot.edge('gate_select', f'expert_{ep_group}_{expert_id}', style='dashed')
            dot.edge(f'expert_{ep_group}_{expert_id}', f'expert_{ep_group}_{expert_id}_tp')
            dot.edge(f'expert_{ep_group}_{expert_id}_tp', 'expert_agg')
    
    dot.edge('expert_agg', 'ep_comm')
    dot.edge('ep_comm', 'layer_output')
    
    return dot

if __name__ == "__main__":
    # Generate the detailed DAG
    detailed_dag = create_moe_dag()
    
    # Save as DOT file
    with open('../outputs/2025-12-05-09-02-16/moe_detailed_dag.dot', 'w') as f:
        f.write(detailed_dag.source)
    
    # Render as SVG
    detailed_dag.render('../outputs/2025-12-05-09-02-16/moe_detailed_dag', format='svg', cleanup=True)
    
    # Generate the simplified DAG
    simplified_dag = create_simplified_moe_dag()
    
    # Save as DOT file
    with open('../outputs/2025-12-05-09-02-16/moe_simplified_dag.dot', 'w') as f:
        f.write(simplified_dag.source)
    
    # Render as SVG
    simplified_dag.render('../outputs/2025-12-05-09-02-16/moe_simplified_dag', format='svg', cleanup=True)
    
    print("DAG files generated successfully!")
    print("Files created:")
    print("- ../outputs/2025-12-05-09-02-16/moe_detailed_dag.dot")
    print("- ../outputs/2025-12-05-09-02-16/moe_detailed_dag.svg")
    print("- ../outputs/2025-12-05-09-02-16/moe_simplified_dag.dot")
    print("- ../outputs/2025-12-05-09-02-16/moe_simplified_dag.svg")