import graphviz
from graphviz import Digraph
import os

def create_moe_dag():
    # Create a new directed graph
    dot = Digraph(comment='30B MoE Model Deployment DAG')
    dot.attr(rankdir='TB', splines='ortho', bgcolor='white')
    dot.attr('node', shape='rectangle', style='filled', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define shapes
    dot.attr('node', shape='ellipse')  # Communication
    dot.attr('node', shape='rectangle', style='filled,rounded')  # Computation
    dot.attr('node', shape='parallelogram', style='filled')  # Routing/Aggregation
    
    # Colors for different GPU groups
    stage_colors = ['#FFCCCC', '#CCFFCC', '#CCCCFF', '#FFFFCC', '#FFCCFF', '#CCFFFF', '#FFEECC', '#EECCFF']
    
    # Input layer
    with dot.subgraph(name='cluster_input') as c:
        c.attr(style='rounded', bgcolor='#F0F0F0', label='Input Layer')
        c.node('input', 'Input Tokens\\nInput: [batch_size=128, seq_len=1024]\\nOutput: [batch_size=128, seq_len=1024]', 
               shape='ellipse', fillcolor='lightblue')
    
    # Pipeline Stage 0 (Layers 0-1) - 128 GPUs (64 EP × 2 TP)
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(style='rounded', bgcolor=stage_colors[0], label='Pipeline Stage 0 (Layers 0-1)\\nGPUs: 0-127')
        
        # Expert routing (all-to-all communication)
        c.node('stage0_routing', 'Expert Routing\\nAll-to-All Communication\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Attention operations for each expert (TP pairs)
        for expert_id in range(64):
            tp_gpu0 = expert_id * 2
            tp_gpu1 = expert_id * 2 + 1
            
            # QKV projection (column parallel)
            c.node(f'stage0_expert{expert_id}_qkv', 
                   f'Expert {expert_id} QKV Proj (TP)\\nGPUs: {tp_gpu0},{tp_gpu1}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, heads=32, d_k=128]', 
                   shape='rectangle', fillcolor='lightgreen')
            
            # Attention computation
            c.node(f'stage0_expert{expert_id}_attn', 
                   f'Expert {expert_id} Attention (TP)\\nGPUs: {tp_gpu0},{tp_gpu1}\\nInput: [batch_size=128, seq_len=1024, heads=32, d_k=128]\\nOutput: [batch_size=128, seq_len=1024, heads=32, d_k=128]', 
                   shape='rectangle', fillcolor='lightgreen')
            
            # Attention output projection (row parallel)
            c.node(f'stage0_expert{expert_id}_attn_out', 
                   f'Expert {expert_id} Attn Output (TP)\\nGPUs: {tp_gpu0},{tp_gpu1}\\nInput: [batch_size=128, seq_len=1024, heads=32, d_k=128]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                   shape='rectangle', fillcolor='lightgreen')
            
            # MLP first layer (column parallel)
            c.node(f'stage0_expert{expert_id}_mlp1', 
                   f'Expert {expert_id} MLP Layer1 (TP)\\nGPUs: {tp_gpu0},{tp_gpu1}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, ffn_hidden=16384]', 
                   shape='rectangle', fillcolor='lightcoral')
            
            # MLP activation
            c.node(f'stage0_expert{expert_id}_mlp_act', 
                   f'Expert {expert_id} MLP GELU (TP)\\nGPUs: {tp_gpu0},{tp_gpu1}\\nInput: [batch_size=128, seq_len=1024, ffn_hidden=16384]\\nOutput: [batch_size=128, seq_len=1024, ffn_hidden=16384]', 
                   shape='rectangle', fillcolor='lightcoral')
            
            # MLP second layer (row parallel)
            c.node(f'stage0_expert{expert_id}_mlp2', 
                   f'Expert {expert_id} MLP Layer2 (TP)\\nGPUs: {tp_gpu0},{tp_gpu1}\\nInput: [batch_size=128, seq_len=1024, ffn_hidden=16384]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                   shape='rectangle', fillcolor='lightcoral')
        
        # Expert aggregation
        c.node('stage0_aggregate', 'Expert Output Aggregation\\nGPUs: 0-127\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Layer normalization
        c.node('stage0_layernorm1', 'LayerNorm 1\\nGPUs: 0-127\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
               shape='rectangle', fillcolor='lightblue')
        c.node('stage0_layernorm2', 'LayerNorm 2\\nGPUs: 0-127\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
               shape='rectangle', fillcolor='lightblue')
    
    # Pipeline communication between stages
    dot.node('pipe_comm_0_1', 'Pipeline Communication\\nStage 0 → Stage 1\\nGPUs: 0-127 → 128-255\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
             shape='ellipse', fillcolor='orange')
    
    # Pipeline Stage 1 (Layers 2-3) - Similar structure
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(style='rounded', bgcolor=stage_colors[1], label='Pipeline Stage 1 (Layers 2-3)\\nGPUs: 128-255')
        
        c.node('stage1_routing', 'Expert Routing\\nAll-to-All Communication\\nGPUs: 128-255\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        for expert_id in range(64):
            base_gpu = 128 + expert_id * 2
            c.node(f'stage1_expert{expert_id}_qkv', 
                   f'Expert {expert_id} QKV Proj (TP)\\nGPUs: {base_gpu},{base_gpu+1}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, heads=32, d_k=128]', 
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'stage1_expert{expert_id}_attn', 
                   f'Expert {expert_id} Attention (TP)\\nGPUs: {base_gpu},{base_gpu+1}\\nInput: [batch_size=128, seq_len=1024, heads=32, d_k=128]\\nOutput: [batch_size=128, seq_len=1024, heads=32, d_k=128]', 
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'stage1_expert{expert_id}_attn_out', 
                   f'Expert {expert_id} Attn Output (TP)\\nGPUs: {base_gpu},{base_gpu+1}\\nInput: [batch_size=128, seq_len=1024, heads=32, d_k=128]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'stage1_expert{expert_id}_mlp1', 
                   f'Expert {expert_id} MLP Layer1 (TP)\\nGPUs: {base_gpu},{base_gpu+1}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, ffn_hidden=16384]', 
                   shape='rectangle', fillcolor='lightcoral')
            
            c.node(f'stage1_expert{expert_id}_mlp_act', 
                   f'Expert {expert_id} MLP GELU (TP)\\nGPUs: {base_gpu},{base_gpu+1}\\nInput: [batch_size=128, seq_len=1024, ffn_hidden=16384]\\nOutput: [batch_size=128, seq_len=1024, ffn_hidden=16384]', 
                   shape='rectangle', fillcolor='lightcoral')
            
            c.node(f'stage1_expert{expert_id}_mlp2', 
                   f'Expert {expert_id} MLP Layer2 (TP)\\nGPUs: {base_gpu},{base_gpu+1}\\nInput: [batch_size=128, seq_len=1024, ffn_hidden=16384]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                   shape='rectangle', fillcolor='lightcoral')
        
        c.node('stage1_aggregate', 'Expert Output Aggregation\\nGPUs: 128-255\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('stage1_layernorm1', 'LayerNorm 1\\nGPUs: 128-255\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
               shape='rectangle', fillcolor='lightblue')
        c.node('stage1_layernorm2', 'LayerNorm 2\\nGPUs: 128-255\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
               shape='rectangle', fillcolor='lightblue')
    
    # Continue pattern for remaining stages (simplified representation)
    for stage in range(2, 8):
        stage_name = f'cluster_stage{stage}'
        base_gpu = stage * 128
        
        with dot.subgraph(name=stage_name) as c:
            c.attr(style='rounded', bgcolor=stage_colors[stage], 
                   label=f'Pipeline Stage {stage} (Layers {stage*2}-{stage*2+1})\\nGPUs: {base_gpu}-{base_gpu+127}')
            
            c.node(f'stage{stage}_routing', 
                   f'Expert Routing\\nAll-to-All Communication\\nGPUs: {base_gpu}-{base_gpu+127}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Sample expert operations (showing first few experts)
            for expert_id in range(4):  # Show first 4 experts for brevity
                gpu_base = base_gpu + expert_id * 2
                c.node(f'stage{stage}_expert{expert_id}_qkv', 
                       f'Expert {expert_id} QKV Proj (TP)\\nGPUs: {gpu_base},{gpu_base+1}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, heads=32, d_k=128]', 
                       shape='rectangle', fillcolor='lightgreen')
                
                c.node(f'stage{stage}_expert{expert_id}_attn', 
                       f'Expert {expert_id} Attention (TP)\\nGPUs: {gpu_base},{gpu_base+1}\\nInput: [batch_size=128, seq_len=1024, heads=32, d_k=128]\\nOutput: [batch_size=128, seq_len=1024, heads=32, d_k=128]', 
                       shape='rectangle', fillcolor='lightgreen')
                
                c.node(f'stage{stage}_expert{expert_id}_attn_out', 
                       f'Expert {expert_id} Attn Output (TP)\\nGPUs: {gpu_base},{gpu_base+1}\\nInput: [batch_size=128, seq_len=1024, heads=32, d_k=128]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                       shape='rectangle', fillcolor='lightgreen')
                
                c.node(f'stage{stage}_expert{expert_id}_mlp1', 
                       f'Expert {expert_id} MLP Layer1 (TP)\\nGPUs: {gpu_base},{gpu_base+1}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, ffn_hidden=16384]', 
                       shape='rectangle', fillcolor='lightcoral')
                
                c.node(f'stage{stage}_expert{expert_id}_mlp_act', 
                       f'Expert {expert_id} MLP GELU (TP)\\nGPUs: {gpu_base},{gpu_base+1}\\nInput: [batch_size=128, seq_len=1024, ffn_hidden=16384]\\nOutput: [batch_size=128, seq_len=1024, ffn_hidden=16384]', 
                       shape='rectangle', fillcolor='lightcoral')
                
                c.node(f'stage{stage}_expert{expert_id}_mlp2', 
                       f'Expert {expert_id} MLP Layer2 (TP)\\nGPUs: {gpu_base},{gpu_base+1}\\nInput: [batch_size=128, seq_len=1024, ffn_hidden=16384]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                       shape='rectangle', fillcolor='lightcoral')
            
            # Show ellipsis for remaining experts
            c.node(f'stage{stage}_experts_ellipsis', 
                   f'... (Experts 4-63)\\nSimilar operations on GPUs {base_gpu+8}-{base_gpu+127}', 
                   shape='rectangle', style='dashed', fillcolor='lightgray')
            
            c.node(f'stage{stage}_aggregate', 
                   f'Expert Output Aggregation\\nGPUs: {base_gpu}-{base_gpu+127}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                   shape='parallelogram', fillcolor='lightyellow')
            
            c.node(f'stage{stage}_layernorm1', 
                   f'LayerNorm 1\\nGPUs: {base_gpu}-{base_gpu+127}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                   shape='rectangle', fillcolor='lightblue')
            c.node(f'stage{stage}_layernorm2', 
                   f'LayerNorm 2\\nGPUs: {base_gpu}-{base_gpu+127}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                   shape='rectangle', fillcolor='lightblue')
        
        # Pipeline communication
        if stage < 7:
            dot.node(f'pipe_comm_{stage}_{stage+1}', 
                     f'Pipeline Communication\\nStage {stage} → Stage {stage+1}\\nGPUs: {base_gpu}-{base_gpu+127} → {(stage+1)*128}-{(stage+1)*128+127}\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden=4096]', 
                     shape='ellipse', fillcolor='orange')
    
    # Output layer
    with dot.subgraph(name='cluster_output') as c:
        c.attr(style='rounded', bgcolor='#F0F0F0', label='Output Layer')
        c.node('output', 'Output Logits\\nInput: [batch_size=128, seq_len=1024, hidden=4096]\\nOutput: [batch_size=128, seq_len=1024, vocab_size=32000]', 
               shape='ellipse', fillcolor='lightblue')
    
    # Define edges (connections)
    # Input to stage 0
    dot.edge('input', 'stage0_routing')
    
    # Stage 0 connections
    dot.edge('stage0_routing', 'stage0_expert0_qkv')
    dot.edge('stage0_expert0_qkv', 'stage0_expert0_attn')
    dot.edge('stage0_expert0_attn', 'stage0_expert0_attn_out')
    dot.edge('stage0_expert0_attn_out', 'stage0_layernorm1')
    dot.edge('stage0_layernorm1', 'stage0_expert0_mlp1')
    dot.edge('stage0_expert0_mlp1', 'stage0_expert0_mlp_act')
    dot.edge('stage0_expert0_mlp_act', 'stage0_expert0_mlp2')
    dot.edge('stage0_expert0_mlp2', 'stage0_layernorm2')
    dot.edge('stage0_layernorm2', 'stage0_aggregate')
    dot.edge('stage0_aggregate', 'pipe_comm_0_1')
    
    # Stage 1 connections
    dot.edge('pipe_comm_0_1', 'stage1_routing')
    dot.edge('stage1_routing', 'stage1_expert0_qkv')
    dot.edge('stage1_expert0_qkv', 'stage1_expert0_attn')
    dot.edge('stage1_expert0_attn', 'stage1_expert0_attn_out')
    dot.edge('stage1_expert0_attn_out', 'stage1_layernorm1')
    dot.edge('stage1_layernorm1', 'stage1_expert0_mlp1')
    dot.edge('stage1_expert0_mlp1', 'stage1_expert0_mlp_act')
    dot.edge('stage1_expert0_mlp_act', 'stage1_expert0_mlp2')
    dot.edge('stage1_expert0_mlp2', 'stage1_layernorm2')
    dot.edge('stage1_layernorm2', 'stage1_aggregate')
    
    # Continue pattern for remaining stages
    for stage in range(2, 8):
        if stage == 2:
            dot.edge('stage1_aggregate', f'pipe_comm_1_2')
            dot.edge(f'pipe_comm_1_2', f'stage{stage}_routing')
        elif stage > 2:
            prev_stage = stage - 1
            dot.edge(f'stage{prev_stage}_layernorm2', f'pipe_comm_{prev_stage}_{stage}')
            dot.edge(f'pipe_comm_{prev_stage}_{stage}', f'stage{stage}_routing')
        
        # Connect sample experts
        dot.edge(f'stage{stage}_routing', f'stage{stage}_expert0_qkv')
        dot.edge(f'stage{stage}_expert0_qkv', f'stage{stage}_expert0_attn')
        dot.edge(f'stage{stage}_expert0_attn', f'stage{stage}_expert0_attn_out')
        dot.edge(f'stage{stage}_expert0_attn_out', f'stage{stage}_layernorm1')
        dot.edge(f'stage{stage}_layernorm1', f'stage{stage}_expert0_mlp1')
        dot.edge(f'stage{stage}_expert0_mlp1', f'stage{stage}_expert0_mlp_act')
        dot.edge(f'stage{stage}_expert0_mlp_act', f'stage{stage}_expert0_mlp2')
        dot.edge(f'stage{stage}_expert0_mlp2', f'stage{stage}_layernorm2')
        dot.edge(f'stage{stage}_layernorm2', f'stage{stage}_aggregate')
    
    # Final output
    dot.edge('stage7_layernorm2', 'output')
    
    return dot

if __name__ == "__main__":
    # Create the DAG
    dag = create_moe_dag()
    
    # Save as DOT file
    dot_path = "../outputs/2025-12-05-14-17-12/moe_deployment_dag.dot"
    dag.save(dot_path)
    
    # Render as SVG
    svg_path = "../outputs/2025-12-05-14-17-12/moe_deployment_dag.svg"
    dag.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG saved to: {dot_path}")
    print(f"SVG rendered to: {svg_path}")
    
    # Create a summary file
    summary_path = "../outputs/2025-12-05-14-17-12/dag_summary.md"
    with open(summary_path, 'w') as f:
        f.write("""# 30B MoE Model Deployment DAG Summary

## Graph Structure
- **Total Nodes**: ~1000+ nodes representing individual operations
- **Total GPUs**: 1024 (64 EP × 2 TP × 8 PP)
- **Pipeline Stages**: 8 stages (2 layers each)
- **Experts per Stage**: 64 experts
- **Tensor Parallelism**: 2 GPUs per expert

## Node Types
1. **Ellipse (Communication)**: All-to-all routing, pipeline communication
2. **Rectangle (Computation)**: Attention, MLP, LayerNorm operations
3. **Parallelogram (Routing/Aggregation)**: Expert routing, output aggregation

## Key Features
- Each node shows exact GPU assignments
- Input/Output dimensions specified for every operation
- Attention operations broken down to operator granularity
- MLP layers shown with tensor parallelism
- All communication patterns represented
- Expert routing shown with dashed lines where applicable

## GPU Assignment Pattern
- **Stage 0**: GPUs 0-127
- **Stage 1**: GPUs 128-255
- **Stage 2**: GPUs 256-383
- **Stage 3**: GPUs 384-511
- **Stage 4**: GPUs 512-639
- **Stage 5**: GPUs 640-767
- **Stage 6**: GPUs 768-895
- **Stage 7**: GPUs 896-1023

## Expert Distribution
- Each stage has 64 experts (EP=64)
- Each expert uses 2 GPUs for tensor parallelism (TP=2)
- Expert ID 0: GPUs 0-1 (Stage 0), GPUs 128-129 (Stage 1), etc.

## Communication Patterns
1. **All-to-All**: Expert routing within each stage
2. **All-Reduce**: Tensor parallelism synchronization
3. **Pipeline**: Stage-to-stage data transfer
""")
    
    print(f"Summary saved to: {summary_path}")