#!/usr/bin/env python3

"""
Generate complete DAG for 30B MoE model with Hybrid EP64_TP16_PP4 strategy
All 16 layers fully detailed with proper pipeline connections
"""

def generate_layer_nodes(layer_num, gpu_start, gpu_range):
    """Generate nodes for a single layer"""
    nodes = []
    
    # Attention block nodes
    nodes.extend([
        f'qkv_c{layer_num}   [shape=box, label="QKV Column-Parallel Linear (TP=16)\\lINPUT: [128,10240,64]\\lOUTPUT: [128,10240,192]\\l"];',
        f'qkv_ar{layer_num}  [shape=ellipse, label="All-Reduce QKV (TP group)\\lINPUT: [128,10240,192]\\lOUTPUT: same\\l"];',
        f'sph{layer_num}     [shape=parallelogram, label="Split Heads (16 heads)\\lINPUT: [128,10240,192]\\lOUTPUT: [128,10240,16,64]\\l"];',
        f'sdp{layer_num}     [shape=box, label="Scaled Dot-Product (per head)\\lINPUT: [128,10240,16,64]\\lOUTPUT: same\\l"];',
        f'sm{layer_num}      [shape=box, label="Softmax (per head)\\lINPUT: same\\lOUTPUT: same\\l"];',
        f'do{layer_num}      [shape=box, label="Dropout (attn)\\lINPUT: same\\lOUTPUT: same\\l"];',
        f'mgh{layer_num}     [shape=parallelogram, label="Merge Heads\\lINPUT: [128,10240,16,64]\\lOUTPUT: [128,10240,1024]\\l"];',
        f'proj_r{layer_num}  [shape=box, label="Attn Proj Row-Parallel (TP=16)\\lINPUT: [128,10240,1024]\\lOUTPUT: [128,10240,64]\\l"];',
        f'proj_ar{layer_num} [shape=ellipse, label="All-Reduce Proj (TP)\\lINPUT: [128,10240,64]\\lOUTPUT: same\\l"];',
        f'norm{layer_num}    [shape=box, label="Add & LayerNorm\\lINPUT: [128,10240,1024]\\lOUTPUT: same\\l"];'
    ])
    
    # MoE block nodes
    nodes.extend([
        f'gate_c{layer_num}  [shape=box, label="Gate Column-Parallel (TP=16)\\lINPUT: [128,10240,1024]\\lOUTPUT: [128,10240,4]\\l"];',
        f'gate_ar{layer_num} [shape=ellipse, label="All-Reduce Gate (TP)\\lINPUT: same\\lOUTPUT: same\\l"];',
        f'route{layer_num}   [shape=parallelogram, label="Top-2 Routing\\lINPUT: same\\lOUTPUT: indices+weights\\l"];',
        f'a2a_s{layer_num}   [shape=ellipse, label="All-to-All Send tokens (EP=64)\\lINPUT: [128,10240,1024]\\lOUTPUT: scattered to 64 experts\\l"];',
        f'exp{layer_num}     [shape=box, label="Expert FFN Row-Parallel (TP=16)\\lINPUT: local tokens, hidden=1024\\lOUTPUT: same shape\\l"];',
        f'exp_ar{layer_num}  [shape=ellipse, label="All-Reduce Expert (TP)\\lINPUT: same\\lOUTPUT: same\\l"];',
        f'a2a_r{layer_num}   [shape=ellipse, label="All-to-All Recv results\\lINPUT: scattered\\lOUTPUT: [128,10240,1024]\\l"];',
        f'agg{layer_num}     [shape=parallelogram, label="Weighted Aggregate\\lINPUT: [128,10240,1024]\\lOUTPUT: same\\l"];',
        f'norm_m{layer_num}  [shape=box, label="Add & LayerNorm (MoE)\\lINPUT: same\\lOUTPUT: same\\l"];'
    ])
    
    return nodes

def generate_layer_edges(layer_num):
    """Generate edges for a single layer"""
    edges = []
    
    # Attention block edges
    edges.extend([
        f'split{layer_num//4 if layer_num % 4 == 0 else ""} -> qkv_c{layer_num};' if layer_num % 4 == 0 else f'norm_m{layer_num-1} -> qkv_c{layer_num};',
        f'qkv_c{layer_num} -> qkv_ar{layer_num};',
        f'qkv_ar{layer_num} -> sph{layer_num};',
        f'sph{layer_num} -> sdp{layer_num};',
        f'sdp{layer_num} -> sm{layer_num};',
        f'sm{layer_num} -> do{layer_num};',
        f'do{layer_num} -> mgh{layer_num};',
        f'mgh{layer_num} -> proj_r{layer_num};',
        f'proj_r{layer_num} -> proj_ar{layer_num};',
        f'proj_ar{layer_num} -> norm{layer_num};'
    ])
    
    # MoE block edges
    edges.extend([
        f'norm{layer_num} -> gate_c{layer_num};',
        f'gate_c{layer_num} -> gate_ar{layer_num};',
        f'gate_ar{layer_num} -> route{layer_num};',
        f'route{layer_num} -> a2a_s{layer_num};',
        f'a2a_s{layer_num} -> exp{layer_num};',
        f'exp{layer_num} -> exp_ar{layer_num};',
        f'exp_ar{layer_num} -> a2a_r{layer_num};',
        f'a2a_r{layer_num} -> agg{layer_num};',
        f'agg{layer_num} -> norm_m{layer_num};'
    ])
    
    return edges

def generate_pp_stage(stage_num, layer_start, layer_end, gpu_range, color):
    """Generate a complete pipeline stage"""
    lines = []
    
    # Stage cluster header
    lines.append(f'    // ============================================================')
    lines.append(f'    // PP Stage {stage_num} – GPUs {gpu_range}  (Layers {layer_start}-{layer_end-1})')
    lines.append(f'    // ============================================================')
    lines.append(f'    subgraph cluster_pp{stage_num} {{')
    lines.append(f'        label="PP Stage {stage_num} (Layers {layer_start}-{layer_end-1}) – GPUs {gpu_range}";')
    lines.append(f'        style=rounded;')
    lines.append(f'        color={color};')
    lines.append(f'')
    
    # Add split node for first layer in stage
    if stage_num > 0:
        lines.append(f'        split{stage_num} [shape=parallelogram,')
        lines.append(f'                label="Split to TP=16 GPUs\\lINPUT: [128,10240,1024]\\lOUTPUT: 16× [128,10240,64]\\l"];')
        lines.append(f'')
    
    # Add all layer nodes
    for layer in range(layer_start, layer_end):
        layer_nodes = generate_layer_nodes(layer, gpu_range[0], gpu_range)
        for node in layer_nodes:
            lines.append(f'        {node}')
        lines.append(f'')
    
    lines.append(f'    }}')
    lines.append(f'')
    
    return lines

def main():
    """Generate complete DAG"""
    
    # DAG header
    dot_content = '''digraph G {
    rankdir=TB;
    bgcolor="white";
    node [shape=box, style=rounded, fontname="Helvetica"];
    edge [fontname="Helvetica", fontsize=10];

    // ---------- legend ----------
    subgraph cluster_legend {
        label="Legend";
        style=dotted;
        comp [shape=box,     label="Computation"];
        comm [shape=ellipse, label="Communication"];
        route [shape=parallelogram, label="Split / Route / Aggregate"];
    }

    // ---------- input ----------
    input [shape=parallelogram,
           label="Input\\lINPUT: [batch_size=128, seq_len=10240, hidden=1024]\\lOUTPUT: same\\l"];

'''

    # Generate all 4 pipeline stages
    stages = [
        (0, 0, 4, "0-15", "blue"),    # PP Stage 0: Layers 0-3, GPUs 0-15
        (1, 4, 8, "16-31", "green"),  # PP Stage 1: Layers 4-7, GPUs 16-31
        (2, 8, 12, "32-47", "orange"), # PP Stage 2: Layers 8-11, GPUs 32-47
        (3, 12, 16, "48-63", "red")   # PP Stage 3: Layers 12-15, GPUs 48-63
    ]
    
    for stage_num, layer_start, layer_end, gpu_range, color in stages:
        stage_lines = generate_pp_stage(stage_num, layer_start, layer_end, gpu_range, color)
        dot_content += '\n'.join(stage_lines) + '\n'
    
    # Final output node
    dot_content += '''    // ---------- final output ----------
    output [shape=parallelogram,
            label="Aggregate Final Hidden\\lINPUT: [128,10240,1024]\\lOUTPUT: same\\l"];

'''

    # Generate all edges
    dot_content += '''    // ============================================================
    // Edges – forward only (guaranteed acyclic)
    // ============================================================
    input -> split0;

'''

    # Generate edges for all layers
    for layer in range(16):
        layer_edges = generate_layer_edges(layer)
        dot_content += f'    // Layer {layer}\n'
        for edge in layer_edges:
            dot_content += f'    {edge}\n'
        dot_content += '\n'
    
    # Add pipeline stage connection edges (the missing ones from feedback)
    dot_content += '''    // Pipeline stage connections
    norm_m3 -> split1;
    norm_m7 -> split2;
    norm_m11 -> split3;
    norm_m15 -> output;
'''

    # Close the graph
    dot_content += '}\n'
    
    # Write the complete DAG file
    with open('../outputs/2025-12-05-11-13-37/complete_parallel_strategy.dot', 'w') as f:
        f.write(dot_content)
    
    print("Generated complete DAG: complete_parallel_strategy.dot")
    
    # Also generate a Python file to create SVG
    python_content = '''import graphviz

# Read the dot file
with open('../outputs/2025-12-05-11-13-37/complete_parallel_strategy.dot', 'r') as f:
    dot_content = f.read()

# Render to SVG
source = graphviz.Source(dot_content)
source.render('../outputs/2025-12-05-11-13-37/complete_parallel_strategy', format='svg', cleanup=True)
print("Generated SVG: complete_parallel_strategy.svg")
'''
    
    with open('../outputs/2025-12-05-11-13-37/render_dag.py', 'w') as f:
        f.write(python_content)
    
    print("Generated render script: render_dag.py")

if __name__ == "__main__":
    main()