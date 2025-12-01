#!/usr/bin/env python3
"""
EP64_TP2_PP1 Deployment DAG Generator
Generates a complete DAG for the corrected parallel strategy with:
- 128 GPUs total
- 64 expert parallel groups
- 2-way tensor parallelism
- Perfect load balancing (1 expert per GPU)
- Complete operator-level detail
"""

import os

def generate_ep64_tp2_dag():
    """Generate the complete EP64_TP2_PP1 deployment DAG"""
    
    dot_content = '''digraph EP64_TP2_PP1_Deployment {
    rankdir=TB;
    bgcolor=white;
    fontname="Arial";
    fontsize=12;
    
    // Graph attributes
    node [fontname="Arial", fontsize=10, shape=box, style=filled, fillcolor=lightblue];
    edge [fontname="Arial", fontsize=9, arrowhead=normal, color=black];
    
    // Input node (ellipse for data)
    Input [shape=ellipse, fillcolor=lightgreen, label="Input\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=1024]"];
    
    // Embedding layer - TP2 split across GPU 0-1
    subgraph cluster_embedding {
        label="Embedding Layer (TP2)";
        style=rounded;
        fillcolor=lightyellow;
        
        // Embedding computation nodes
        Embed_GPU0 [label="Embed_GPU0\\nGPU: 0\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        Embed_GPU1 [label="Embed_GPU1\\nGPU: 1\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        
        // Embedding aggregation
        Embed_Agg [label="Embed_Aggregate\\nGPU: 0-1\\nInput: [batch_size=4, seq_len=2048, hidden=512]\\nOutput: [batch_size=4, seq_len=2048, hidden=1024]", shape=parallelogram, fillcolor=orange];
    }
    
    // Layer normalization after embedding
    LayerNorm_Embed [label="LayerNorm_Embed\\nGPU: 0-1\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=1024]", shape=box, fillcolor=lightblue];
    
    // Now generate all 16 layers with 64 experts each
    // Each layer has: MHA + Expert + FFN + Residual connections
    '''
    
    # Generate all 16 layers
    for layer in range(16):
        dot_content += f'''
    // Layer {layer} - Complete with 64 experts (EP64_TP2)
    subgraph cluster_layer_{layer} {{
        label="Layer {layer} (EP64_TP2)";
        style=rounded;
        fillcolor=lightcyan;
        '''
        
        # Multi-Head Attention for this layer
        dot_content += f'''
        // Multi-Head Attention - TP2 split
        MHA_Q_GPU0_{layer} [label="MHA_Q_GPU0_L{layer}\\nGPU: 0\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        MHA_K_GPU0_{layer} [label="MHA_K_GPU0_L{layer}\\nGPU: 0\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        MHA_V_GPU0_{layer} [label="MHA_V_GPU0_L{layer}\\nGPU: 0\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        
        MHA_Q_GPU1_{layer} [label="MHA_Q_GPU1_L{layer}\\nGPU: 1\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        MHA_K_GPU1_{layer} [label="MHA_K_GPU1_L{layer}\\nGPU: 1\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        MHA_V_GPU1_{layer} [label="MHA_V_GPU1_L{layer}\\nGPU: 1\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        
        // Attention computation and aggregation
        Attention_Comp_GPU0_{layer} [label="Attention_Comp_GPU0_L{layer}\\nGPU: 0\\nInput: [batch_size=4, seq_len=2048, hidden=512]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        Attention_Comp_GPU1_{layer} [label="Attention_Comp_GPU1_L{layer}\\nGPU: 1\\nInput: [batch_size=4, seq_len=2048, hidden=512]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        
        Attention_Agg_{layer} [label="Attention_Agg_L{layer}\\nGPU: 0-1\\nInput: [batch_size=4, seq_len=2048, hidden=512]\\nOutput: [batch_size=4, seq_len=2048, hidden=1024]", shape=parallelogram, fillcolor=orange];
        '''
        
        # Expert routing and gate computation
        dot_content += f'''
        // Expert Gate - determines which tokens go to which experts
        Gate_Compute_{layer} [label="Gate_Compute_L{layer}\\nGPU: 0-1\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, num_experts=64]", shape=box, fillcolor=lightblue];
        Gate_Softmax_{layer} [label="Gate_Softmax_L{layer}\\nGPU: 0-1\\nInput: [batch_size=4, seq_len=2048, num_experts=64]\\nOutput: [batch_size=4, seq_len=2048, num_experts=64]", shape=box, fillcolor=lightblue];
        '''
        
        # Generate all 64 experts for this layer
        for expert in range(64):
            gpu_base = expert * 2  # Each expert uses 2 GPUs for TP2
            gpu0 = gpu_base
            gpu1 = gpu_base + 1
            
            dot_content += f'''
        // Expert {expert} in Layer {layer} - EP64_TP2
        Expert_{layer}_{expert}_GPU0 [label="Expert_L{layer}_E{expert}_GPU0\\nGPU: {gpu0}\\nInput: [batch_size=4, seq_len=2048, hidden=16]\\nOutput: [batch_size=4, seq_len=2048, hidden=16]", shape=box, fillcolor=lightgreen];
        Expert_{layer}_{expert}_GPU1 [label="Expert_L{layer}_E{expert}_GPU1\\nGPU: {gpu1}\\nInput: [batch_size=4, seq_len=2048, hidden=16]\\nOutput: [batch_size=4, seq_len=2048, hidden=16]", shape=box, fillcolor=lightgreen];
        '''
        
        # Expert routing communication (dashed lines for gate selection)
        dot_content += f'''
        // Expert routing communication
        Expert_Route_Comm_{layer} [label="Expert_Route_Comm_L{layer}\\nGPU: 0-127\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=16]", shape=ellipse, fillcolor=yellow, style="dashed,filled"];
        '''
        
        # Expert aggregation
        dot_content += f'''
        // Expert aggregation - collect outputs from all experts
        Expert_Agg_{layer} [label="Expert_Aggregate_L{layer}\\nGPU: 0-127\\nInput: [batch_size=4, seq_len=2048, hidden=16]\\nOutput: [batch_size=4, seq_len=2048, hidden=1024]", shape=parallelogram, fillcolor=orange];
        '''
        
        # FFN after experts
        dot_content += f'''
        // FFN - TP2 split
        FFN_GPU0_{layer} [label="FFN_GPU0_L{layer}\\nGPU: 0\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        FFN_GPU1_{layer} [label="FFN_GPU1_L{layer}\\nGPU: 1\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=512]", shape=box, fillcolor=lightblue];
        
        FFN_Agg_{layer} [label="FFN_Aggregate_L{layer}\\nGPU: 0-1\\nInput: [batch_size=4, seq_len=2048, hidden=512]\\nOutput: [batch_size=4, seq_len=2048, hidden=1024]", shape=parallelogram, fillcolor=orange];
        '''
        
        # Layer normalization
        dot_content += f'''
        // Layer normalization
        LayerNorm_{layer} [label="LayerNorm_L{layer}\\nGPU: 0-1\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, hidden=1024]", shape=box, fillcolor=lightblue];
        '''
        
        dot_content += '\n    }'  # Close layer cluster
    
    # Final output processing
    dot_content += '''
    // Final output processing
    Output_Processing [label="Output_Processing\\nCPU\\nInput: [batch_size=4, seq_len=2048, hidden=1024]\\nOutput: [batch_size=4, seq_len=2048, vocab_size]", shape=box, fillcolor=lightcoral];
    
    // Output node
    Output [label="Output\\nInput: [batch_size=4, seq_len=2048, vocab_size]\\nOutput: [batch_size=4, seq_len=2048, vocab_size]", shape=ellipse, fillcolor=lightgreen];
    '''
    
    # Now add all the connections
    dot_content += '''
    // Connections - Input to Embedding
    Input -> Embed_GPU0;
    Input -> Embed_GPU1;
    Embed_GPU0 -> Embed_Agg;
    Embed_GPU1 -> Embed_Agg;
    Embed_Agg -> LayerNorm_Embed;
    '''
    
    # Connections for each layer
    for layer in range(16):
        if layer == 0:
            dot_content += f'''
    // Layer {layer} connections
    LayerNorm_Embed -> MHA_Q_GPU0_{layer};
    LayerNorm_Embed -> MHA_K_GPU0_{layer};
    LayerNorm_Embed -> MHA_V_GPU0_{layer};
    LayerNorm_Embed -> MHA_Q_GPU1_{layer};
    LayerNorm_Embed -> MHA_K_GPU1_{layer};
    LayerNorm_Embed -> MHA_V_GPU1_{layer};
    '''
        else:
            dot_content += f'''
    // Layer {layer} connections
    LayerNorm_{layer-1} -> MHA_Q_GPU0_{layer};
    LayerNorm_{layer-1} -> MHA_K_GPU0_{layer};
    LayerNorm_{layer-1} -> MHA_V_GPU0_{layer};
    LayerNorm_{layer-1} -> MHA_Q_GPU1_{layer};
    LayerNorm_{layer-1} -> MHA_K_GPU1_{layer};
    LayerNorm_{layer-1} -> MHA_V_GPU1_{layer};
    '''
        
        # MHA connections
        dot_content += f'''
    MHA_Q_GPU0_{layer} -> Attention_Comp_GPU0_{layer};
    MHA_K_GPU0_{layer} -> Attention_Comp_GPU0_{layer};
    MHA_V_GPU0_{layer} -> Attention_Comp_GPU0_{layer};
    MHA_Q_GPU1_{layer} -> Attention_Comp_GPU1_{layer};
    MHA_K_GPU1_{layer} -> Attention_Comp_GPU1_{layer};
    MHA_V_GPU1_{layer} -> Attention_Comp_GPU1_{layer};
    
    Attention_Comp_GPU0_{layer} -> Attention_Agg_{layer};
    Attention_Comp_GPU1_{layer} -> Attention_Agg_{layer};
    '''
        
        # Gate connections
        dot_content += f'''
    Attention_Agg_{layer} -> Gate_Compute_{layer};
    Gate_Compute_{layer} -> Gate_Softmax_{layer};
    '''
        
        # Expert routing (dashed)
        dot_content += f'''
    Gate_Softmax_{layer} -> Expert_Route_Comm_{layer} [style=dashed];
    Attention_Agg_{layer} -> Expert_Route_Comm_{layer};
    '''
        
        # Expert connections
        for expert in range(64):
            dot_content += f'''
    Expert_Route_Comm_{layer} -> Expert_{layer}_{expert}_GPU0;
    Expert_Route_Comm_{layer} -> Expert_{layer}_{expert}_GPU1;
    Expert_{layer}_{expert}_GPU0 -> Expert_Agg_{layer};
    Expert_{layer}_{expert}_GPU1 -> Expert_Agg_{layer};
    '''
        
        # FFN and final connections
        dot_content += f'''
    Expert_Agg_{layer} -> FFN_GPU0_{layer};
    Expert_Agg_{layer} -> FFN_GPU1_{layer};
    FFN_GPU0_{layer} -> FFN_Agg_{layer};
    FFN_GPU1_{layer} -> FFN_Agg_{layer};
    FFN_Agg_{layer} -> LayerNorm_{layer};
    '''
    
    # Final output connections
    dot_content += '''
    // Final output connections
    LayerNorm_15 -> Output_Processing;
    Output_Processing -> Output;
    '''
    
    dot_content += '\n}'  # Close digraph
    
    return dot_content

if __name__ == "__main__":
    # Generate the DOT content
    dot_content = generate_ep64_tp2_dag()
    
    # Write to file
    output_dir = "../outputs/2025-12-01-16-48-00"
    os.makedirs(output_dir, exist_ok=True)
    
    dot_file = os.path.join(output_dir, "ep64_tp2_deployment.dot")
    with open(dot_file, 'w') as f:
        f.write(dot_content)
    
    print(f"Generated DOT file: {dot_file}")
    
    # Generate SVG using Graphviz
    svg_file = os.path.join(output_dir, "ep64_tp2_deployment.svg")
    try:
        import subprocess
        result = subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Generated SVG file: {svg_file}")
        else:
            print(f"Error generating SVG: {result.stderr}")
    except FileNotFoundError:
        print("Graphviz 'dot' command not found. SVG generation skipped.")
        print("To generate SVG manually, run: dot -Tsvg ep64_tp2_deployment.dot -o ep64_tp2_deployment.svg")
    
    print("\nDAG Summary:")
    print("- Total GPUs: 128")
    print("- Expert Parallelism: 64 groups")
    print("- Tensor Parallelism: 2-way")
    print("- Layers: 16")
    print("- Experts per layer: 64")
    print("- Load balancing: 1 expert per GPU (perfect)")
    print(f"- Total nodes: {dot_content.count('[')}")
    print(f"- File saved: {dot_file}")