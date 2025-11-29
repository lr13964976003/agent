#!/usr/bin/env python3

import os

def generate_baseline_dag():
    """Generate corrected baseline DAG with TP=8, PP=2 without duplicate edges or cycles"""
    
    dot_content = '''// Baseline DAG: TP=8, PP=2 (Corrected - No Cycles)
digraph {
    nodesep=0.5 rankdir=TB ranksep=1.0 size="30,20"
    node [fontname=Arial fontsize=10]
    edge [fontname=Arial fontsize=8]
    
    // Input node
    input [label="Input\\nBatch:128, Seq:10000\\nDim:4096" fillcolor=lightcoral shape=ellipse]
    
    // Pipeline Stage 0 (GPUs 0-7)
    subgraph cluster_stage0 {
        fillcolor=lightgray label="Pipeline Stage 0 (GPUs 0-7)" style=rounded
        
        // Layer 0
        split_tp_l0 [label="Split TP\\nLayer 0" fillcolor=lightyellow shape=parallelogram]
        '''
    
    # Add attention nodes for layer 0 (8 GPUs)
    for gpu in range(8):
        dot_content += f'''        attn_l0_g{gpu} [label="Attention L0\\nGPU {gpu}\\nQKV Proj+Attn+Output" fillcolor=lightgreen shape=rectangle]
'''
    
    dot_content += '''        ar_attn_l0 [label="All-Reduce\\nAttention L0" fillcolor=lightblue shape=ellipse]
        
        // MLP nodes for layer 0
'''
    
    for gpu in range(8):
        dot_content += f'''        mlp1_l0_g{gpu} [label="MLP1 L0\\nGPU {gpu}\\nColParallel\\n16384->8192" fillcolor=lightgreen shape=rectangle]
        mlp2_l0_g{gpu} [label="MLP2 L0\\nGPU {gpu}\\nRowParallel\\n8192->16384" fillcolor=lightgreen shape=rectangle]
'''
    
    dot_content += '''        ar_mlp_l0 [label="All-Reduce\\nMLP L0" fillcolor=lightblue shape=ellipse]
        
        // LayerNorm for layer 0
'''
    
    for gpu in range(8):
        dot_content += f'''        norm_l0_g{gpu} [label="LayerNorm L0\\nGPU {gpu}" fillcolor=lightgreen shape=rectangle]
'''
    
    # Add connections for layer 0
    dot_content += '''        
        // Connections for layer 0
        split_tp_l0 -> attn_l0_g0
        split_tp_l0 -> attn_l0_g1
        split_tp_l0 -> attn_l0_g2
        split_tp_l0 -> attn_l0_g3
        split_tp_l0 -> attn_l0_g4
        split_tp_l0 -> attn_l0_g5
        split_tp_l0 -> attn_l0_g6
        split_tp_l0 -> attn_l0_g7
        
        attn_l0_g0 -> ar_attn_l0
        attn_l0_g1 -> ar_attn_l0
        attn_l0_g2 -> ar_attn_l0
        attn_l0_g3 -> ar_attn_l0
        attn_l0_g4 -> ar_attn_l0
        attn_l0_g5 -> ar_attn_l0
        attn_l0_g6 -> ar_attn_l0
        attn_l0_g7 -> ar_attn_l0
        
        ar_attn_l0 -> mlp1_l0_g0
        ar_attn_l0 -> mlp1_l0_g1
        ar_attn_l0 -> mlp1_l0_g2
        ar_attn_l0 -> mlp1_l0_g3
        ar_attn_l0 -> mlp1_l0_g4
        ar_attn_l0 -> mlp1_l0_g5
        ar_attn_l0 -> mlp1_l0_g6
        ar_attn_l0 -> mlp1_l0_g7
        
        mlp1_l0_g0 -> mlp2_l0_g0
        mlp1_l0_g1 -> mlp2_l0_g1
        mlp1_l0_g2 -> mlp2_l0_g2
        mlp1_l0_g3 -> mlp2_l0_g3
        mlp1_l0_g4 -> mlp2_l0_g4
        mlp1_l0_g5 -> mlp2_l0_g5
        mlp1_l0_g6 -> mlp2_l0_g6
        mlp1_l0_g7 -> mlp2_l0_g7
        
        mlp2_l0_g0 -> ar_mlp_l0
        mlp2_l0_g1 -> ar_mlp_l0
        mlp2_l0_g2 -> ar_mlp_l0
        mlp2_l0_g3 -> ar_mlp_l0
        mlp2_l0_g4 -> ar_mlp_l0
        mlp2_l0_g5 -> ar_mlp_l0
        mlp2_l0_g6 -> ar_mlp_l0
        mlp2_l0_g7 -> ar_mlp_l0
        
        ar_mlp_l0 -> norm_l0_g0
        ar_mlp_l0 -> norm_l0_g1
        ar_mlp_l0 -> norm_l0_g2
        ar_mlp_l0 -> norm_l0_g3
        ar_mlp_l0 -> norm_l0_g4
        ar_mlp_l0 -> norm_l0_g5
        ar_mlp_l0 -> norm_l0_g6
        ar_mlp_l0 -> norm_l0_g7
    }
    
    // Add remaining layers (1-15) following same pattern but without duplicate edges
'''

    # Add remaining layers 1-15
    for layer in range(1, 16):
        stage = 0 if layer < 8 else 1
        gpu_offset = 0 if layer < 8 else 8
        
        dot_content += f'''    
    // Layer {layer}
    split_tp_l{layer} [label="Split TP\\nLayer {layer}" fillcolor=lightyellow shape=parallelogram]
    ar_attn_l{layer} [label="All-Reduce\\nAttention L{layer}" fillcolor=lightblue shape=ellipse]
    ar_mlp_l{layer} [label="All-Reduce\\nMLP L{layer}" fillcolor=lightblue shape=ellipse]
'''
        
        # Add attention and MLP nodes
        for gpu in range(8):
            actual_gpu = gpu + gpu_offset
            dot_content += f'''    attn_l{layer}_g{actual_gpu} [label="Attention L{layer}\\nGPU {actual_gpu}\\nQKV Proj+Attn+Output" fillcolor=lightgreen shape=rectangle]
    mlp1_l{layer}_g{actual_gpu} [label="MLP1 L{layer}\\nGPU {actual_gpu}\\nColParallel\\n16384->8192" fillcolor=lightgreen shape=rectangle]
    mlp2_l{layer}_g{actual_gpu} [label="MLP2 L{layer}\\nGPU {actual_gpu}\\nRowParallel\\n8192->16384" fillcolor=lightgreen shape=rectangle]
    norm_l{layer}_g{actual_gpu} [label="LayerNorm L{layer}\\nGPU {actual_gpu}" fillcolor=lightgreen shape=rectangle]
'''
        
        # Add connections for this layer
        for gpu in range(8):
            actual_gpu = gpu + gpu_offset
            dot_content += f'''    split_tp_l{layer} -> attn_l{layer}_g{actual_gpu}
    attn_l{layer}_g{actual_gpu} -> ar_attn_l{layer}
    ar_attn_l{layer} -> mlp1_l{layer}_g{actual_gpu}
    mlp1_l{layer}_g{actual_gpu} -> mlp2_l{layer}_g{actual_gpu}
    mlp2_l{layer}_g{actual_gpu} -> ar_mlp_l{layer}
    ar_mlp_l{layer} -> norm_l{layer}_g{actual_gpu}
'''

    # Add inter-layer connections
    for layer in range(1, 16):
        gpu_offset = 0 if layer < 8 else 8
        prev_gpu_offset = 0 if layer-1 < 8 else 8
        
        for gpu in range(8):
            actual_gpu = gpu + gpu_offset
            prev_actual_gpu = gpu + prev_gpu_offset
            dot_content += f'''    norm_l{layer-1}_g{prev_actual_gpu} -> split_tp_l{layer}
'''

    # Add output connections
    for gpu in range(8):
        actual_gpu = gpu + 8  # Last stage GPUs
        dot_content += f'''    norm_l15_g{actual_gpu} -> output
'''

    dot_content += '''    
    // Output node
    output [label="Output\\nBatch:128, Seq:10000\\nDim:4096" fillcolor=lightcoral shape=ellipse]
    
    // Input connections
    input -> split_tp_l0
}
'''
    
    return dot_content

def generate_optimized_dag():
    """Generate corrected optimized DAG with proper residual connections"""
    
    dot_content = '''// Optimized DAG: Layer-wise Partitioning (Corrected - No Cycles)
digraph {
    nodesep=0.5 rankdir=TB ranksep=1.0 size="30,20"
    node [fontname=Arial fontsize=10]
    edge [fontname=Arial fontsize=8]
    
    // Input node
    input [label="Input\\nBatch:128, Seq:10000\\nDim:4096" fillcolor=lightcoral shape=ellipse]
    
    // GPU 0: Layers 0-3
    subgraph cluster_gpu0 {
        fillcolor=lightblue label="GPU 0: Layers 0-3 (Cache Optimized)" style=rounded
        
        // Layer 0
        attn_l0_g0 [label="Attention L0\\nGPU 0\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l0_g0 [label="MLP L0\\nGPU 0\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l0_g0 [label="LayerNorm L0\\nGPU 0" fillcolor=lightgreen shape=rectangle]
        resid_l0_g0 [label="ResidAdd L0\\nGPU 0" fillcolor=orange shape=parallelogram]
        
        // Layer 1
        attn_l1_g0 [label="Attention L1\\nGPU 0\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l1_g0 [label="MLP L1\\nGPU 0\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l1_g0 [label="LayerNorm L1\\nGPU 0" fillcolor=lightgreen shape=rectangle]
        resid_l1_g0 [label="ResidAdd L1\\nGPU 0" fillcolor=orange shape=parallelogram]
        
        // Layer 2
        attn_l2_g0 [label="Attention L2\\nGPU 0\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l2_g0 [label="MLP L2\\nGPU 0\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l2_g0 [label="LayerNorm L2\\nGPU 0" fillcolor=lightgreen shape=rectangle]
        resid_l2_g0 [label="ResidAdd L2\\nGPU 0" fillcolor=orange shape=parallelogram]
        
        // Layer 3
        attn_l3_g0 [label="Attention L3\\nGPU 0\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l3_g0 [label="MLP L3\\nGPU 0\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l3_g0 [label="LayerNorm L3\\nGPU 0" fillcolor=lightgreen shape=rectangle]
        resid_l3_g0 [label="ResidAdd L3\\nGPU 0" fillcolor=orange shape=parallelogram]
    }
    
    // GPU 1: Layers 4-7
    subgraph cluster_gpu1 {
        fillcolor=lightblue label="GPU 1: Layers 4-7 (Cache Optimized)" style=rounded
        
        // Layer 4
        attn_l4_g1 [label="Attention L4\\nGPU 1\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l4_g1 [label="MLP L4\\nGPU 1\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l4_g1 [label="LayerNorm L4\\nGPU 1" fillcolor=lightgreen shape=rectangle]
        resid_l4_g1 [label="ResidAdd L4\\nGPU 1" fillcolor=orange shape=parallelogram]
        
        // Layer 5
        attn_l5_g1 [label="Attention L5\\nGPU 1\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l5_g1 [label="MLP L5\\nGPU 1\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l5_g1 [label="LayerNorm L5\\nGPU 1" fillcolor=lightgreen shape=rectangle]
        resid_l5_g1 [label="ResidAdd L5\\nGPU 1" fillcolor=orange shape=parallelogram]
        
        // Layer 6
        attn_l6_g1 [label="Attention L6\\nGPU 1\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l6_g1 [label="MLP L6\\nGPU 1\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l6_g1 [label="LayerNorm L6\\nGPU 1" fillcolor=lightgreen shape=rectangle]
        resid_l6_g1 [label="ResidAdd L6\\nGPU 1" fillcolor=orange shape=parallelogram]
        
        // Layer 7
        attn_l7_g1 [label="Attention L7\\nGPU 1\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l7_g1 [label="MLP L7\\nGPU 1\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l7_g1 [label="LayerNorm L7\\nGPU 1" fillcolor=lightgreen shape=rectangle]
        resid_l7_g1 [label="ResidAdd L7\\nGPU 1" fillcolor=orange shape=parallelogram]
    }
    
    // GPU 2: Layers 8-11
    subgraph cluster_gpu2 {
        fillcolor=lightblue label="GPU 2: Layers 8-11 (Cache Optimized)" style=rounded
        
        // Layer 8
        attn_l8_g2 [label="Attention L8\\nGPU 2\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l8_g2 [label="MLP L8\\nGPU 2\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l8_g2 [label="LayerNorm L8\\nGPU 2" fillcolor=lightgreen shape=rectangle]
        resid_l8_g2 [label="ResidAdd L8\\nGPU 2" fillcolor=orange shape=parallelogram]
        
        // Layer 9
        attn_l9_g2 [label="Attention L9\\nGPU 2\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l9_g2 [label="MLP L9\\nGPU 2\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l9_g2 [label="LayerNorm L9\\nGPU 2" fillcolor=lightgreen shape=rectangle]
        resid_l9_g2 [label="ResidAdd L9\\nGPU 2" fillcolor=orange shape=parallelogram]
        
        // Layer 10
        attn_l10_g2 [label="Attention L10\\nGPU 2\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l10_g2 [label="MLP L10\\nGPU 2\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l10_g2 [label="LayerNorm L10\\nGPU 2" fillcolor=lightgreen shape=rectangle]
        resid_l10_g2 [label="ResidAdd L10\\nGPU 2" fillcolor=orange shape=parallelogram]
        
        // Layer 11
        attn_l11_g2 [label="Attention L11\\nGPU 2\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l11_g2 [label="MLP L11\\nGPU 2\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l11_g2 [label="LayerNorm L11\\nGPU 2" fillcolor=lightgreen shape=rectangle]
        resid_l11_g2 [label="ResidAdd L11\\nGPU 2" fillcolor=orange shape=parallelogram]
    }
    
    // GPU 3: Layers 12-15
    subgraph cluster_gpu3 {
        fillcolor=lightblue label="GPU 3: Layers 12-15 (Cache Optimized)" style=rounded
        
        // Layer 12
        attn_l12_g3 [label="Attention L12\\nGPU 3\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l12_g3 [label="MLP L12\\nGPU 3\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l12_g3 [label="LayerNorm L12\\nGPU 3" fillcolor=lightgreen shape=rectangle]
        resid_l12_g3 [label="ResidAdd L12\\nGPU 3" fillcolor=orange shape=parallelogram]
        
        // Layer 13
        attn_l13_g3 [label="Attention L13\\nGPU 3\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l13_g3 [label="MLP L13\\nGPU 3\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l13_g3 [label="LayerNorm L13\\nGPU 3" fillcolor=lightgreen shape=rectangle]
        resid_l13_g3 [label="ResidAdd L13\\nGPU 3" fillcolor=orange shape=parallelogram]
        
        // Layer 14
        attn_l14_g3 [label="Attention L14\\nGPU 3\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l14_g3 [label="MLP L14\\nGPU 3\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l14_g3 [label="LayerNorm L14\\nGPU 3" fillcolor=lightgreen shape=rectangle]
        resid_l14_g3 [label="ResidAdd L14\\nGPU 3" fillcolor=orange shape=parallelogram]
        
        // Layer 15
        attn_l15_g3 [label="Attention L15\\nGPU 3\\nQKV Proj+Attn+Output\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        mlp_l15_g3 [label="MLP L15\\nGPU 3\\n16384->16384\\nComplete Layer" fillcolor=lightgreen shape=rectangle]
        norm_l15_g3 [label="LayerNorm L15\\nGPU 3" fillcolor=lightgreen shape=rectangle]
        resid_l15_g3 [label="ResidAdd L15\\nGPU 3" fillcolor=orange shape=parallelogram]
    }
    
    // Transfer nodes
    transfer_g0_g1 [label="Transfer\\nGPU0->GPU1\\n819.2MB" fillcolor=red shape=ellipse]
    transfer_g1_g2 [label="Transfer\\nGPU1->GPU2\\n819.2MB" fillcolor=red shape=ellipse]
    transfer_g2_g3 [label="Transfer\\nGPU2->GPU3\\n819.2MB" fillcolor=red shape=ellipse]
    
    // Output node
    output [label="Output\\nBatch:128, Seq:10000\\nDim:4096" fillcolor=lightcoral shape=ellipse]
    
    // Connections - Fixed to avoid cycles
    // Layer 0
    input -> attn_l0_g0
    attn_l0_g0 -> mlp_l0_g0
    mlp_l0_g0 -> norm_l0_g0
    norm_l0_g0 -> resid_l0_g0
    input -> resid_l0_g0
    resid_l0_g0 -> attn_l1_g0
    
    // Layer 1
    attn_l1_g0 -> mlp_l1_g0
    mlp_l1_g0 -> norm_l1_g0
    norm_l1_g0 -> resid_l1_g0
    resid_l0_g0 -> resid_l1_g0
    resid_l1_g0 -> attn_l2_g0
    
    // Layer 2
    attn_l2_g0 -> mlp_l2_g0
    mlp_l2_g0 -> norm_l2_g0
    norm_l2_g0 -> resid_l2_g0
    resid_l1_g0 -> resid_l2_g0
    resid_l2_g0 -> attn_l3_g0
    
    // Layer 3
    attn_l3_g0 -> mlp_l3_g0
    mlp_l3_g0 -> norm_l3_g0
    norm_l3_g0 -> resid_l3_g0
    resid_l2_g0 -> resid_l3_g0
    resid_l3_g0 -> transfer_g0_g1
    
    // Transfer to GPU 1
    transfer_g0_g1 -> attn_l4_g1
    
    // Layer 4
    attn_l4_g1 -> mlp_l4_g1
    mlp_l4_g1 -> norm_l4_g1
    norm_l4_g1 -> resid_l4_g1
    transfer_g0_g1 -> resid_l4_g1
    resid_l4_g1 -> attn_l5_g1
    
    // Layer 5
    attn_l5_g1 -> mlp_l5_g1
    mlp_l5_g1 -> norm_l5_g1
    norm_l5_g1 -> resid_l5_g1
    resid_l4_g1 -> resid_l5_g1
    resid_l5_g1 -> attn_l6_g1
    
    // Layer 6
    attn_l6_g1 -> mlp_l6_g1
    mlp_l6_g1 -> norm_l6_g1
    norm_l6_g1 -> resid_l6_g1
    resid_l5_g1 -> resid_l6_g1
    resid_l6_g1 -> attn_l7_g1
    
    // Layer 7
    attn_l7_g1 -> mlp_l7_g1
    mlp_l7_g1 -> norm_l7_g1
    norm_l7_g1 -> resid_l7_g1
    resid_l6_g1 -> resid_l7_g1
    resid_l7_g1 -> transfer_g1_g2
    
    // Transfer to GPU 2
    transfer_g1_g2 -> attn_l8_g2
    
    // Layer 8
    attn_l8_g2 -> mlp_l8_g2
    mlp_l8_g2 -> norm_l8_g2
    norm_l8_g2 -> resid_l8_g2
    transfer_g1_g2 -> resid_l8_g2
    resid_l8_g2 -> attn_l9_g2
    
    // Layer 9
    attn_l9_g2 -> mlp_l9_g2
    mlp_l9_g2 -> norm_l9_g2
    norm_l9_g2 -> resid_l9_g2
    resid_l8_g2 -> resid_l9_g2
    resid_l9_g2 -> attn_l10_g2
    
    // Layer 10
    attn_l10_g2 -> mlp_l10_g2
    mlp_l10_g2 -> norm_l10_g2
    norm_l10_g2 -> resid_l10_g2
    resid_l9_g2 -> resid_l10_g2
    resid_l10_g2 -> attn_l11_g2
    
    // Layer 11
    attn_l11_g2 -> mlp_l11_g2
    mlp_l11_g2 -> norm_l11_g2
    norm_l11_g2 -> resid_l11_g2
    resid_l10_g2 -> resid_l11_g2
    resid_l11_g2 -> transfer_g2_g3
    
    // Transfer to GPU 3
    transfer_g2_g3 -> attn_l12_g3
    
    // Layer 12
    attn_l12_g3 -> mlp_l12_g3
    mlp_l12_g3 -> norm_l12_g3
    norm_l12_g3 -> resid_l12_g3
    transfer_g2_g3 -> resid_l12_g3
    resid_l12_g3 -> attn_l13_g3
    
    // Layer 13
    attn_l13_g3 -> mlp_l13_g3
    mlp_l13_g3 -> norm_l13_g3
    norm_l13_g3 -> resid_l13_g3
    resid_l12_g3 -> resid_l13_g3
    resid_l13_g3 -> attn_l14_g3
    
    // Layer 14
    attn_l14_g3 -> mlp_l14_g3
    mlp_l14_g3 -> norm_l14_g3
    norm_l14_g3 -> resid_l14_g3
    resid_l13_g3 -> resid_l14_g3
    resid_l14_g3 -> attn_l15_g3
    
    // Layer 15
    attn_l15_g3 -> mlp_l15_g3
    mlp_l15_g3 -> norm_l15_g3
    norm_l15_g3 -> resid_l15_g3
    resid_l14_g3 -> resid_l15_g3
    resid_l15_g3 -> output
}
'''
    
    return dot_content

def main():
    # Generate corrected baseline DAG
    baseline_content = generate_baseline_dag()
    baseline_path = "../outputs/2025-11-29-15-46-15/baseline_dag_corrected.dot"
    
    with open(baseline_path, 'w') as f:
        f.write(baseline_content)
    
    # Generate corrected optimized DAG
    optimized_content = generate_optimized_dag()
    optimized_path = "../outputs/2025-11-29-15-46-15/optimized_dag_corrected.dot"
    
    with open(optimized_path, 'w') as f:
        f.write(optimized_content)
    
    print(f"Generated corrected DAGs:")
    print(f"- {baseline_path}")
    print(f"- {optimized_path}")
    
    # Convert to SVG
    os.system(f"dot -Tsvg {baseline_path} -o {baseline_path.replace('.dot', '.svg')}")
    os.system(f"dot -Tsvg {optimized_path} -o {optimized_path.replace('.dot', '.svg')}")
    
    print(f"Generated SVG images:")
    print(f"- {baseline_path.replace('.dot', '.svg')}")
    print(f"- {optimized_path.replace('.dot', '.svg')}")

if __name__ == "__main__":
    main()