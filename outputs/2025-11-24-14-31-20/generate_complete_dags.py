#!/usr/bin/env python3
"""
Generate complete MoE DAGs for Large-Scale Cross-Node Expert Parallelism
This script creates engineering-grade DAGs with complete connectivity and GPU assignments
"""

import os

def generate_proposed_ep16_dag():
    """Generate complete DAG for proposed EP=16 method with 1 expert per GPU"""
    
    nodes = []
    edges = []
    
    # Model input
    nodes.append('input [label="Model Input\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=box, style=filled, fillcolor=lightblue]')
    
    # Generate all 16 complete layers
    for layer_idx in range(16):
        # MHA Layer
        nodes.append(f'layer{layer_idx}_mha [label="MHA Layer {layer_idx}\\nAll GPUs\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=rectangle, style=filled, fillcolor=lightcyan]')
        
        # LayerNorm after MHA
        nodes.append(f'layer{layer_idx}_ln1 [label="LayerNorm MHA {layer_idx}\\nAll GPUs\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=rectangle]')
        
        # MHA Residual Add
        nodes.append(f'layer{layer_idx}_add1 [label="MHA Residual Add {layer_idx}\\nInput1: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nInput2: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=rectangle, style=filled, fillcolor=yellow]')
        
        # Gate Layer
        nodes.append(f'layer{layer_idx}_gate [label="Gate Layer {layer_idx}\\nAll GPUs\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, num_experts=16]", shape=parallelogram, style=filled, fillcolor=lightgreen]')
        
        # Generate all 16 experts for this layer
        for expert_idx in range(16):
            gpu_id = expert_idx
            # Route to expert
            nodes.append(f'layer{layer_idx}_route_exp{expert_idx} [label="Route to Expert {expert_idx}\\nGPU {gpu_id}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [tokens_per_expert, hidden_dim=4096]", shape=ellipse, style=dashed, fillcolor=orange]')
            
            # Expert computation
            nodes.append(f'layer{layer_idx}_expert{expert_idx} [label="MLP Expert {expert_idx}\\nGPU {gpu_id}\\nInput: [tokens_per_expert, hidden_dim=4096]\\nOutput: [tokens_per_expert, hidden_dim=4096]\\nHidden: [tokens_per_expert, 16384]", shape=rectangle, style=filled, fillcolor=lightcoral]')
            
            # Aggregate from expert
            nodes.append(f'layer{layer_idx}_agg_exp{expert_idx} [label="Aggregate from Expert {expert_idx}\\nGPU {gpu_id}\\nInput: [tokens_per_expert, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=ellipse, style=filled, fillcolor=lightpink]')
        
        # MoE Final Aggregation
        nodes.append(f'layer{layer_idx}_moe_agg [label="MoE Final Aggregation {layer_idx}\\nAll GPUs\\nInput: 16×[batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=ellipse, style=filled, fillcolor=gold]')
        
        # LayerNorm after MoE
        nodes.append(f'layer{layer_idx}_ln2 [label="LayerNorm MoE {layer_idx}\\nAll GPUs\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=rectangle]')
        
        # MoE Residual Add
        nodes.append(f'layer{layer_idx}_moe_output [label="MoE Residual Add {layer_idx}\\nInput1: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nInput2: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=rectangle, style=filled, fillcolor=yellow]')
    
    # Model output
    nodes.append('output [label="Model Output\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=box, style=filled, fillcolor=lightblue]')
    
    # Create all edges with complete connectivity
    # Input to layer 0
    edges.append('input -> layer0_mha')
    edges.append('layer0_mha -> layer0_ln1')
    edges.append('layer0_ln1 -> layer0_add1')
    edges.append('input -> layer0_add1')
    edges.append('layer0_add1 -> layer0_gate')
    
    # Connect all experts in layer 0
    for expert_idx in range(16):
        edges.append(f'layer0_gate -> layer0_route_exp{expert_idx}')
        edges.append(f'layer0_route_exp{expert_idx} -> layer0_expert{expert_idx}')
        edges.append(f'layer0_expert{expert_idx} -> layer0_agg_exp{expert_idx}')
        edges.append(f'layer0_agg_exp{expert_idx} -> layer0_moe_agg')
    
    edges.append('layer0_moe_agg -> layer0_ln2')
    edges.append('layer0_ln2 -> layer0_moe_output')
    edges.append('layer0_add1 -> layer0_moe_output')
    
    # Connect all layers sequentially
    for layer_idx in range(1, 16):
        prev_layer = layer_idx - 1
        edges.append(f'layer{prev_layer}_moe_output -> layer{layer_idx}_mha')
        edges.append(f'layer{layer_idx}_mha -> layer{layer_idx}_ln1')
        edges.append(f'layer{layer_idx}_ln1 -> layer{layer_idx}_add1')
        edges.append(f'layer{prev_layer}_moe_output -> layer{layer_idx}_add1')
        edges.append(f'layer{layer_idx}_add1 -> layer{layer_idx}_gate')
        
        # Connect all experts in this layer
        for expert_idx in range(16):
            gpu_id = expert_idx
            edges.append(f'layer{layer_idx}_gate -> layer{layer_idx}_route_exp{expert_idx}')
            edges.append(f'layer{layer_idx}_route_exp{expert_idx} -> layer{layer_idx}_expert{expert_idx}')
            edges.append(f'layer{layer_idx}_expert{expert_idx} -> layer{layer_idx}_agg_exp{expert_idx}')
            edges.append(f'layer{layer_idx}_agg_exp{expert_idx} -> layer{layer_idx}_moe_agg')
        
        edges.append(f'layer{layer_idx}_moe_agg -> layer{layer_idx}_ln2')
        edges.append(f'layer{layer_idx}_ln2 -> layer{layer_idx}_moe_output')
        edges.append(f'layer{layer_idx}_add1 -> layer{layer_idx}_moe_output')
    
    # Final output
    edges.append('layer15_moe_output -> output')
    
    # Build the complete DOT content
    dot_content = 'digraph Proposed_Large_EP16_Final {\n    rankdir=TB;\n    splines=ortho;\n    node [fontname="Arial", fontsize=10];\n    \n'
    dot_content += '    ' + '\n    '.join(nodes) + '\n    \n'
    dot_content += '    ' + '\n    '.join(edges) + '\n}'
    
    return dot_content

def generate_baseline_dag():
    """Generate complete DAG for baseline method with TP=8, PP=2, 8 experts/GPU"""
    
    nodes = []
    edges = []
    
    # Model input
    nodes.append('input [label="Model Input\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=box, style=filled, fillcolor=lightblue]')
    
    # Generate all 16 complete layers with TP=8, PP=2
    for layer_idx in range(16):
        stage = 0 if layer_idx < 8 else 1
        gpu_start = 0 if stage == 0 else 8
        gpu_end = 7 if stage == 0 else 15
        
        # MHA Layer
        nodes.append(f'layer{layer_idx}_mha [label="MHA Layer {layer_idx}\\nTP=8 across GPUs {gpu_start}-{gpu_end}\\nStage {stage}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=rectangle, style=filled, fillcolor=lightcyan]')
        
        # LayerNorm after MHA
        nodes.append(f'layer{layer_idx}_ln1 [label="LayerNorm MHA {layer_idx}\\nTP=8 across GPUs {gpu_start}-{gpu_end}\\nStage {stage}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=rectangle]')
        
        # MHA Residual Add
        nodes.append(f'layer{layer_idx}_add1 [label="MHA Residual Add {layer_idx}\\nInput1: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nInput2: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=rectangle, style=filled, fillcolor=yellow]')
        
        # Gate Layer
        nodes.append(f'layer{layer_idx}_gate [label="Gate Layer {layer_idx}\\nTP=8 across GPUs {gpu_start}-{gpu_end}\\nStage {stage}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, num_experts=16]", shape=parallelogram, style=filled, fillcolor=lightgreen]')
        
        # Generate all 16 experts distributed across 8 GPUs in the stage
        for expert_idx in range(16):
            gpu_id = gpu_start + (expert_idx // 2)  # 2 experts per GPU
            
            # Route to GPU
            nodes.append(f'layer{layer_idx}_route_gpu{gpu_id} [label="Route to GPU {gpu_id}\\nStage {stage}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [tokens_per_gpu, hidden_dim=4096]", shape=ellipse, style=dashed, fillcolor=orange]')
            
            # Expert computation (8 experts on this GPU)
            nodes.append(f'layer{layer_idx}_experts_gpu{gpu_id} [label="8 Experts GPU {gpu_id}\\nStage {stage}\\nInput: [tokens_per_gpu, hidden_dim=4096]\\nOutput: [tokens_per_gpu, hidden_dim=4096]\\nHidden: [tokens_per_gpu, 16384]", shape=rectangle, style=filled, fillcolor=lightcoral]')
            
            # Aggregate from GPU
            nodes.append(f'layer{layer_idx}_agg_gpu{gpu_id} [label="Aggregate GPU {gpu_id}\\nStage {stage}\\nInput: [tokens_per_gpu, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=ellipse, style=filled, fillcolor=lightpink]')
        
        # MoE Final Aggregation
        nodes.append(f'layer{layer_idx}_moe_agg [label="MoE Final Aggregation {layer_idx}\\nTP=8 across GPUs {gpu_start}-{gpu_end}\\nStage {stage}\\nInput: 16×[batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=ellipse, style=filled, fillcolor=gold]')
        
        # LayerNorm after MoE
        nodes.append(f'layer{layer_idx}_ln2 [label="LayerNorm MoE {layer_idx}\\nTP=8 across GPUs {gpu_start}-{gpu_end}\\nStage {stage}\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=rectangle]')
        
        # MoE Residual Add
        nodes.append(f'layer{layer_idx}_moe_output [label="MoE Residual Add {layer_idx}\\nInput1: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nInput2: [batch_size=128, seq_len=10000, hidden_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=rectangle, style=filled, fillcolor=yellow]')
    
    # Model output
    nodes.append('output [label="Model Output\\nInput: [batch_size=128, seq_len=10000, hidden_dim=4096]", shape=box, style=filled, fillcolor=lightblue]')
    
    # Create all edges with complete connectivity
    # Input to layer 0
    edges.append('input -> layer0_mha')
    edges.append('layer0_mha -> layer0_ln1')
    edges.append('layer0_ln1 -> layer0_add1')
    edges.append('input -> layer0_add1')
    edges.append('layer0_add1 -> layer0_gate')
    
    # Connect all experts in layer 0
    for gpu_id in range(8):
        edges.append(f'layer0_gate -> layer0_route_gpu{gpu_id}')
        edges.append(f'layer0_route_gpu{gpu_id} -> layer0_experts_gpu{gpu_id}')
        edges.append(f'layer0_experts_gpu{gpu_id} -> layer0_agg_gpu{gpu_id}')
        edges.append(f'layer0_agg_gpu{gpu_id} -> layer0_moe_agg')
    
    edges.append('layer0_moe_agg -> layer0_ln2')
    edges.append('layer0_ln2 -> layer0_moe_output')
    edges.append('layer0_add1 -> layer0_moe_output')
    
    # Connect all layers sequentially
    for layer_idx in range(1, 16):
        prev_layer = layer_idx - 1
        stage = 0 if layer_idx < 8 else 1
        edges.append(f'layer{prev_layer}_moe_output -> layer{layer_idx}_mha')
        edges.append(f'layer{layer_idx}_mha -> layer{layer_idx}_ln1')
        edges.append(f'layer{layer_idx}_ln1 -> layer{layer_idx}_add1')
        edges.append(f'layer{prev_layer}_moe_output -> layer{layer_idx}_add1')
        edges.append(f'layer{layer_idx}_add1 -> layer{layer_idx}_gate')
        
        # Connect all experts in this layer
        gpu_start = 0 if stage == 0 else 8
        gpu_end = 7 if stage == 0 else 15
        for gpu_id in range(gpu_start, gpu_end + 1):
            edges.append(f'layer{layer_idx}_gate -> layer{layer_idx}_route_gpu{gpu_id}')
            edges.append(f'layer{layer_idx}_route_gpu{gpu_id} -> layer{layer_idx}_experts_gpu{gpu_id}')
            edges.append(f'layer{layer_idx}_experts_gpu{gpu_id} -> layer{layer_idx}_agg_gpu{gpu_id}')
            edges.append(f'layer{layer_idx}_agg_gpu{gpu_id} -> layer{layer_idx}_moe_agg')
        
        edges.append(f'layer{layer_idx}_moe_agg -> layer{layer_idx}_ln2')
        edges.append(f'layer{layer_idx}_ln2 -> layer{layer_idx}_moe_output')
        edges.append(f'layer{layer_idx}_add1 -> layer{layer_idx}_moe_output')
    
    # Final output
    edges.append('layer15_moe_output -> output')
    
    # Build the complete DOT content
    dot_content = 'digraph Baseline_MoE_Final {\n    rankdir=TB;\n    splines=ortho;\n    node [fontname="Arial", fontsize=10];\n    \n'
    dot_content += '    ' + '\n    '.join(nodes) + '\n    \n'
    dot_content += '    ' + '\n    '.join(edges) + '\n}'
    
    return dot_content

def main():
    # Save proposed EP16 DAG
    proposed_dag = generate_proposed_ep16_dag()
    with open('../outputs/2025-11-24-14-31-20/proposed_ep16_complete.dot', 'w') as f:
        f.write(proposed_dag)
    
    # Save baseline DAG
    baseline_dag = generate_baseline_dag()
    with open('../outputs/2025-11-24-14-31-20/baseline_complete.dot', 'w') as f:
        f.write(baseline_dag)
    
    print("Complete DAGs generated successfully")
    print("- Proposed EP16: ../outputs/2025-11-24-14-31-20/proposed_ep16_complete.dot")
    print("- Baseline: ../outputs/2025-11-24-14-31-20/baseline_complete.dot")

if __name__ == "__main__":
    main()