#!/usr/bin/env python3

import os

def generate_fixed_moe_dag():
    """Generate corrected MoE DAG with proper expert node connections"""
    
    dot_content = """// Qwen3-235B MoE Parallel Strategy DAG - Fixed Version
digraph {
    rankdir=TB
    size="20,30"
    node [fontname=Arial fontsize=10]
    
    // Define node styles
    node [fillcolor=lightblue shape=ellipse style=filled]  // Communication
    node [fillcolor=lightgreen shape=rectangle style=filled]  // Computation  
    node [fillcolor=lightyellow shape=parallelogram style=filled]  // Routing/Aggregation
    
    // Input node
    input [label="Input\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]" fillcolor=lightcoral shape=rectangle]
    
    // Token Embedding on all GPUs
"""
    
    # Token embedding nodes
    for gpu_id in range(8):
        dot_content += f'''    embed_gpu{gpu_id} [label="GPU{gpu_id}: Token Embedding\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]" fillcolor=lightgreen shape=rectangle]
    input -> embed_gpu{gpu_id}
'''

    # Attention operations for each GPU
    for gpu_id in range(8):
        dot_content += f'''
    // GPU{gpu_id} Attention operations
    attn_qkv_gpu{gpu_id}_layer1 [label="GPU{gpu_id}: Layer1 Attention QKV Proj\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=8, d_k=64]" fillcolor=lightgreen shape=rectangle]
    comm_qkv_gpu{gpu_id}_layer1 [label="GPU{gpu_id}: All-Reduce QKV\\nInput: [batch_size=128, seq_len=2048, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=8, d_k=64]" fillcolor=lightblue shape=ellipse]
    attn_comp_gpu{gpu_id}_layer1 [label="GPU{gpu_id}: Layer1 Attention Compute\\nInput: [batch_size=128, seq_len=2048, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=8, d_k=64]" fillcolor=lightgreen shape=rectangle]
    comm_attn_out_gpu{gpu_id}_layer1 [label="GPU{gpu_id}: All-Reduce Attention Output\\nInput: [batch_size=128, seq_len=2048, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]" fillcolor=lightblue shape=ellipse]
    
    embed_gpu{gpu_id} -> attn_qkv_gpu{gpu_id}_layer1
    attn_qkv_gpu{gpu_id}_layer1 -> comm_qkv_gpu{gpu_id}_layer1
    comm_qkv_gpu{gpu_id}_layer1 -> attn_comp_gpu{gpu_id}_layer1
    attn_comp_gpu{gpu_id}_layer1 -> comm_attn_out_gpu{gpu_id}_layer1
'''

    # Gate routing nodes
    for gpu_id in range(8):
        dot_content += f'''
    gate_gpu{gpu_id}_layer1 [label="GPU{gpu_id}: Layer1 Gate Routing\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, experts=8]" fillcolor=lightyellow shape=parallelogram]
    comm_attn_out_gpu{gpu_id}_layer1 -> gate_gpu{gpu_id}_layer1
'''

    # Expert nodes (4 experts per GPU for demonstration)
    for gpu_id in range(8):
        for expert_id in range(4):
            dot_content += f'''    expert_gpu{gpu_id}_exp{expert_id}_layer1 [label="GPU{gpu_id}: Expert {expert_id}\\nInput: [batch_size=?, seq_len=?, hidden=4096]\\nOutput: [batch_size=?, seq_len=?, hidden=1536]" fillcolor=lightgreen shape=rectangle]
'''

    # Communication nodes for expert send/receive
    for gpu_id in range(8):
        dot_content += f'''
    comm_expert_send_gpu{gpu_id}_layer1 [label="GPU{gpu_id}: Send Tokens to Experts\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=?, seq_len=?, hidden=4096]" fillcolor=lightblue shape=ellipse]
    comm_expert_recv_gpu{gpu_id}_layer1 [label="GPU{gpu_id}: Receive from Experts\\nInput: [batch_size=?, seq_len=?, hidden=1536]\\nOutput: [batch_size=128, seq_len=2048, hidden=1536]" fillcolor=lightblue shape=ellipse]
    
    gate_gpu{gpu_id}_layer1 -> comm_expert_send_gpu{gpu_id}_layer1
'''

    # Connect experts to receive nodes and add gate routing connections
    for gpu_id in range(8):
        # Connect send to experts
        for expert_id in range(4):
            dot_content += f'    comm_expert_send_gpu{gpu_id}_layer1 -> expert_gpu{gpu_id}_exp{expert_id}_layer1\n'
        
        # Connect experts to receive
        for expert_id in range(4):
            dot_content += f'    expert_gpu{gpu_id}_exp{expert_id}_layer1 -> comm_expert_recv_gpu{gpu_id}_layer1\n'

    # Gate routing connections (dashed lines)
    for src_gpu in range(8):
        for dst_gpu in range(8):
            dot_content += f'    gate_gpu{src_gpu}_layer1 -> expert_gpu{dst_gpu}_exp0_layer1 [label="GPU{src_gpu} selects experts on GPU{dst_gpu}" color=red style=dashed]\n'

    # Expert aggregation and MLP operations
    for gpu_id in range(8):
        dot_content += f'''
    expert_agg_gpu{gpu_id}_layer1 [label="GPU{gpu_id}: Expert Aggregation\\nInput: [batch_size=128, seq_len=2048, experts=8, hidden=1536]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]" fillcolor=lightyellow shape=parallelogram]
    mlp_gpu{gpu_id}_layer1 [label="GPU{gpu_id}: MLP Compute\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]" fillcolor=lightgreen shape=rectangle]
    mlp_agg_gpu{gpu_id}_layer1 [label="GPU{gpu_id}: MLP All-Reduce\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]" fillcolor=lightblue shape=ellipse]
    
    comm_expert_recv_gpu{gpu_id}_layer1 -> expert_agg_gpu{gpu_id}_layer1
    expert_agg_gpu{gpu_id}_layer1 -> mlp_gpu{gpu_id}_layer1
    mlp_gpu{gpu_id}_layer1 -> mlp_agg_gpu{gpu_id}_layer1
'''

    # Output node
    dot_content += '''
    output [label="Output\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]" fillcolor=lightcoral shape=rectangle]
'''

    # Connect all MLP outputs to final output
    for gpu_id in range(8):
        dot_content += f'    mlp_agg_gpu{gpu_id}_layer1 -> output\n'

    dot_content += '}\n'
    
    return dot_content

def main():
    # Generate the fixed DAG
    dot_content = generate_fixed_moe_dag()
    
    # Save to file
    output_path = './outputs/2026-01-04-09-49-07/moe_parallel_dag_fixed.dot'
    with open(output_path, 'w') as f:
        f.write(dot_content)
    
    print(f"Fixed DAG generated and saved to {output_path}")
    
    # Generate SVG image
    svg_path = './outputs/2026-01-04-09-49-07/moe_parallel_dag_fixed.svg'
    os.system(f'dot -Tsvg {output_path} -o {svg_path}')
    print(f"SVG image saved to {svg_path}")

if __name__ == "__main__":
    main()