#!/usr/bin/env python3
"""
Generate a complete model deployment DAG for the enhanced LLM deployment strategy.
This DAG represents the EP64_TP2 cross-node expert parallelism with advanced optimizations.
"""

import os
import subprocess

def generate_enhanced_deployment_dag():
    """Generate the complete DAG for the enhanced deployment method."""
    
    # Configuration from deployment method
    batch_size = 128
    seq_len = 10000
    hidden_size = 4096
    ffn_hidden_size = 16384
    num_heads = 32
    head_dim = 128
    num_layers = 16
    num_experts = 64
    tp_degree = 2
    ep_degree = 64
    total_gpus = 128
    
    # Node counter for unique IDs
    node_id = 0
    
    def next_id():
        nonlocal node_id
        node_id += 1
        return f"node_{node_id}"
    
    # Start building the DOT content
    dot_content = "digraph EnhancedLLMDeployment {\n"
    dot_content += "  rankdir=TB;\n"
    dot_content += "  node [shape=record, fontname=\"Helvetica\"];\n"
    dot_content += "  edge [fontname=\"Helvetica\"];\n"
    dot_content += "  \n"
    
    # Input node
    input_id = next_id()
    dot_content += f'  {input_id} [shape=box, label="Input\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]", style=filled, fillcolor=lightblue];\n'
    
    prev_output = input_id
    
    # Process each layer
    for layer_idx in range(num_layers):
        dot_content += f"  \n  // ===== Layer {layer_idx} =====\n"
        
        # RMSNorm for attention input
        rmsnorm_attn_id = next_id()
        dot_content += f'  {rmsnorm_attn_id} [shape=box, label="Layer{layer_idx}_RMSNorm_Attn\\nGPU: ALL\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]"];\n'
        dot_content += f'  {prev_output} -> {rmsnorm_attn_id};\n'
        
        # Multi-Head Attention with Tensor Parallelism (TP=2)
        # QKV projection - column parallel
        qkv_tp0_id = next_id()
        qkv_tp1_id = next_id()
        dot_content += f'  {qkv_tp0_id} [shape=box, label="Layer{layer_idx}_QKV_TP0\\nGPU: 0-63\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim//2}]"];\n'
        dot_content += f'  {qkv_tp1_id} [shape=box, label="Layer{layer_idx}_QKV_TP1\\nGPU: 64-127\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim//2}]"];\n'
        dot_content += f'  {rmsnorm_attn_id} -> {qkv_tp0_id};\n'
        dot_content += f'  {rmsnorm_attn_id} -> {qkv_tp1_id};\n'
        
        # Flash Attention computation
        flash_attn_tp0_id = next_id()
        flash_attn_tp1_id = next_id()
        dot_content += f'  {flash_attn_tp0_id} [shape=box, label="Layer{layer_idx}_FlashAttn_TP0\\nGPU: 0-63\\nINPUT: [batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim//2}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim//2}]"];\n'
        dot_content += f'  {flash_attn_tp1_id} [shape=box, label="Layer{layer_idx}_FlashAttn_TP1\\nGPU: 64-127\\nINPUT: [batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim//2}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim//2}]"];\n'
        dot_content += f'  {qkv_tp0_id} -> {flash_attn_tp0_id};\n'
        dot_content += f'  {qkv_tp1_id} -> {flash_attn_tp1_id};\n'
        
        # Attention output projection - row parallel with all-reduce
        attn_out_tp0_id = next_id()
        attn_out_tp1_id = next_id()
        dot_content += f'  {attn_out_tp0_id} [shape=box, label="Layer{layer_idx}_AttnOut_TP0\\nGPU: 0-63\\nINPUT: [batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim//2}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//2}]"];\n'
        dot_content += f'  {attn_out_tp1_id} [shape=box, label="Layer{layer_idx}_AttnOut_TP1\\nGPU: 64-127\\nINPUT: [batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim//2}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//2}]"];\n'
        dot_content += f'  {flash_attn_tp0_id} -> {attn_out_tp0_id};\n'
        dot_content += f'  {flash_attn_tp1_id} -> {attn_out_tp1_id};\n'
        
        # All-reduce communication for attention output
        attn_allreduce_id = next_id()
        dot_content += f'  {attn_allreduce_id} [shape=ellipse, label="Layer{layer_idx}_AttnAllReduce\\nGPU: ALL\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size//2}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]", style=dashed];\n'
        dot_content += f'  {attn_out_tp0_id} -> {attn_allreduce_id} [style=dashed];\n'
        dot_content += f'  {attn_out_tp1_id} -> {attn_allreduce_id} [style=dashed];\n'
        
        # Residual connection
        residual_attn_id = next_id()
        dot_content += f'  {residual_attn_id} [shape=parallelogram, label="Layer{layer_idx}_Residual_Attn\\nGPU: ALL\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]"];\n'
        dot_content += f'  {prev_output} -> {residual_attn_id};\n'
        dot_content += f'  {attn_allreduce_id} -> {residual_attn_id};\n'
        
        # RMSNorm for MLP input
        rmsnorm_mlp_id = next_id()
        dot_content += f'  {rmsnorm_mlp_id} [shape=box, label="Layer{layer_idx}_RMSNorm_MLP\\nGPU: ALL\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]"];\n'
        dot_content += f'  {residual_attn_id} -> {rmsnorm_mlp_id};\n'
        
        # Expert Parallelism with 64 experts (EP=64)
        # Gating/Router - determines which tokens go to which experts
        router_id = next_id()
        dot_content += f'  {router_id} [shape=parallelogram, label="Layer{layer_idx}_Router\\nGPU: ALL\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, experts={num_experts//4}]", style=dashed];\n'
        dot_content += f'  {rmsnorm_mlp_id} -> {router_id} [style=dashed];\n'
        
        # All-to-all communication for expert dispatch
        all2all_dispatch_id = next_id()
        dot_content += f'  {all2all_dispatch_id} [shape=ellipse, label="Layer{layer_idx}_All2All_Dispatch\\nGPU: ALL\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]", style=dashed];\n'
        dot_content += f'  {router_id} -> {all2all_dispatch_id} [style=dashed];\n'
        
        # Expert processing (each expert handles 1/64 of tokens)
        # Each expert is split across 2 GPUs with TP=2
        expert_outputs = []
        for expert_idx in range(num_experts//4):  # Show 16 experts (1/4 of total)
            expert_base_id = next_id()
            expert_tp0_id = next_id()
            expert_tp1_id = next_id()
            
            # Expert base (routing aggregation)
            dot_content += f'  {expert_base_id} [shape=parallelogram, label="Layer{layer_idx}_Expert{expert_idx}_Base\\nGPU: {expert_idx*2}-{expert_idx*2+1}\\nINPUT: [batch={batch_size//16}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size//16}, seq={seq_len}, hidden={hidden_size}]"];\n'
            
            # Expert MLP TP0 (first linear - column parallel)
            dot_content += f'  {expert_tp0_id} [shape=box, label="Layer{layer_idx}_Expert{expert_idx}_MLP_TP0\\nGPU: {expert_idx*2}\\nINPUT: [batch={batch_size//16}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size//16}, seq={seq_len}, ffn={ffn_hidden_size//2}]"];\n'
            
            # Expert MLP TP1 (first linear - column parallel)
            dot_content += f'  {expert_tp1_id} [shape=box, label="Layer{layer_idx}_Expert{expert_idx}_MLP_TP1\\nGPU: {expert_idx*2+1}\\nINPUT: [batch={batch_size//16}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size//16}, seq={seq_len}, ffn={ffn_hidden_size//2}]"];\n'
            
            # GELU activation
            gelu_tp0_id = next_id()
            gelu_tp1_id = next_id()
            dot_content += f'  {gelu_tp0_id} [shape=box, label="Layer{layer_idx}_Expert{expert_idx}_GELU_TP0\\nGPU: {expert_idx*2}\\nINPUT: [batch={batch_size//16}, seq={seq_len}, ffn={ffn_hidden_size//2}]\\nOUTPUT: [batch={batch_size//16}, seq={seq_len}, ffn={ffn_hidden_size//2}]"];\n'
            dot_content += f'  {gelu_tp1_id} [shape=box, label="Layer{layer_idx}_Expert{expert_idx}_GELU_TP1\\nGPU: {expert_idx*2+1}\\nINPUT: [batch={batch_size//16}, seq={seq_len}, ffn={ffn_hidden_size//2}]\\nOUTPUT: [batch={batch_size//16}, seq={seq_len}, ffn={ffn_hidden_size//2}]"];\n'
            
            # Expert output projection - row parallel
            expert_out_tp0_id = next_id()
            expert_out_tp1_id = next_id()
            dot_content += f'  {expert_out_tp0_id} [shape=box, label="Layer{layer_idx}_Expert{expert_idx}_Out_TP0\\nGPU: {expert_idx*2}\\nINPUT: [batch={batch_size//16}, seq={seq_len}, ffn={ffn_hidden_size//2}]\\nOUTPUT: [batch={batch_size//16}, seq={seq_len}, hidden={hidden_size//2}]"];\n'
            dot_content += f'  {expert_out_tp1_id} [shape=box, label="Layer{layer_idx}_Expert{expert_idx}_Out_TP1\\nGPU: {expert_idx*2+1}\\nINPUT: [batch={batch_size//16}, seq={seq_len}, ffn={ffn_hidden_size//2}]\\nOUTPUT: [batch={batch_size//16}, seq={seq_len}, hidden={hidden_size//2}]"];\n'
            
            # Expert all-reduce
            expert_allreduce_id = next_id()
            dot_content += f'  {expert_allreduce_id} [shape=ellipse, label="Layer{layer_idx}_Expert{expert_idx}_AllReduce\\nGPU: {expert_idx*2}-{expert_idx*2+1}\\nINPUT: [batch={batch_size//16}, seq={seq_len}, hidden={hidden_size//2}]\\nOUTPUT: [batch={batch_size//16}, seq={seq_len}, hidden={hidden_size}]", style=dashed];\n'
            
            # Connect expert chain
            dot_content += f'  {all2all_dispatch_id} -> {expert_base_id};\n'
            dot_content += f'  {expert_base_id} -> {expert_tp0_id};\n'
            dot_content += f'  {expert_base_id} -> {expert_tp1_id};\n'
            dot_content += f'  {expert_tp0_id} -> {gelu_tp0_id};\n'
            dot_content += f'  {expert_tp1_id} -> {gelu_tp1_id};\n'
            dot_content += f'  {gelu_tp0_id} -> {expert_out_tp0_id};\n'
            dot_content += f'  {gelu_tp1_id} -> {expert_out_tp1_id};\n'
            dot_content += f'  {expert_out_tp0_id} -> {expert_allreduce_id} [style=dashed];\n'
            dot_content += f'  {expert_out_tp1_id} -> {expert_allreduce_id} [style=dashed];\n'
            
            expert_outputs.append(expert_allreduce_id)
        
        # All-to-all communication for expert combine
        all2all_combine_id = next_id()
        dot_content += f'  {all2all_combine_id} [shape=ellipse, label="Layer{layer_idx}_All2All_Combine\\nGPU: ALL\\nINPUT: [batch={batch_size//16}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]", style=dashed];\n'
        for expert_out in expert_outputs:
            dot_content += f'  {expert_out} -> {all2all_combine_id} [style=dashed];\n'
        
        # MLP output projection (weighted sum of expert outputs)
        mlp_out_id = next_id()
        dot_content += f'  {mlp_out_id} [shape=box, label="Layer{layer_idx}_MLP_Out\\nGPU: ALL\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]"];\n'
        dot_content += f'  {all2all_combine_id} -> {mlp_out_id};\n'
        
        # Residual connection for MLP
        residual_mlp_id = next_id()
        dot_content += f'  {residual_mlp_id} [shape=parallelogram, label="Layer{layer_idx}_Residual_MLP\\nGPU: ALL\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]"];\n'
        dot_content += f'  {residual_attn_id} -> {residual_mlp_id};\n'
        dot_content += f'  {mlp_out_id} -> {residual_mlp_id};\n'
        
        prev_output = residual_mlp_id
    
    # Final RMSNorm
    final_norm_id = next_id()
    dot_content += f'  {final_norm_id} [shape=box, label="Final_RMSNorm\\nGPU: ALL\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]"];\n'
    dot_content += f'  {prev_output} -> {final_norm_id};\n'
    
    # Output node
    output_id = next_id()
    dot_content += f'  {output_id} [shape=box, label="Output\\nINPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOUTPUT: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]", style=filled, fillcolor=lightgreen];\n'
    dot_content += f'  {final_norm_id} -> {output_id};\n'
    
    dot_content += "}\n"
    
    return dot_content

def generate_summary_dag():
    """Generate a summary DAG showing high-level structure."""
    
    batch_size = 128
    seq_len = 10000
    hidden_size = 4096
    num_layers = 16
    
    dot_content = "digraph EnhancedLLMDeploymentSummary {\n"
    dot_content += "  rankdir=TB;\n"
    dot_content += "  node [shape=record, fontname=\"Helvetica\"];\n"
    dot_content += "  edge [fontname=\"Helvetica\"];\n"
    dot_content += "  \n"
    
    # Input
    dot_content += '  Input [shape=box, label="Input\\n[batch=128, seq=10000, hidden=4096]", style=filled, fillcolor=lightblue];\n'
    
    # Layer blocks
    prev = "Input"
    for i in range(num_layers):
        layer_id = f"Layer{i}"
        dot_content += f'  {layer_id} [shape=box, label="Layer {i}\\nRMSNorm + MHA + Residual\\nRMSNorm + MoE + Residual\\nTP=2, EP=64\\n[batch=128, seq=10000, hidden=4096]"];\n'
        dot_content += f'  {prev} -> {layer_id};\n'
        prev = layer_id
    
    # Final norm and output
    dot_content += '  FinalNorm [shape=box, label="Final RMSNorm\\n[batch=128, seq=10000, hidden=4096]"];\n'
    dot_content += '  Output [shape=box, label="Output\\n[batch=128, seq=10000, hidden=4096]", style=filled, fillcolor=lightgreen];\n'
    dot_content += f'  {prev} -> FinalNorm;\n'
    dot_content += '  FinalNorm -> Output;\n'
    
    dot_content += "}\n"
    
    return dot_content

if __name__ == "__main__":
    # Generate comprehensive DAG
    comprehensive_dot = generate_enhanced_deployment_dag()
    
    # Generate summary DAG
    summary_dot = generate_summary_dag()
    
    # Save DOT files
    output_dir = "../outputs/2025-12-04-11-44-06"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comprehensive DAG
    with open(f"{output_dir}/enhanced_llm_deployment_comprehensive.dot", "w") as f:
        f.write(comprehensive_dot)
    
    # Save summary DAG
    with open(f"{output_dir}/enhanced_llm_deployment_summary.dot", "w") as f:
        f.write(summary_dot)
    
    # Generate SVG images using dot command
    try:
        subprocess.run(["dot", "-Tsvg", f"{output_dir}/enhanced_llm_deployment_comprehensive.dot", "-o", f"{output_dir}/enhanced_llm_deployment_comprehensive.svg"], check=True)
        subprocess.run(["dot", "-Tsvg", f"{output_dir}/enhanced_llm_deployment_summary.dot", "-o", f"{output_dir}/enhanced_llm_deployment_summary.svg"], check=True)
        print(f"Generated DAG files and SVG images in {output_dir}:")
        print(f"- enhanced_llm_deployment_comprehensive.dot")
        print(f"- enhanced_llm_deployment_comprehensive.svg")
        print(f"- enhanced_llm_deployment_summary.dot")
        print(f"- enhanced_llm_deployment_summary.svg")
    except subprocess.CalledProcessError as e:
        print(f"Error generating SVG images: {e}")
        print(f"DOT files generated successfully in {output_dir}")
    except FileNotFoundError:
        print("Graphviz 'dot' command not found. SVG images not generated.")
        print(f"DOT files generated successfully in {output_dir}")