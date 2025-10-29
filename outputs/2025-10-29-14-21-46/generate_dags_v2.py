#!/usr/bin/env python3
"""
Generator for transformer model parallel deployment DAGs
"""

import os
from pathlib import Path

class DAGGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_attention_dag(self, model_name, config, device_id, tensor_group_size):
        """Generate attention layer DAG with tensor parallelism"""
        hidden_dim = config['hidden_dimension']
        heads = config.get('attention_heads', hidden_dim // 128)
        per_device_dim = hidden_dim // tensor_group_size
        
        dag_lines = [
            f'digraph {model_name}_attention_layer_{device_id} {{',
            '    rankdir=TB;',
            '    node [shape=rectangle, style=filled];',
            '',
            '    // Input handling',
            f'    input [label="Input\\nInput: [batch_size=1, seq_len=2048, hidden={hidden_dim}]\\nGPU: {device_id}", shape=parallelogram, fillcolor=lightblue];',
            '',
            '    // LayerNorm (replicated)',
            f'    layernorm1 [label="LayerNorm\\nInput: [1,2048,{hidden_dim}]\\nOutput: [1,2048,{hidden_dim}]\\nGPU: {device_id}", fillcolor=yellow];',
            '',
            '    // QKV Projection - Column Parallel',
            f'    qkv_split [label="Split QKV Input\\nInput: [1,2048,{hidden_dim}]\\nOutput: [1,2048,{per_device_dim}]\\nGPU: {device_id}", shape=ellipse, fillcolor=lightgreen];',
            '',
            f'    q_proj [label="Q Projection\\nInput: [1,2048,{hidden_dim}]\\nOutput: [1,2048,{per_device_dim}]\\nGPU: {device_id}", fillcolor=orange];',
            f'    k_proj [label="K Projection\\nInput: [1,2048,{hidden_dim}]\\nOutput: [1,2048,{per_device_dim}]\\nGPU: {device_id}", fillcolor=orange];',
            f'    v_proj [label="V Projection\\nInput: [1,2048,{hidden_dim}]\\nOutput: [1,2048,{per_device_dim}]\\nGPU: {device_id}", fillcolor=orange];',
            '',
            '    // Attention computation',
            f'    qk_matmul [label="QK^T MatMul\\nInput: [1,2048,{per_device_dim}], [1,2048,{per_device_dim}]\\nOutput: [1,2048,2048]\\nGPU: {device_id}", fillcolor=pink];',
            '    scale [label="Scale\\nInput: [1,2048,2048]\\nOutput: [1,2048,2048]\\nGPU: {device_id}", fillcolor=lightcyan];',
            '    softmax [label="Softmax\\nInput: [1,2048,2048]\\nOutput: [1,2048,2048]\\nGPU: {device_id}", fillcolor=lightcyan];',
            '    dropout [label="Attention Dropout\\nInput: [1,2048,2048]\\nOutput: [1,2048,2048]\\nGPU: {device_id}", fillcolor=lightcyan];',
            '',
            '    // Output projection',
            f'    attn_output [label="Attn Output MatMul\\nInput: [1,2048,2048], [1,2048,{per_device_dim}]\\nOutput: [1,2048,{per_device_dim}]\\nGPU: {device_id}", fillcolor=pink];',
            '',
            '    // All-reduce communication',
            f'    allreduce [label="All-Reduce\\nInput: [1,2048,{per_device_dim}]\\nOutput: [1,2048,{hidden_dim}]\\nAll GPUs in tensor group", shape=hexagon, fillcolor=red];',
            '',
            '    // Residual connection',
            f'    residual1 [label="Residual Add\\nInput: [1,2048,{hidden_dim}], [1,2048,{hidden_dim}]\\nOutput: [1,2048,{hidden_dim}]\\nGPU: {device_id}", fillcolor=grey];',
            '',
            '    // Connections',
            '    input -> layernorm1;',
            '    layernorm1 -> qkv_split;',
            '    qkv_split -> q_proj;',
            '    qkv_split -> k_proj;',
            '    qkv_split -> v_proj;',
            '    q_proj -> qk_matmul;',
            '    k_proj -> qk_matmul;',
            '    qk_matmul -> scale;',
            '    scale -> softmax;',
            '    softmax -> dropout;',
            '    dropout -> attn_output;',
            '    v_proj -> attn_output;',
            '    attn_output -> allreduce;',
            '    allreduce -> residual1;',
            f'    input -> residual1 [style=dashed];',
            '}'
        ]
        
        return '\\n'.join(dag_lines)
    
    def generate_mlp_dag(self, model_name, config, device_id, tensor_group_size):
        """Generate MLP layer DAG with tensor parallelism"""
        hidden_dim = config['hidden_dimension']
        ffn_hidden = hidden_dim * 4
        per_device_input = hidden_dim
        per_device_hidden = ffn_hidden // tensor_group_size
        
        dag_lines = [
            f'digraph {model_name}_mlp_layer_{device_id} {{',
            '    rankdir=TB;',
            '    node [shape=rectangle, style=filled];',
            '',
            '    // Input from attention output',
            f'    input [label="MLP Input\\nInput: [batch_size=1, seq_len=2048, hidden={hidden_dim}]\\nGPU: {device_id}", shape=parallelogram, fillcolor=lightblue];',
            '',
            '    // LayerNorm',
            f'    layernorm2 [label="LayerNorm\\nInput: [1,2048,{hidden_dim}]\\nOutput: [1,2048,{hidden_dim}]\\nGPU: {device_id}", fillcolor=yellow];',
            '',
            '    // FC1 - Column Parallel',
            f'    fc1 [label="FC1 Linear\\nInput: [1,2048,{hidden_dim}]\\nOutput: [1,2048,{per_device_hidden}]\\nGPU: {device_id}", fillcolor=orange];',
            '',
            '    // Activation',
            f'    gelu [label="GELU Activation\\nInput: [1,2048,{per_device_hidden}]\\nOutput: [1,2048,{per_device_hidden}]\\nGPU: {device_id}", fillcolor=lightgreen];',
            '',
            '    // FC2 - Row Parallel',
            f'    fc2 [label="FC2 Linear\\nInput: [1,2048,{per_device_hidden}]\\nOutput: [1,2048,{hidden_dim//tensor_group_size}]\\nGPU: {device_id}", fillcolor=orange];',
            '',
            '    // All-reduce',
            f'    allreduce2 [label="All-Reduce\\nInput: [1,2048,{hidden_dim//tensor_group_size}]\\nOutput: [1,2048,{hidden_dim}]\\nAll GPUs in tensor group", shape=hexagon, fillcolor=red];',
            '',
            '    // Residual connection',
            f'    residual2 [label="Residual Add\\nInput: [1,2048,{hidden_dim}], [1,2048,{hidden_dim}]\\nOutput: [1,2048,{hidden_dim}]\\nGPU: {device_id}", fillcolor=grey];',
            '',
            '    // Connections',
            '    input -> layernorm2;',
            '    layernorm2 -> fc1;',
            '    fc1 -> gelu;',
            '    gelu -> fc2;',
            '    fc2 -> allreduce2;',
            '    allreduce2 -> residual2;',
            f'    input -> residual2 [style=dashed];',
            '}'
        ]
        
        return '\\n'.join(dag_lines)
    
    def generate_baseline_dag(self, model_name, config):
        """Generate baseline sequential DAG without parallelism"""
        dag_lines = [
            f'digraph {model_name}_baseline {{',
            '    rankdir=TB;',
            '    node [shape=rectangle, style=filled];',
            '',
            '    // Model input',
            '    input [label="Model Input\\nInput: [batch_size=1, seq_len=2048]\\nGPU: 0", shape=parallelogram, fillcolor=lightblue];',
            '',
            '    // Embedding',
            f'    token_embed [label="Token Embedding\\nInput: [1,2048]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=yellow];',
            f'    pos_embed [label="Position Embedding\\nInput: [1,2048]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=yellow];',
            f'    embed_add [label="Add Embeddings\\nInput: [1,2048,{config["hidden_dimension"]}], [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=grey];',
            '',
            '    // Transformer layers (sequential)'
        ]
        
        previous_layer = "embed_add"
        
        for layer in range(config['layers']):
            dag_lines.extend([
                '',
                f'    // Layer {layer}',
                f'    layernorm1_{layer} [label="Layer {layer}\\nLayerNorm 1\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=yellow];',
                f'    self_attn_{layer} [label="Layer {layer}\\nSelf-Attention\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=orange];',
                f'    residual1_{layer} [label="Layer {layer}\\nResidual Add 1\\nInput: [1,2048,{config["hidden_dimension"]}], [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=grey];',
                f'    layernorm2_{layer} [label="Layer {layer}\\nLayerNorm 2\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=yellow];',
                f'    mlp_{layer} [label="Layer {layer}\\nMLP\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=lightgreen];',
                f'    residual2_{layer} [label="Layer {layer}\\nResidual Add 2\\nInput: [1,2048,{config["hidden_dimension"]}], [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=grey];',
                '',
                f'    {previous_layer} -> layernorm1_{layer};',
                f'    layernorm1_{layer} -> self_attn_{layer};',
                f'    self_attn_{layer} -> residual1_{layer};',
                f'    {previous_layer} -> residual1_{layer} [style=dashed];',
                f'    residual1_{layer} -> layernorm2_{layer};',
                f'    layernorm2_{layer} -> mlp_{layer};',
                f'    mlp_{layer} -> residual2_{layer};',
                f'    residual1_{layer} -> residual2_{layer} [style=dashed];'
            ])
            
            previous_layer = f"residual2_{layer}"
        
        dag_lines.extend([
            '',
            '    // Final processing',
            f'    final_layernorm [label="Final LayerNorm\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=yellow];',
            f'    {previous_layer} -> final_layernorm;',
            '',
            f'    lm_head [label="Language Model Head\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["vocabulary_size"]}]\\nGPU: 0", fillcolor=orange];',
            '    final_layernorm -> lm_head;',
            '}'
        ])
        
        return '\\n'.join(dag_lines)

def main():
    output_dir = "../outputs/2025-10-29-14-21-46"
    generator = DAGGenerator(output_dir)
    
    # Model configurations from deployment_config.json
    configs = {
        "megatron_8_3b": {
            "hidden_dimension": 4096,
            "layers": 24,
            "vocabulary_size": 51200,
            "attention_heads": 32
        },
        "megatron_530b": {
            "hidden_dimension": 20480,
            "layers": 105,
            "vocabulary_size": 51200,
            "attention_heads": 128
        },
        "megatron_1t": {
            "hidden_dimension": 25600,
            "layers": 128,
            "vocabulary_size": 51200,
            "attention_heads": 160
        },
        "gopher_280b": {
            "hidden_dimension": 16384,
            "layers": 80,
            "vocabulary_size": 32000,
            "attention_heads": 128
        },
        "palm_540b": {
            "hidden_dimension": 18432,
            "layers": 118,
            "vocabulary_size": 256000,
            "attention_heads": 48
        },
        "gpt3_175b": {
            "hidden_dimension": 12288,
            "layers": 96,
            "vocabulary_size": 50000,
            "attention_heads": 96
        }
    }
    
    # Generate DAGs for each model
    for model_name, config in configs.items():
        # Baseline DAG (no parallelism)
        baseline_dag = generator.generate_baseline_dag(f"{model_name}_baseline", config)
        with open(f"{output_dir}/{model_name}_baseline.dot", "w") as f:
            f.write(baseline_dag)
        
        # Optimized DAGs for tensor parallel devices
        if model_name == "palm_540b":
            tensor_group_size = 12
        else:
            tensor_group_size = 8
            
        for device in range(tensor_group_size):
            attention_dag = generator.generate_attention_dag(f"{model_name}_optimized", config, device, tensor_group_size)
            with open(f"{output_dir}/{model_name}_attention_device_{device}.dot", "w") as f:
                f.write(attention_dag)
            
            mlp_dag = generator.generate_mlp_dag(f"{model_name}_optimized", config, device, tensor_group_size)
            with open(f"{output_dir}/{model_name}_mlp_device_{device}.dot", "w") as f:
                f.write(mlp_dag)

if __name__ == "__main__":
    main()