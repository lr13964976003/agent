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
        dag = f"""
digraph {model_name}_attention_layer_{device_id} {{
    rankdir=TB;
    node [shape=rectangle, style=filled];
    
    // Input handling
    input [label="Input\\nInput: [batch_size=1, seq_len=2048, hidden={config['hidden_dimension']}]\\nGPU: {device_id}", shape=parallelogram, fillcolor=lightblue];
    
    // LayerNorm (replicated)
    layernorm1 [label="LayerNorm\\nInput: [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: {device_id}", fillcolor=yellow];
    
    // QKV Projection - Column Parallel
    qkv_split [label="Split QKV Input\\nInput: [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nGPU: {device_id}", shape=ellipse, fillcolor=lightgreen];
    
    q_proj [label="Q Projection\\nInput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nOutput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nGPU: {device_id}", fillcolor=orange];
    k_proj [label="K Projection\\nInput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nOutput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nGPU: {device_id}", fillcolor=orange];
    v_proj [label="V Projection\\nInput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nOutput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nGPU: {device_id}", fillcolor=orange];
    
    // Attention computation
    qk_matmul [label="QK^T MatMul\\nInput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nOutput: [1,2048,2048]\\nGPU: {device_id}", fillcolor=pink];
    scale [label="Scale\\nInput: [1,2048,2048]\\nOutput: [1,2048,2048]\\nGPU: {device_id}", fillcolor=lightcyan];
    softmax [label="Softmax\\nInput: [1,2048,2048]\\nOutput: [1,2048,2048]\\nGPU: {device_id}", fillcolor=lightcyan];
    attention [label="Attention Dropout\\nInput: [1,2048,2048]\\nOutput: [1,2048,2048]\\nGPU: {device_id}", fillcolor=lightcyan];
    
    // Output projection
    attn_output [label="Attn Output MatMul\\nInput: [1,2048,2048], [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nOutput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nGPU: {device_id}", fillcolor=pink];
    
    // All-reduce communication
    allreduce [label="All-Reduce\\nInput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nAll GPUs in tensor group", shape=hexagon, fillcolor=red];
    
    // Residual connection
    residual1 [label="Residual Add\\nInput: [1,2048,{config['hidden_dimension']}], [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: {device_id}", fillcolor=grey];
    
    // Connections
    input -> layernorm1;
    layernorm1 -> qkv_split;
    qkv_split -> q_proj;
    qkv_split -> k_proj;
    qkv_split -> v_proj;
    q_proj -> qk_matmul;
    k_proj -> qk_matmul;
    qk_matmul -> scale;
    scale -> softmax;
    softmax -> attention;
    attention -> attn_output;
    v_proj -> attn_output;
    attn_output -> allreduce;
    allreduce -> residual1;
    input -> residual1 [style=dashed];
}}
"""
        return dag
    
    def generate_mlp_dag(self, model_name, config, device_id, tensor_group_size):
        """Generate MLP layer DAG with tensor parallelism"""
        ffn_hidden = config['hidden_dimension'] * 4
        dag = f"""
digraph {model_name}_mlp_layer_{device_id} {{
    rankdir=TB;
    node [shape=rectangle, style=filled];
    
    // Input from attention output
    input [label="MLP Input\\nInput: [batch_size=1, seq_len=2048, hidden={config['hidden_dimension']}]\\nGPU: {device_id}", shape=parallelogram, fillcolor=lightblue];
    
    // LayerNorm
    layernorm2 [label="LayerNorm\\nInput: [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: {device_id}", fillcolor=yellow];
    
    // FC1 - Column Parallel
    fc1_split [label="Split FC1 Input\\nInput: [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: {device_id}", shape=ellipse, fillcolor=lightgreen];
    fc1 [label="FC1 Linear\\nInput: [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{ffn_hidden//tensor_group_size}]\\nGPU: {device_id}", fillcolor=orange];
    
    // Activation
    gelu [label="GELU Activation\\nInput: [1,2048,{ffn_hidden//tensor_group_size}]\\nOutput: [1,2048,{ffn_hidden//tensor_group_size}]\\nGPU: {device_id}", fillcolor=lightgreen];
    
    // FC2 - Row Parallel
    fc2 [label="FC2 Linear\\nInput: [1,2048,{ffn_hidden//tensor_group_size}]\\nOutput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nGPU: {device_id}", fillcolor=orange];
    
    // All-reduce
    allreduce2 [label="All-Reduce\\nInput: [1,2048,{config['hidden_dimension']//tensor_group_size}]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nAll GPUs in tensor group", shape=hexagon, fillcolor=red];
    
    // Residual connection
    residual2 [label="Residual Add\\nInput: [1,2048,{config['hidden_dimension']}], [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: {device_id}", fillcolor=grey];
    
    // Connections
    input -> layernorm2;
    layernorm2 -> fc1_split;
    fc1_split -> fc1;
    fc1 -> gelu;
    gelu -> fc2;
    fc2 -> allreduce2;
    allreduce2 -> residual2;
    input -> residual2 [style=dashed];
}}
"""
        return dag
    
    def generate_full_model_dag(self, model_name, config, total_devices, tensor_parallel, pipeline_parallel):
        """Generate complete model DAG with pipeline and tensor parallelism"""
        devices_per_pipeline = total_devices // pipeline_parallel
        tensor_devices = devices_per_pipeline // tensor_parallel
        
        dag = f"""
digraph {model_name}_full_model {{
    rankdir=LR;
    node [shape=rectangle, style=filled];
    
    // Model input
    model_input [label="Model Input\\nInput: [batch_size=1, seq_len=2048]\\nGPU: 0", shape=parallelogram, fillcolor=lightblue];
    
    // Embedding layer
    embedding [label="Token Embedding\\nInput: [1,2048]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: 0", fillcolor=yellow];
    
    // Position embedding
    pos_embedding [label="Position Embedding\\nInput: [1,2048]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: 0", fillcolor=yellow];
    
    // Embedding add
    embed_add [label="Add Embeddings\\nInput: [1,2048,{config['hidden_dimension']}], [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: 0", fillcolor=grey];
    
    // Pipeline stages
    pipeline_stages = []
    for stage in range(pipeline_parallel):
        stage_start = stage * devices_per_pipeline
        stage_label = f"Pipeline Stage {stage}\\nLayers {stage*config['layers']//pipeline_parallel}-{(stage+1)*config['layers']//pipeline_parallel-1}\\nGPUs: {stage_start}-{stage_start+devices_per_pipeline-1}"
        pipeline_stages.append(f'stage_{stage} [label="{stage_label}", shape=box, fillcolor=lightcyan];')
    
    dag += "\\n".join(pipeline_stages)
    
    // Communication between pipeline stages
    comm_edges = []
    for i in range(pipeline_parallel - 1):
        comm_edges.append(f'stage_{i} -> stage_{i+1} [label="Send Activations\\n[1,2048,{config[\"hidden_dimension\"]}]\\nGPU {i*devices_per_pipeline+devices_per_pipeline-1} -> GPU {(i+1)*devices_per_pipeline}", style=dashed];')
    
    dag += "\\n" + "\\n".join(comm_edges)
    
    // Final layers
    final_layernorm [label="Final LayerNorm\\nInput: [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: {total_devices-1}", fillcolor=yellow];
    
    lm_head [label="LM Head\\nInput: [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{config['vocabulary_size']}]\\nGPU: {total_devices-1}", fillcolor=orange];
    
    // Connections
    dag += f"""
    
    model_input -> embedding;
    model_input -> pos_embedding;
    embedding -> embed_add;
    pos_embedding -> embed_add;
    embed_add -> stage_0;
    stage_{pipeline_parallel-1} -> final_layernorm;
    final_layernorm -> lm_head;
}}
"""
        return dag
    
    def generate_baseline_dag(self, model_name, config):
        """Generate baseline sequential DAG without parallelism"""
        dag = f"""
digraph {model_name}_baseline {{
    rankdir=TB;
    node [shape=rectangle, style=filled];
    
    // Model input
    input [label="Model Input\\nInput: [batch_size=1, seq_len=2048]\\nGPU: 0", shape=parallelogram, fillcolor=lightblue];
    
    // Embedding
    token_embed [label="Token Embedding\\nInput: [1,2048]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: 0", fillcolor=yellow];
    pos_embed [label="Position Embedding\\nInput: [1,2048]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: 0", fillcolor=yellow];
    embed_add [label="Add Embeddings\\nInput: [1,2048,{config['hidden_dimension']}], [1,2048,{config['hidden_dimension']}]\\nOutput: [1,2048,{config['hidden_dimension']}]\\nGPU: 0", fillcolor=grey];
    
    // Transformer layers (sequential)
    previous_layer = "embed_add"
    
    for layer in range(config['layers']):
        layer_label = f"Layer {layer}"
        
        # LayerNorm 1
        dag += f'layernorm1_{layer} [label="{layer_label}\\nLayerNorm 1\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=yellow];\\n'
        dag += f'{previous_layer} -> layernorm1_{layer};\\n'
        
        # Self-attention
        dag += f'self_attn_{layer} [label="{layer_label}\\nSelf-Attention\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=orange];\\n'
        dag += f'layernorm1_{layer} -> self_attn_{layer};\\n'
        
        # First residual
        dag += f'residual1_{layer} [label="{layer_label}\\nResidual Add 1\\nInput: [1,2048,{config["hidden_dimension"]}], [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=grey];\\n'
        dag += f'{previous_layer} -> residual1_{layer} [style=dashed];\\n'
        dag += f'self_attn_{layer} -> residual1_{layer};\\n'
        
        # LayerNorm 2
        dag += f'layernorm2_{layer} [label="{layer_label}\\nLayerNorm 2\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=yellow];\\n'
        dag += f'residual1_{layer} -> layernorm2_{layer};\\n'
        
        # MLP
        dag += f'mlp_{layer} [label="{layer_label}\\nMLP\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=lightgreen];\\n'
        dag += f'layernorm2_{layer} -> mlp_{layer};\\n'
        
        # Second residual
        dag += f'residual2_{layer} [label="{layer_label}\\nResidual Add 2\\nInput: [1,2048,{config["hidden_dimension"]}], [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=grey];\\n'
        dag += f'residual1_{layer} -> residual2_{layer} [style=dashed];\\n'
        dag += f'mlp_{layer} -> residual2_{layer};\\n'
        
        previous_layer = f"residual2_{layer}"
    
    # Final processing
    dag += f'final_layernorm [label="Final LayerNorm\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["hidden_dimension"]}]\\nGPU: 0", fillcolor=yellow];\\n'
    dag += f'{previous_layer} -> final_layernorm;\\n'
    
    dag += f'lm_head [label="Language Model Head\\nInput: [1,2048,{config["hidden_dimension"]}]\\nOutput: [1,2048,{config["vocabulary_size"]}]\\nGPU: 0", fillcolor=orange];\\n'
    dag += f'final_layernorm -> lm_head;\\n'
    
    dag += "}\\n"
    return dag

def main():
    output_dir = "../outputs/2025-10-29-14-21-46"
    generator = DAGGenerator(output_dir)
    
    # Model configurations
    configs = {
        "megatron_8_3b": {
            "hidden_dimension": 4096,
            "layers": 24,
            "vocabulary_size": 51200
        },
        "megatron_530b": {
            "hidden_dimension": 20480,
            "layers": 105,
            "vocabulary_size": 51200
        },
        "megatron_1t": {
            "hidden_dimension": 25600,
            "layers": 128,
            "vocabulary_size": 51200
        },
        "gopher_280b": {
            "hidden_dimension": 16384,
            "layers": 80,
            "vocabulary_size": 32000
        },
        "palm_540b": {
            "hidden_dimension": 18432,
            "layers": 118,
            "vocabulary_size": 256000
        },
        "gpt3_175b": {
            "hidden_dimension": 12288,
            "layers": 96,
            "vocabulary_size": 50000
        }
    }
    
    # Generate DAGs for each model
    for model_name, config in configs.items():
        # Baseline DAG (no parallelism)
        baseline_dag = generator.generate_baseline_dag(f"{model_name}_baseline", config)
        with open(f"{output_dir}/{model_name}_baseline.dot", "w") as f:
            f.write(baseline_dag)
        
        # Optimized DAG with full parallelism
        # Using actual deployment configurations
        if model_name == "megatron_8_3b":
            total_devices, tensor_par, pipeline_par = 512, 8, 1
        elif model_name == "megatron_530b":
            total_devices, tensor_par, pipeline_par = 3360, 8, 35
        elif model_name == "megatron_1t":
            total_devices, tensor_par, pipeline_par = 512, 8, 64
        elif model_name == "gopher_280b":
            total_devices, tensor_par, pipeline_par = 4096, 8, 4
        elif model_name == "palm_540b":
            total_devices, tensor_par, pipeline_par = 6144, 12, 1
        else:  # gpt3_175b
            total_devices, tensor_par, pipeline_par = 1024, 8, 16
        
        optimized_dag = generator.generate_full_model_dag(f"{model_name}_optimized", config, 
                                                         total_devices, tensor_par, pipeline_par)
        with open(f"{output_dir}/{model_name}_optimized.dot", "w") as f:
            f.write(optimized_dag)
        
        # Generate individual layer DAGs for tensor parallel groups
        for device in range(min(8, tensor_par)):  # First 8 devices for tensor parallel
            attention_dag = generator.generate_attention_dag(f"{model_name}", config, device, tensor_par)
            with open(f"{output_dir}/{model_name}_attention_device_{device}.dot", "w") as f:
                f.write(attention_dag)
            
            mlp_dag = generator.generate_mlp_dag(f"{model_name}", config, device, tensor_par)
            with open(f"{output_dir}/{model_name}_mlp_device_{device}.dot", "w") as f:
                f.write(mlp_dag)

if __name__ == "__main__":
    main()