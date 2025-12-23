#!/usr/bin/env python3
"""
Complete LLM Deployment DAG Generator for PP(4) x TP(2) Strategy
This generates a fully detailed DAG with all computational nodes and proper connections.
"""

def generate_attention_block(stage, layer, gpu_pair, color):
    """Generate complete attention block with all submodules"""
    nodes = []
    edges = []
    
    # QKV Linear splits
    for gpu in gpu_pair:
        nodes.append(f'layer{layer}_qkv_{gpu} [label="QKV Linear\\nLayer {layer}\\nGPU:{gpu}\\nInput:[4,2048,8192]\\nOutput:[4,2048,12288]" fillcolor="{color}" shape=rectangle]')
    
    # QKV All-Reduce
    nodes.append(f'qkv_ar_{layer}_{stage} [label="All-Reduce\\nQKV\\nLayer {layer}\\nGPU:{gpu_pair}" fillcolor=lightblue shape=ellipse]')
    
    # Attention computation (this would be the actual attention mechanism)
    for gpu in gpu_pair:
        nodes.append(f'layer{layer}_attn_{gpu} [label="Self-Attention\\nLayer {layer}\\nGPU:{gpu}\\nInput:[4,2048,12288]\\nOutput:[4,2048,8192]" fillcolor="{color}" shape=rectangle]')
    
    # Attention output projection
    for gpu in gpu_pair:
        nodes.append(f'layer{layer}_attn_out_{gpu} [label="Attention Output\\nLayer {layer}\\nGPU:{gpu}\\nInput:[4,2048,8192]\\nOutput:[4,2048,8192]" fillcolor="{color}" shape=rectangle]')
    
    # Attention All-Reduce
    nodes.append(f'attn_out_ar_{layer}_{stage} [label="All-Reduce\\nAttention\\nLayer {layer}\\nGPU:{gpu_pair}" fillcolor=lightblue shape=ellipse]')
    
    return nodes, edges

def generate_ffn_block(stage, layer, gpu_pair, color):
    """Generate complete FFN block with all submodules"""
    nodes = []
    edges = []
    
    # FFN Gate
    for gpu in gpu_pair:
        nodes.append(f'layer{layer}_ffn_gate_{gpu} [label="FFN Gate\\nLayer {layer}\\nGPU:{gpu}\\nInput:[4,2048,8192]\\nOutput:[4,2048,28672]" fillcolor="{color}" shape=rectangle]')
    
    # FFN Up
    for gpu in gpu_pair:
        nodes.append(f'layer{layer}_ffn_up_{gpu} [label="FFN Up\\nLayer {layer}\\nGPU:{gpu}\\nInput:[4,2048,8192]\\nOutput:[4,2048,28672]" fillcolor="{color}" shape=rectangle]')
    
    # FFN All-Reduce
    nodes.append(f'ffn_ar_{layer}_{stage} [label="All-Reduce\\nFFN\\nLayer {layer}\\nGPU:{gpu_pair}" fillcolor=lightblue shape=ellipse]')
    
    # FFN Down
    for gpu in gpu_pair:
        nodes.append(f'layer{layer}_ffn_down_{gpu} [label="FFN Down\\nLayer {layer}\\nGPU:{gpu}\\nInput:[4,2048,28672]\\nOutput:[4,2048,8192]" fillcolor="{color}" shape=rectangle]')
    
    # FFN Output All-Reduce
    nodes.append(f'ffn_out_ar_{layer}_{stage} [label="All-Reduce\\nFFN Output\\nLayer {layer}\\nGPU:{gpu_pair}" fillcolor=lightblue shape=ellipse]')
    
    return nodes, edges

def generate_stage(stage_id, start_layer, end_layer, gpu_pair, color):
    """Generate complete stage with all layers"""
    nodes = []
    edges = []
    
    # Stage header
    stage_label = f"Stage {stage_id}: GPUs {gpu_pair}\\nLayers {start_layer}-{end_layer}"
    nodes.append(f'subgraph cluster_stage{stage_id} {{')
    nodes.append(f'\tbgcolor=lightgray label="{stage_label}" style=rounded')
    
    # Input split for first layer
    if stage_id == 0:
        nodes.append(f'\tsplit_0 [label="Input Split\\n[batch_size=4, seq_len=2048, hidden=4096]" fillcolor=lightyellow shape=parallelogram]')
        
        # Embedding layers
        for gpu in gpu_pair:
            nodes.append(f'\tlayer0_embed_{gpu} [label="Embedding\\nGPU:{gpu}\\nInput:[4,2048]\\nOutput:[4,2048,8192]" fillcolor="{color}" shape=rectangle]')
        nodes.append(f'\tembed_ag_0 [label="All-Gather\\nEmbedding\\nGPU:{gpu_pair}" fillcolor=lightblue shape=ellipse]')
    
    # Generate all layers
    for layer in range(start_layer, end_layer + 1):
        # Attention block
        attn_nodes, attn_edges = generate_attention_block(stage_id, layer, gpu_pair, color)
        nodes.extend([f'\t{node}' for node in attn_nodes])
        
        # FFN block
        ffn_nodes, ffn_edges = generate_ffn_block(stage_id, layer, gpu_pair, color)
        nodes.extend([f'\t{node}' for node in ffn_nodes])
    
    nodes.append('}')
    
    return nodes, edges

def generate_pipeline_connections():
    """Generate pipeline communication connections"""
    connections = []
    
    # Stage 0 → 1
    connections.append(f'pp_send_0_1 [label="Pipeline Send\\nStage 0→1\\nGPU:[1]→[2]" fillcolor=lightcoral shape=ellipse]')
    
    # Stage 1 → 2  
    connections.append(f'pp_send_1_2 [label="Pipeline Send\\nStage 1→2\\nGPU:[3]→[4]" fillcolor=lightcoral shape=ellipse]')
    
    # Stage 2 → 3
    connections.append(f'pp_send_2_3 [label="Pipeline Send\\nStage 2→3\\nGPU:[5]→[6]" fillcolor=lightcoral shape=ellipse]')
    
    return connections

def generate_output_stage():
    """Generate final output processing"""
    nodes = []
    
    # RMSNorm for final output
    nodes.append(f'output_norm_0 [label="RMSNorm\\nGPU:6\\nInput:[4,2048,8192]\\nOutput:[4,2048,8192]" fillcolor="#FFF0E6" shape=rectangle]')
    nodes.append(f'output_norm_1 [label="RMSNorm\\nGPU:7\\nInput:[4,2048,8192]\\nOutput:[4,2048,8192]" fillcolor="#FFF0E6" shape=rectangle]')
    
    # All-Gather final output
    nodes.append(f'final_ag [label="All-Gather\\nFinal Output\\nGPU:[6,7]" fillcolor=lightblue shape=ellipse]')
    
    # LM Head
    nodes.append(f'lm_head_0 [label="LM Head\\nGPU:6\\nInput:[4,2048,8192]\\nOutput:[4,2048,128256]" fillcolor="#FFF0E6" shape=rectangle]')
    nodes.append(f'lm_head_1 [label="LM Head\\nGPU:7\\nInput:[4,2048,8192]\\nOutput:[4,2048,128256]" fillcolor="#FFF0E6" shape=rectangle]')
    
    # All-Reduce logits
    nodes.append(f'final_ar [label="All-Reduce\\nLogits\\nGPU:[6,7]" fillcolor=lightblue shape=ellipse]')
    
    return nodes

def generate_connections():
    """Generate all connections between nodes"""
    connections = []
    
    # Input connections
    connections.append('input -> split_0')
    connections.append('split_0 -> layer0_embed_0')
    connections.append('split_0 -> layer0_embed_1')
    connections.append('layer0_embed_0 -> embed_ag_0')
    connections.append('layer0_embed_1 -> embed_ag_0')
    
    # Stage 0 layer connections
    for layer in range(80):  # 0-79
        stage = layer // 20
        
        if stage == 0:  # Stage 0 (layers 0-19)
            if layer == 0:
                # First layer connects from embedding
                connections.append(f'embed_ag_0 -> layer1_qkv_0')
                connections.append(f'embed_ag_0 -> layer1_qkv_1')
            else:
                # Connect from previous layer
                connections.append(f'ffn_out_ar_{layer-1}_{stage} -> layer{layer}_qkv_0')
                connections.append(f'ffn_out_ar_{layer-1}_{stage} -> layer{layer}_qkv_1')
        
        # Within-layer connections for each stage
        gpu_pair = [stage*2, stage*2+1]
        for gpu in gpu_pair:
            # QKV connections
            connections.append(f'layer{layer}_qkv_{gpu} -> qkv_ar_{layer}_{stage}')
            connections.append(f'qkv_ar_{layer}_{stage} -> layer{layer}_attn_{gpu}')
            connections.append(f'layer{layer}_attn_{gpu} -> layer{layer}_attn_out_{gpu}')
            connections.append(f'layer{layer}_attn_out_{gpu} -> attn_out_ar_{layer}_{stage}')
            
            # FFN connections
            connections.append(f'attn_out_ar_{layer}_{stage} -> layer{layer}_ffn_gate_{gpu}')
            connections.append(f'attn_out_ar_{layer}_{stage} -> layer{layer}_ffn_up_{gpu}')
            connections.append(f'layer{layer}_ffn_gate_{gpu} -> ffn_ar_{layer}_{stage}')
            connections.append(f'layer{layer}_ffn_up_{gpu} -> ffn_ar_{layer}_{stage}')
            connections.append(f'ffn_ar_{layer}_{stage} -> layer{layer}_ffn_down_{gpu}')
            connections.append(f'layer{layer}_ffn_down_{gpu} -> ffn_out_ar_{layer}_{stage}')
    
    # Pipeline connections
    connections.append('ffn_out_ar_19_0 -> pp_send_0_1')
    connections.append('pp_send_0_1 -> layer20_qkv_0')
    connections.append('pp_send_0_1 -> layer20_qkv_1')
    
    connections.append('ffn_out_ar_39_1 -> pp_send_1_2')
    connections.append('pp_send_1_2 -> layer40_qkv_0')
    connections.append('pp_send_1_2 -> layer40_qkv_1')
    
    connections.append('ffn_out_ar_59_2 -> pp_send_2_3')
    connections.append('pp_send_2_3 -> layer60_qkv_0')
    connections.append('pp_send_2_3 -> layer60_qkv_1')
    
    # Output connections
    connections.append('ffn_out_ar_79_3 -> output_norm_0')
    connections.append('ffn_out_ar_79_3 -> output_norm_1')
    connections.append('output_norm_0 -> final_ag')
    connections.append('output_norm_1 -> final_ag')
    connections.append('final_ag -> lm_head_0')
    connections.append('final_ag -> lm_head_1')
    connections.append('lm_head_0 -> final_ar')
    connections.append('lm_head_1 -> final_ar')
    connections.append('final_ar -> output')
    
    return connections

def main():
    """Generate complete LLM deployment DAG"""
    
    # Stage configurations
    stages = [
        (0, 0, 19, [0, 1], "#FFE6E6"),    # Stage 0: GPUs [0,1], Layers 0-19
        (1, 20, 39, [2, 3], "#E6F3FF"),   # Stage 1: GPUs [2,3], Layers 20-39
        (2, 40, 59, [4, 5], "#E6FFE6"),   # Stage 2: GPUs [4,5], Layers 40-59
        (3, 60, 79, [6, 7], "#FFF0E6")    # Stage 3: GPUs [6,7], Layers 60-79
    ]
    
    # Generate complete DAG
    print("// Complete LLM Deployment DAG - PP(4) x TP(2)")
    print("digraph {")
    print("\tnodesep=0.5 rankdir=TB ranksep=1.0 size=\"20,30\"")
    print("\tnode [fillcolor=lightblue shape=ellipse style=filled]")
    print("\tnode [fillcolor=lightgreen shape=rectangle style=filled]")
    print("\tnode [fillcolor=lightyellow shape=parallelogram style=filled]")
    
    # Input node
    print('\tinput [label="Input\\n[batch_size=4, seq_len=2048, hidden=8192]" fillcolor=white shape=ellipse style=filled]')
    
    # Generate all stages
    all_nodes = []
    all_connections = []
    
    for stage_id, start_layer, end_layer, gpu_pair, color in stages:
        stage_nodes, stage_edges = generate_stage(stage_id, start_layer, end_layer, gpu_pair, color)
        all_nodes.extend(stage_nodes)
        all_connections.extend(stage_edges)
    
    # Generate pipeline connections
    pipeline_nodes = generate_pipeline_connections()
    
    # Generate output stage
    output_nodes = generate_output_stage()
    
    # Print all nodes
    for node in all_nodes:
        print(f"\t{node}")
    
    for node in pipeline_nodes:
        print(f"\t{node}")
        
    for node in output_nodes:
        print(f"\t{node}")
    
    # Generate and print connections
    connections = generate_connections()
    
    print("")
    for connection in connections:
        print(f"\t{connection}")
    
    print("}")

if __name__ == "__main__":
    main()