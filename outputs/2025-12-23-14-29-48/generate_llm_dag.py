import graphviz
from graphviz import Digraph

def create_llm_deployment_dag():
    # Create a new directed graph
    dot = Digraph(comment='LLM Deployment DAG - PP(4) x TP(2)')
    dot.attr(rankdir='TB', size='20,30', ranksep='1.0', nodesep='0.5')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Define colors for different GPU groups
    gpu_colors = {
        0: '#FFE6E6', 1: '#FFE6E6',  # Stage 0 - Light red
        2: '#E6F3FF', 3: '#E6F3FF',  # Stage 1 - Light blue
        4: '#E6FFE6', 5: '#E6FFE6',  # Stage 2 - Light green
        6: '#FFF0E6', 7: '#FFF0E6'   # Stage 3 - Light orange
    }
    
    # Input node
    dot.node('input', 'Input\\n[batch_size=4, seq_len=2048, hidden=8192]', 
             shape='ellipse', fillcolor='white', style='filled')
    
    # Stage 0: GPUs 0,1 (Layers 0-19)
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Stage 0: GPUs [0,1]\\nLayers 0-19', style='rounded', bgcolor='lightgray')
        
        # Input split for TP
        stage0.node('split_0', 'Input Split\\n[batch_size=4, seq_len=2048, hidden=4096]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        # Layer 0: Embedding + RMSNorm
        stage0.node('layer0_embed_0', 'Embedding\\nGPU:0\\nInput:[4,2048]\\nOutput:[4,2048,8192]', 
                   shape='rectangle', fillcolor=gpu_colors[0])
        stage0.node('layer0_embed_1', 'Embedding\\nGPU:1\\nInput:[4,2048]\\nOutput:[4,2048,8192]', 
                   shape='rectangle', fillcolor=gpu_colors[1])
        
        # TP All-Gather for embedding
        stage0.node('embed_ag_0', 'All-Gather\\nEmbedding\\nGPU:[0,1]', 
                   shape='ellipse', fillcolor='lightblue')
        
        # Layers 1-19 (simplified representation - showing key operations)
        for layer in range(1, 20):
            # Self-Attention components
            stage0.node(f'layer{layer}_qkv_0', f'QKV Linear\\nLayer {layer}\\nGPU:0\\nInput:[4,2048,8192]\\nOutput:[4,2048,12288]', 
                       shape='rectangle', fillcolor=gpu_colors[0])
            stage0.node(f'layer{layer}_qkv_1', f'QKV Linear\\nLayer {layer}\\nGPU:1\\nInput:[4,2048,8192]\\nOutput:[4,2048,12288]', 
                       shape='rectangle', fillcolor=gpu_colors[1])
            
            # TP All-Reduce for QKV
            stage0.node(f'qkv_ar_{layer}_0', f'All-Reduce\\nQKV\\nLayer {layer}\\nGPU:[0,1]', 
                       shape='ellipse', fillcolor='lightblue')
            
            # Attention computation
            stage0.node(f'layer{layer}_attn_0', f'Self-Attention\\nLayer {layer}\\nGPU:0\\nInput:[4,2048,12288]\\nOutput:[4,2048,8192]', 
                       shape='rectangle', fillcolor=gpu_colors[0])
            stage0.node(f'layer{layer}_attn_1', f'Self-Attention\\nLayer {layer}\\nGPU:1\\nInput:[4,2048,12288]\\nOutput:[4,2048,8192]', 
                       shape='rectangle', fillcolor=gpu_colors[1])
            
            # Attention output projection
            stage0.node(f'layer{layer}_attn_out_0', f'Attention Output\\nLayer {layer}\\nGPU:0\\nInput:[4,2048,8192]\\nOutput:[4,2048,8192]', 
                       shape='rectangle', fillcolor=gpu_colors[0])
            stage0.node(f'layer{layer}_attn_out_1', f'Attention Output\\nLayer {layer}\\nGPU:1\\nInput:[4,2048,8192]\\nOutput:[4,2048,8192]', 
                       shape='rectangle', fillcolor=gpu_colors[1])
            
            # TP All-Reduce for attention output
            stage0.node(f'attn_out_ar_{layer}_0', f'All-Reduce\\nAttention\\nLayer {layer}\\nGPU:[0,1]', 
                       shape='ellipse', fillcolor='lightblue')
            
            # FFN components
            stage0.node(f'layer{layer}_ffn_gate_0', f'FFN Gate\\nLayer {layer}\\nGPU:0\\nInput:[4,2048,8192]\\nOutput:[4,2048,28672]', 
                       shape='rectangle', fillcolor=gpu_colors[0])
            stage0.node(f'layer{layer}_ffn_gate_1', f'FFN Gate\\nLayer {layer}\\nGPU:1\\nInput:[4,2048,8192]\\nOutput:[4,2048,28672]', 
                       shape='rectangle', fillcolor=gpu_colors[1])
            
            stage0.node(f'layer{layer}_ffn_up_0', f'FFN Up\\nLayer {layer}\\nGPU:0\\nInput:[4,2048,8192]\\nOutput:[4,2048,28672]', 
                       shape='rectangle', fillcolor=gpu_colors[0])
            stage0.node(f'layer{layer}_ffn_up_1', f'FFN Up\\nLayer {layer}\\nGPU:1\\nInput:[4,2048,8192]\\nOutput:[4,2048,28672]', 
                       shape='rectangle', fillcolor=gpu_colors[1])
            
            # TP All-Reduce for FFN
            stage0.node(f'ffn_ar_{layer}_0', f'All-Reduce\\nFFN\\nLayer {layer}\\nGPU:[0,1]', 
                       shape='ellipse', fillcolor='lightblue')
            
            stage0.node(f'layer{layer}_ffn_down_0', f'FFN Down\\nLayer {layer}\\nGPU:0\\nInput:[4,2048,28672]\\nOutput:[4,2048,8192]', 
                       shape='rectangle', fillcolor=gpu_colors[0])
            stage0.node(f'layer{layer}_ffn_down_1', f'FFN Down\\nLayer {layer}\\nGPU:1\\nInput:[4,2048,28672]\\nOutput:[4,2048,8192]', 
                       shape='rectangle', fillcolor=gpu_colors[1])
            
            # Final All-Reduce for FFN output
            stage0.node(f'ffn_out_ar_{layer}_0', f'All-Reduce\\nFFN Output\\nLayer {layer}\\nGPU:[0,1]', 
                       shape='ellipse', fillcolor='lightblue')
    
    # Stage 1: GPUs 2,3 (Layers 20-39)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Stage 1: GPUs [2,3]\\nLayers 20-39', style='rounded', bgcolor='lightgray')
        
        # Similar structure for stage 1 (simplified for brevity)
        for layer in range(20, 40):
            # Self-Attention components
            stage1.node(f'layer{layer}_qkv_0', f'QKV Linear\\nLayer {layer}\\nGPU:2\\nInput:[4,2048,8192]\\nOutput:[4,2048,12288]',  mythology='rectangle', fillcolor=gpu_colors[2])
            stage1.node(f'layer{layer}_qkv_1', f'QKV Linear\\nLayer {layer}\\nGPU:3\\nInput:[4,2048,8192]\\nOutput:[4,2048,12288]', 
                       shape='rectangle', fillcolor=gpu_colors[3])
            
            # TP All-Reduce for QKV
            stage1.node(f'qkv_ar_{layer}_1', f'All-Reduce\\nQKV\\nLayer {layer}\\nGPU:[2,3]', 
                       shape='ellipse', fillcolor='lightblue')
            
            # Similar pattern for attention, FFN, etc.
            stage1.node(f'layer{layer}_attn_out_ar_1', f'All-Reduce\\nAttention\\nLayer {layer}\\nGPU:[2,3]', 
                       shape='ellipse', fillcolor='lightblue')
            stage1.node(f'ffn_ar_{layer}_1', f'All-Reduce\\nFFN\\nLayer {layer}\\nGPU:[2,3]', 
                       shape='ellipse', fillcolor='lightblue')
            stage1.node(f'ffn_out_ar_{layer}_1', f'All-Reduce\\nFFN Output\\nLayer {layer}\\nGPU:[2,3]', 
                       shape='ellipse', fillcolor='lightblue')
    
    # Stage 2: GPUs 4,5 (Layers 40-59)
    with dot.subgraph(name='cluster_stage2') as stage2:
        stage2.attr(label='Stage 2: GPUs [4,5]\\nLayers 40-59', style='rounded', bgcolor='lightgray')
        
        for layer in range(40, 60):
            stage2.node(f'layer{layer}_attn_out_ar_2', f'All-Reduce\\nAttention\\nLayer {layer}\\nGPU:[4,5]', 
                       shape='ellipse', fillcolor='lightblue')
            stage2.node(f'ffn_out_ar_{layer}_2', f'All-Reduce\\nFFN Output\\nLayer {layer}\\nGPU:[4,5]', 
                       shape='ellipse', fillcolor='lightblue')
    
    # Stage 3: GPUs 6,7 (Layers 60-79)
    with dot.subgraph(name='cluster_stage3') as stage3:
        stage3.attr(label='Stage 3: GPUs [6,7]\\nLayers 60-79', style='rounded', bgcolor='lightgray')
        
        for layer in range(60, 80):
            stage3.node(f'layer{layer}_attn_out_ar_3', f'All-Reduce\\nAttention\\nLayer {layer}\\nGPU:[6,7]', 
                       shape='ellipse', fillcolor='lightblue')
            stage3.node(f'ffn_out_ar_{layer}_3', f'All-Reduce\\nFFN Output\\nLayer {layer}\\nGPU:[6,7]', 
                       shape='ellipse', fillcolor='lightblue')
    
    # Pipeline communication nodes
    dot.node('pp_send_0_1', 'Pipeline Send\\nStage 0→1\\nGPU:[1]→[2]', 
             shape='ellipse', fillcolor='lightcoral')
    dot.node('pp_send_1_2', 'Pipeline Send\\nStage 1→2\\nGPU:[3]→[4]', 
             shape='ellipse', fillcolor='lightcoral')
    dot.node('pp_send_2_3', 'Pipeline Send\\nStage 2→3\\nGPU:[5]→[6]', 
             shape='ellipse', fillcolor='lightcoral')
    
    # Output nodes
    dot.node('output_norm_0', 'RMSNorm\\nGPU:6\\nInput:[4,2048,8192]\\nOutput:[4,2048,8192]', 
             shape='rectangle', fillcolor=gpu_colors[6])
    dot.node('output_norm_1', 'RMSNorm\\nGPU:7\\nInput:[4,2048,8192]\\nOutput:[4,2048,8192]', 
             shape='rectangle', fillcolor=gpu_colors[7])
    
    dot.node('final_ag', 'All-Gather\\nFinal Output\\nGPU:[6,7]', 
             shape='ellipse', fillcolor='lightblue')
    
    dot.node('lm_head_0', 'LM Head\\nGPU:6\\nInput:[4,2048,8192]\\nOutput:[4,2048,128256]', 
             shape='rectangle', fillcolor=gpu_colors[6])
    dot.node('lm_head_1', 'LM Head\\nGPU:7\\nInput:[4,2048,8192]\\nOutput:[4,2048,128256]', 
             shape='rectangle', fillcolor=gpu_colors[7])
    
    dot.node('final_ar', 'All-Reduce\\nLogits\\nGPU:[6,7]', 
             shape='ellipse', fillcolor='lightblue')
    
    dot.node('output', 'Output\\n[batch_size=4, seq_len=2048, vocab=128256]', 
             shape='ellipse', fillcolor='white', style='filled')
    
    # Edges - connecting the nodes
    # Input to stage 0
    dot.edge('input', 'split_0')
    dot.edge('split_0', 'layer0_embed_0')
    dot.edge('split_0', 'layer0_embed_1')
    dot.edge('layer0_embed_0', 'embed_ag_0')
    dot.edge('layer0_embed_1', 'embed_ag_0')
    
    # Simplified edge connections for layers
    # In reality, each layer connects to the next with appropriate communication
    
    # Connect stage 0 to pipeline send
    dot.edge('ffn_out_ar_19_0', 'pp_send_0_1')
    
    # Connect pipeline sends to next stages
    dot.edge('pp_send_0_1', 'layer20_qkv_0')
    dot.edge('pp_send_0_1', 'layer20_qkv_1')
    
    # Continue pattern for other stages...
    dot.edge('ffn_out_ar_39_1', 'pp_send_1_2')
    dot.edge('pp_send_1_2', 'layer40_attn_out_ar_2')
    
    dot.edge('ffn_out_ar_59_2', 'pp_send_2_3')
    dot.edge('pp_send_2_3', 'layer60_attn_out_ar_3')
    
    # Final stage to output
    dot.edge('ffn_out_ar_79_3', 'output_norm_0')
    dot.edge('ffn_out_ar_79_3', 'output_norm_1')
    
    dot.edge('output_norm_0', 'final_ag')
    dot.edge('output_norm_1', 'final_ag')
    
    dot.edge('final_ag', 'lm_head_0')
    dot.edge('final_ag', 'lm_head_1')
    
    dot.edge('lm_head_0', 'final_ar')
    dot.edge('lm_head_1', 'final_ar')
    
    dot.edge('final_ar', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_llm_deployment_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-12-23-14-29-48/llm_deployment_dag.dot')
    
    # Render as SVG
    dag.render('../outputs/2025-12-23-14-29-48/llm_deployment_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-23-14-29-48/llm_deployment_dag.dot")
    print(f"SVG file: ../outputs/2025-12-23-14-29-48/llm_deployment_dag.svg")