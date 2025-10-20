import graphviz
import os

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2 configuration"""
    dot = graphviz.Digraph('baseline_transformer', 
                          comment='4-Layer Transformer Baseline (TP=8, PP=2)',
                          graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.5'})
    
    # Set node attributes
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input Embedding\\nGPU: [0-7], [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(label='Stage 0 - GPUs [0-7]\\nTP=8, Partition=512', style='dashed', color='red')
        
        # Layer 0 RMSNorm
        c.node('layer0_rmsnorm', 'Layer 0 RMSNorm\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 0 Attention (across 8 GPUs)
        c.node('layer0_attn_qkv', 'Layer 0 QKV Projection\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128, partition=512]')
        c.node('layer0_attn_scores', 'Layer 0 Attention Scores\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]')
        c.node('layer0_attn_weighted', 'Layer 0 Weighted Sum\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128], [batch_size=1024, seq_len=?, seq_len=?]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]')
        c.node('layer0_attn_out', 'Layer 0 Attention Output\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        c.node('layer0_res_add1', 'Layer 0 Residual Add 1\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512], [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 0 FFN
        c.node('layer0_ffn1', 'Layer 0 FFN Up\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]')
        c.node('layer0_gelu', 'Layer 0 GELU\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]\\nOutput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]')
        c.node('layer0_ffn2', 'Layer 0 FFN Down\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        c.node('layer0_res_add2', 'Layer 0 Residual Add 2\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512], [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 1 RMSNorm
        c.node('layer1_rmsnorm', 'Layer 1 RMSNorm\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 1 Attention (same structure as layer 0)
        c.node('layer1_attn_qkv', 'Layer 1 QKV Projection\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128, partition=512]')
        c.node('layer1_attn_scores', 'Layer 1 Attention Scores\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]')
        c.node('layer1_attn_weighted', 'Layer 1 Weighted Sum\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128], [batch_size=1024, seq_len=?, seq_len=?]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]')
        c.node('layer1_attn_out', 'Layer 1 Attention Output\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        c.node('layer1_res_add1', 'Layer 1 Residual Add 1\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512], [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 1 FFN
        c.node('layer1_ffn1', 'Layer 1 FFN Up\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]')
        c.node('layer1_gelu', 'Layer 1 GELU\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]\\nOutput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]')
        c.node('layer1_ffn2', 'Layer 1 FFN Down\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        c.node('layer1_res_add2', 'Layer 1 Residual Add 2\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512], [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
    
    # Stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Stage 1 - GPUs [8-15]\\nTP=8, Partition=512', style='dashed', color='blue')
        
        # Pipeline communication between stages
        c.node('pipeline_comm_1', 'Pipeline Communication\\nGPU: [7] → [8]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]', 
               shape='parallelogram', fillcolor='yellow')
        
        # Layer 2 RMSNorm
        c.node('layer2_rmsnorm', 'Layer 2 RMSNorm\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 2 Attention
        c.node('layer2_attn_qkv', 'Layer 2 QKV Projection\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128, partition=512]')
        c.node('layer2_attn_scores', 'Layer 2 Attention Scores\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]')
        c.node('layer2_attn_weighted', 'Layer 2 Weighted Sum\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128], [batch_size=1024, seq_len=?, seq_len=?]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]')
        c.node('layer2_attn_out', 'Layer 2 Attention Output\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        c.node('layer2_res_add1', 'Layer 2 Residual Add 1\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512], [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 2 FFN
        c.node('layer2_ffn1', 'Layer 2 FFN Up\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]')
        c.node('layer2_gelu', 'Layer 2 GELU\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]\\nOutput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]')
        c.node('layer2_ffn2', 'Layer 2 FFN Down\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        c.node('layer2_res_add2', 'Layer 2 Residual Add 2\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512], [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 3 RMSNorm
        c.node('layer3_rmsnorm', 'Layer 3 RMSNorm\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 3 Attention
        c.node('layer3_attn_qkv', 'Layer 3 QKV Projection\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128, partition=512]')
        c.node('layer3_attn_scores', 'Layer 3 Attention Scores\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]')
        c.node('layer3_attn_weighted', 'Layer 3 Weighted Sum\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128], [batch_size=1024, seq_len=?, seq_len=?]\\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]')
        c.node('layer3_attn_out', 'Layer 3 Attention Output\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        c.node('layer3_res_add1', 'Layer 3 Residual Add 1\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512], [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 3 FFN
        c.node('layer3_ffn1', 'Layer 3 FFN Up\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]')
        c.node('layer3_gelu', 'Layer 3 GELU\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]\\nOutput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]')
        c.node('layer3_ffn2', 'Layer 3 FFN Down\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, ffn_hidden=16384, partition=2048]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        c.node('layer3_res_add2', 'Layer 3 Residual Add 2\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512], [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Output projection
        c.node('output_proj', 'Output Projection\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, vocab_size=?, partition=?]')
    
    # Output node
    dot.node('output', 'Final Output\\nGPU: [8-15]\\nInput: [batch_size=1024, seq_len=?, vocab_size=?, partition=?]\\nOutput: [batch_size=1024, seq_len=?, vocab_size=?]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connections
    dot.edge('input', 'layer0_rmsnorm')
    dot.edge('layer0_rmsnorm', 'layer0_attn_qkv')
    dot.edge('layer0_attn_qkv', 'layer0_attn_scores')
    dot.edge('layer0_attn_scores', 'layer0_attn_weighted')
    dot.edge('layer0_attn_weighted', 'layer0_attn_out')
    dot.edge('layer0_attn_out', 'layer0_res_add1')
    dot.edge('input', 'layer0_res_add1')  # Residual connection
    dot.edge('layer0_res_add1', 'layer0_ffn1')
    dot.edge('layer0_ffn1', 'layer0_gelu')
    dot.edge('layer0_gelu', 'layer0_ffn2')
    dot.edge('layer0_ffn2', 'layer0_res_add2')
    dot.edge('layer0_res_add1', 'layer0_res_add2')  # Residual connection
    
    dot.edge('layer0_res_add2', 'layer1_rmsnorm')
    dot.edge('layer1_rmsnorm', 'layer1_attn_qkv')
    dot.edge('layer1_attn_qkv', 'layer1_attn_scores')
    dot.edge('layer1_attn_scores', 'layer1_attn_weighted')
    dot.edge('layer1_attn_weighted', 'layer1_attn_out')
    dot.edge('layer1_attn_out', 'layer1_res_add1')
    dot.edge('layer0_res_add2', 'layer1_res_add1')  # Residual connection
    dot.edge('layer1_res_add1', 'layer1_ffn1')
    dot.edge('layer1_ffn1', 'layer1_gelu')
    dot.edge('layer1_gelu', 'layer1_ffn2')
    dot.edge('layer1_ffn2', 'layer1_res_add2')
    dot.edge('layer1_res_add1', 'layer1_res_add2')  # Residual connection
    
    # Pipeline stage transition
    dot.edge('layer1_res_add2', 'pipeline_comm_1')
    dot.edge('pipeline_comm_1', 'layer2_rmsnorm')
    
    dot.edge('layer2_rmsnorm', 'layer2_attn_qkv')
    dot.edge('layer2_attn_qkv', 'layer2_attn_scores')
    dot.edge('layer2_attn_scores', 'layer2_attn_weighted')
    dot.edge('layer2_attn_weighted', 'layer2_attn_out')
    dot.edge('layer2_attn_out', 'layer2_res_add1')
    dot.edge('pipeline_comm_1', 'layer2_res_add1')  # Residual connection
    dot.edge('layer2_res_add1', 'layer2_ffn1')
    dot.edge('layer2_ffn1', 'layer2_gelu')
    dot.edge('layer2_gelu', 'layer2_ffn2')
    dot.edge('layer2_ffn2', 'layer2_res_add2')
    dot.edge('layer2_res_add1', 'layer2_res_add2')  # Residual connection
    
    dot.edge('layer2_res_add2', 'layer3_rmsnorm')
    dot.edge('layer3_rmsnorm', 'layer3_attn_qkv')
    dot.edge('layer3_attn_qkv', 'layer3_attn_scores')
    dot.edge('layer3_attn_scores', 'layer3_attn_weighted')
    dot.edge('layer3_attn_weighted', 'layer3_attn_out')
    dot.edge('layer3_attn_out', 'layer3_res_add1')
    dot.edge('layer2_res_add2', 'layer3_res_add1')  # Residual connection
    dot.edge('layer3_res_add1', 'layer3_ffn1')
    dot.edge('layer3_ffn1', 'layer3_gelu')
    dot.edge('layer3_gelu', 'layer3_ffn2')
    dot.edge('layer3_ffn2', 'layer3_res_add2')
    dot.edge('layer3_res_add1', 'layer3_res_add2')  # Residual connection
    
    dot.edge('layer3_res_add2', 'output_proj')
    dot.edge('output_proj', 'output')
    
    return dot

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('./generated_docs/FA', exist_ok=True)
    
    # Generate baseline DAG
    baseline_dag = create_baseline_dag()
    
    # Save DOT file
    with open('./generated_docs/FA/baseline_transformer.dot', 'w') as f:
        f.write(baseline_dag.source)
    
    # Render to SVG
    baseline_dag.format = 'svg'
    baseline_dag.render('./generated_docs/FA/baseline_transformer', cleanup=True)
    
    print("Baseline DAG generated: ./generated_docs/FA/baseline_transformer.svg")
    print("DOT file saved: ./generated_docs/FA/baseline_transformer.dot")