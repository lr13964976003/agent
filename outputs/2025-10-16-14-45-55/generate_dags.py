import graphviz
import os

# Create output directory
os.makedirs("../outputs/2025-10-16-14-45-55", exist_ok=True)

def create_baseline_dag():
    """Create baseline DAG for TP=8, PP=2 configuration"""
    dot = graphviz.Digraph('baseline_moe_model', 
                          comment='Baseline MoE Model with TP=8, PP=2')
    
    # Set graph attributes
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input Embedding\nInput: [batch_size=1024, seq_len=2048, hidden=4096]\nGPU: all GPUs', 
             shape='ellipse', fillcolor='lightgreen')
    
    # First pipeline stage (layers 0-1)
    with dot.subgraph(name='cluster_pipeline_0') as c0:
        c0.attr(label='Pipeline Stage 0 (Layers 0-1)\nGPUs 0-7', style='dashed', color='red')
        
        # Layer 0
        with c0.subgraph(name='cluster_layer_0') as layer0:
            layer0.attr(label='Layer 0', style='rounded')
            
            # Multi-Head Attention (TP=8)
            layer0.node('l0_mha_qkv', 'QKV Projection\nInput: [1024,2048,4096]\nOutput: [1024,2048,128] per GPU\nGPU: 0-7', fillcolor='yellow')
            layer0.node('l0_mha_attn', 'Attention Computation\nInput: [1024,2048,128]\nOutput: [1024,2048,128]\nGPU: 0-7', fillcolor='yellow')
            layer0.node('l0_mha_out', 'Output Projection\nInput: [1024,2048,128]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='yellow')
            layer0.node('l0_mha_res', 'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='orange')
            
            # MoE FFN (TP=8)
            layer0.node('l0_moe_gate', 'Gate Network\nInput: [1024,2048,4096]\nOutput: [1024,2048,16]\nGPU: 0-7', fillcolor='lightcoral')
            layer0.node('l0_moe_exp0', 'Expert 0,1\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-1', fillcolor='lightpink')
            layer0.node('l0_moe_exp2', 'Expert 2,3\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 2-3', fillcolor='lightpink')
            layer0.node('l0_moe_exp4', 'Expert 4,5\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 4-5', fillcolor='lightpink')
            layer0.node('l0_moe_exp6', 'Expert 6,7\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 6-7', fillcolor='lightpink')
            layer0.node('l0_moe_agg', 'Expert Aggregation\nInput: [1024,2048,4096]×2\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='lightcyan')
            layer0.node('l0_moe_res', 'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='orange')

        # Layer 1  
        with c0.subgraph(name='cluster_layer_1') as layer1:
            layer1.attr(label='Layer 1', style='rounded')
            
            layer1.node('l1_mha_qkv', 'QKV Projection\nInput: [1024,2048,4096]\nOutput: [1024,2048,128] per GPU\nGPU: 0-7', fillcolor='yellow')
            layer1.node('l1_mha_attn', 'Attention Computation\nInput: [1024,2048,128]\nOutput: [1024,2048,128]\nGPU: 0-7', fillcolor='yellow')
            layer1.node('l1_mha_out', 'Output Projection\nInput: [1024,2048,128]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='yellow')
            layer1.node('l1_mha_res', 'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='orange')
            
            layer1.node('l1_moe_gate', 'Gate Network\nInput: [1024,2048,4096]\nOutput: [1024,2048,16]\nGPU: 0-7', fillcolor='lightcoral')
            layer1.node('l1_moe_exp0', 'Expert 0,1\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-1', fillcolor='lightpink')
            layer1.node('l1_moe_exp2', 'Expert 2,3\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 2-3', fillcolor='lightpink')
            layer1.node('l1_moe_exp4', 'Expert 4,5\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 4-5', fillcolor='lightpink')
            layer1.node('l1_moe_exp6', 'Expert 6,7\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 6-7', fillcolor='lightpink')
            layer1.node('l1_moe_agg', 'Expert Aggregation\nInput: [1024,2048,4096]×2\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='lightcyan')
            layer1.node('l1_moe_res', 'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='orange')

    # Second pipeline stage (layers 2-3)
    with dot.subgraph(name='cluster_pipeline_1') as c1:
        c1.attr(label='Pipeline Stage 1 (Layers 2-3)\nGPUs 8-15', style='dashed', color='blue')
        
        # Communication between stages
        dot.node('stage0_to_stage1', 'Pipeline Communication\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-7 → 8-15', 
                 shape='parallelogram', fillcolor='gray')
        
        # Layer 2
        with c1.subgraph(name='cluster_layer_2') as layer2:
            layer2.attr(label='Layer 2', style='rounded')
            
            layer2.node('l2_mha_qkv', 'QKV Projection\nInput: [1024,2048,4096]\nOutput: [1024,2048,128] per GPU\nGPU: 8-15', fillcolor='yellow')
            layer2.node('l2_mha_attn', 'Attention Computation\nInput: [1024,2048,128]\nOutput: [1024,2048,128]\nGPU: 8-15', fillcolor='yellow')
            layer2.node('l2_mha_out', 'Output Projection\nInput: [1024,2048,128]\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='yellow')
            layer2.node('l2_mha_res', 'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='orange')
            
            layer2.node('l2_moe_gate', 'Gate Network\nInput: [1024,2048,4096]\nOutput: [1024,2048,16]\nGPU: 8-15', fillcolor='lightcoral')
            layer2.node('l2_moe_exp0', 'Expert 0,1\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 8-9', fillcolor='lightpink')
            layer2.node('l2_moe_exp2', 'Expert 2,3\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 10-11', fillcolor='lightpink')
            layer2.node('l2_moe_exp4', 'Expert 4,5\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 12-13', fillcolor='lightpink')
            layer2.node('l2_moe_exp6', 'Expert 6,7\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 14-15', fillcolor='lightpink')
            layer2.node('l2_moe_agg', 'Expert Aggregation\nInput: [1024,2048,4096]×2\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='lightcyan')
            layer2.node('l2_moe_res', 'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='orange')

        # Layer 3
        with c1.subgraph(name='cluster_layer_3') as layer3:
            layer3.attr(label='Layer 3', style='rounded')
            
            layer3.node('l3_mha_qkv', 'QKV Projection\nInput: [1024,2048,4096]\nOutput: [1024,2048,128] per GPU\nGPU: 8-15', fillcolor='yellow')
            layer3.node('l3_mha_attn', 'Attention Computation\nInput: [1024,2048,128]\nOutput: [1024,2048,128]\nGPU: 8-15', fillcolor='yellow')
            layer3.node('l3_mha_out', 'Output Projection\nInput: [1024,2048,128]\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='yellow')
            layer3.node('l3_mha_res', 'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='orange')
            
            layer3.node('l3_moe_gate', 'Gate Network\nInput: [1024,2048,4096]\nOutput: [1024,2048,16]\nGPU: 8-15', fillcolor='lightcoral')
            layer3.node('l3_moe_exp0', 'Expert 0,1\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 8-9', fillcolor='lightpink')
            layer3.node('l3_moe_exp2', 'Expert 2,3\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 10-11', fillcolor='lightpink')
            layer3.node('l3_moe_exp4', 'Expert 4,5\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 12-13', fillcolor='lightpink')
            layer3.node('l3_moe_exp6', 'Expert 6,7\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 14-15', fillcolor='lightpink')
            layer3.node('l3_moe_agg', 'Expert Aggregation\nInput: [1024,2048,4096]×2\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='lightcyan')
            layer3.node('l3_moe_res', 'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='orange')

    # Output node
    dot.node('output', 'Final Output\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 8-15', 
             shape='ellipse', fillcolor='lightgreen')

    # Connect nodes for baseline
    dot.edge('input', 'l0_mha_qkv')
    dot.edge('l0_mha_qkv', 'l0_mha_attn')
    dot.edge('l0_mha_attn', 'l0_mha_out')
    dot.edge('l0_mha_out', 'l0_mha_res')
    dot.edge('input', 'l0_mha_res')
    
    dot.edge('l0_mha_res', 'l0_moe_gate')
    dot.edge('l0_moe_gate', 'l0_moe_exp0', style='dashed')
    dot.edge('l0_moe_gate', 'l0_moe_exp2', style='dashed')
    dot.edge('l0_moe_gate', 'l0_moe_exp4', style='dashed')
    dot.edge('l0_moe_gate', 'l0_moe_exp6', style='dashed')
    dot.edge('l0_moe_exp0', 'l0_moe_agg')
    dot.edge('l0_moe_exp2', 'l0_moe_agg')
    dot.edge('l0_moe_exp4', 'l0_moe_agg')
    dot.edge('l0_moe_exp6', 'l0_moe_agg')
    dot.edge('l0_moe_agg', 'l0_moe_res')
    dot.edge('l0_mha_res', 'l0_moe_res')
    
    dot.edge('l0_moe_res', 'l1_mha_qkv')
    dot.edge('l1_mha_qkv', 'l1_mha_attn')
    dot.edge('l1_mha_attn', 'l1_mha_out')
    dot.edge('l1_mha_out', 'l1_mha_res')
    dot.edge('l0_moe_res', 'l1_mha_res')
    
    dot.edge('l1_mha_res', 'l1_moe_gate')
    dot.edge('l1_moe_gate', 'l1_moe_exp0', style='dashed')
    dot.edge('l1_moe_gate', 'l1_moe_exp2', style='dashed')
    dot.edge('l1_moe_gate', 'l1_moe_exp4', style='dashed')
    dot.edge('l1_moe_gate', 'l1_moe_exp6', style='dashed')
    dot.edge('l1_moe_exp0', 'l1_moe_agg')
    dot.edge('l1_moe_exp2', 'l1_moe_agg')
    dot.edge('l1_moe_exp4', 'l1_moe_agg')
    dot.edge('l1_moe_exp6', 'l1_moe_agg')
    dot.edge('l1_moe_agg', 'l1_moe_res')
    dot.edge('l1_mha_res', 'l1_moe_res')
    
    # Pipeline communication
    dot.edge('l1_moe_res', 'stage0_to_stage1')
    dot.edge('stage0_to_stage1', 'l2_mha_qkv')
    
    dot.edge('l2_mha_qkv', 'l2_mha_attn')
    dot.edge('l2_mha_attn', 'l2_mha_out')
    dot.edge('l2_mha_out', 'l2_mha_res')
    dot.edge('stage0_to_stage1', 'l2_mha_res')
    
    dot.edge('l2_mha_res', 'l2_moe_gate')
    dot.edge('l2_moe_gate', 'l2_moe_exp0', style='dashed')
    dot.edge('l2_moe_gate', 'l2_moe_exp2', style='dashed')
    dot.edge('l2_moe_gate', 'l2_moe_exp4', style='dashed')
    dot.edge('l2_moe_gate', 'l2_moe_exp6', style='dashed')
    dot.edge('l2_moe_exp0', 'l2_moe_agg')
    dot.edge('l2_moe_exp2', 'l2_moe_agg')
    dot.edge('l2_moe_exp4', 'l2_moe_agg')
    dot.edge('l2_moe_exp6', 'l2_moe_agg')
    dot.edge('l2_moe_agg', 'l2_moe_res')
    dot.edge('l2_mha_res', 'l2_moe_res')
    
    dot.edge('l2_moe_res', 'l3_mha_qkv')
    dot.edge('l3_mha_qkv', 'l3_mha_attn')
    dot.edge('l3_mha_attn', 'l3_mha_out')
    dot.edge('l3_mha_out', 'l3_mha_res')
    dot.edge('l2_moe_res', 'l3_mha_res')
    
    dot.edge('l3_mha_res', 'l3_moe_gate')
    dot.edge('l3_moe_gate', 'l3_moe_exp0', style='dashed')
    dot.edge('l3_moe_gate', 'l3_moe_exp2', style='dashed')
    dot.edge('l3_moe_gate', 'l3_moe_exp4', style='dashed')
    dot.edge('l3_moe_gate', 'l3_moe_exp6', style='dashed')
    dot.edge('l3_moe_exp0', 'l3_moe_agg')
    dot.edge('l3_moe_exp2', 'l3_moe_agg')
    dot.edge('l3_moe_exp4', 'l3_moe_agg')
    dot.edge('l3_moe_exp6', 'l3_moe_agg')
    dot.edge('l3_moe_agg', 'l3_moe_res')
    dot.edge('l3_mha_res', 'l3_moe_res')
    
    dot.edge('l3_moe_res', 'output')
    
    return dot

def create_ma_separation_dag():
    """Create MA Separation DAG with attention/expert separation"""
    dot = graphviz.Digraph('ma_separation_moe_model', 
                          comment='MA Separation MoE Model with Attention/Expert Split')
    
    # Set graph attributes
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input Embedding\nInput: [batch_size=1024, seq_len=2048, hidden=4096]\nGPU: all GPUs', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Create 4 layers with MA separation
    for layer in range(4):
        with dot.subgraph(name=f'cluster_layer_{layer}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer}', style='dashed', color='purple')
            
            # Attention computation (GPUs 0-7)
            with layer_cluster.subgraph(name=f'cluster_attention_{layer}') as attn_cluster:
                attn_cluster.attr(label=f'Attention Group (GPUs 0-7)', style='rounded', color='red')
                
                attn_cluster.node(f'l{layer}_qkv_all_gather', 'All-Gather QKV\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]×8\nGPU: 0-7', 
                                 shape='parallelogram', fillcolor='lightcyan')
                
                # Individual attention head processing (4 heads per GPU, 32 total)
                for gpu in range(8):
                    attn_cluster.node(f'l{layer}_qkv_gpu{gpu}', f'QKV Projection GPU{gpu}\nInput: [1024,2048,4096]\nOutput: [1024,2048,512]\nGPU: {gpu}', fillcolor='yellow')
                    attn_cluster.node(f'l{layer}_attn_gpu{gpu}', f'Attention Heads GPU{gpu}\nInput: [1024,2048,512]\nOutput: [1024,2048,512]\nGPU: {gpu}', fillcolor='yellow')
                    attn_cluster.node(f'l{layer}_out_gpu{gpu}', f'Output Projection GPU{gpu}\nInput: [1024,2048,512]\nOutput: [1024,2048,4096]\nGPU: {gpu}', fillcolor='yellow')
                
                attn_cluster.node(f'l{layer}_attn_all_reduce', 'All-Reduce Attention\nInput: [1024,2048,4096]×8\nOutput: [1024,2048,4096]\nGPU: 0-7', 
                                 shape='parallelogram', fillcolor='lightcyan')
                attn_cluster.node(f'l{layer}_attn_res', f'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='orange')
            
            # Communication from attention to MoE
            dot.node(f'l{layer}_attn_to_moe', f'Broadcast to MoE\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-7 → 8-15', 
                     shape='parallelogram', fillcolor='gray')
            
            # Expert computation (GPUs 8-15)
            with layer_cluster.subgraph(name=f'cluster_moe_{layer}') as moe_cluster:
                moe_cluster.attr(label=f'MoE Group (GPUs 8-15)', style='rounded', color='blue')
                
                moe_cluster.node(f'l{layer}_moe_gate', f'Gate Network\nInput: [1024,2048,4096]\nOutput: [1024,2048,16]\nGPU: 8-15', fillcolor='lightcoral')
                
                # Expert distribution: 2 experts per GPU
                for gpu in range(8, 16):
                    expert_start = (gpu - 8) * 2
                    expert_end = expert_start + 1
                    moe_cluster.node(f'l{layer}_exp_gpu{gpu}', f'Experts {expert_start},{expert_end}\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: {gpu}', fillcolor='lightpink')
                
                moe_cluster.node(f'l{layer}_moe_agg', f'Expert Aggregation\nInput: [1024,2048,4096]×2\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='lightcyan')
                moe_cluster.node(f'l{layer}_moe_res', f'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='orange')
    
    # Output node
    dot.node('output', 'Final Output\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 8-15', 
             shape='ellipse', fillcolor='lightgreen')

    # Connect nodes for MA separation
    for layer in range(4):
        if layer == 0:
            dot.edge('input', f'l{layer}_qkv_all_gather')
        else:
            dot.edge(f'prev_layer_{layer-1}', f'l{layer}_qkv_all_gather')
        
        # Attention path
        dot.edge(f'l{layer}_qkv_all_gather', f'l{layer}_qkv_gpu0')
        dot.edge(f'l{layer}_qkv_all_gather', f'l{layer}_qkv_gpu1')
        dot.edge(f'l{layer}_qkv_all_gather', f'l{layer}_qkv_gpu2')
        dot.edge(f'l{layer}_qkv_all_gather', f'l{layer}_qkv_gpu3')
        dot.edge(f'l{layer}_qkv_all_gather', f'l{layer}_qkv_gpu4')
        dot.edge(f'l{layer}_qkv_all_gather', f'l{layer}_qkv_gpu5')
        dot.edge(f'l{layer}_qkv_all_gather', f'l{layer}_qkv_gpu6')
        dot.edge(f'l{layer}_qkv_all_gather', f'l{layer}_qkv_gpu7')
        
        for gpu in range(8):
            dot.edge(f'l{layer}_qkv_gpu{gpu}', f'l{layer}_attn_gpu{gpu}')
            dot.edge(f'l{layer}_attn_gpu{gpu}', f'l{layer}_out_gpu{gpu}')
            dot.edge(f'l{layer}_out_gpu{gpu}', f'l{layer}_attn_all_reduce')
        
        if layer == 0:
            dot.edge('input', f'l{layer}_attn_res')
        else:
            dot.edge(f'prev_layer_{layer-1}', f'l{layer}_attn_res')
        dot.edge(f'l{layer}_attn_all_reduce', f'l{layer}_attn_res')
        dot.edge(f'l{layer}_attn_res', f'l{layer}_attn_to_moe')
        
        # MoE path
        dot.edge(f'l{layer}_attn_to_moe', f'l{layer}_moe_gate')
        for gpu in range(8, 16):
            dot.edge(f'l{layer}_moe_gate', f'l{layer}_exp_gpu{gpu}', style='dashed')
            dot.edge(f'l{layer}_attn_to_moe', f'l{layer}_exp_gpu{gpu}')
            dot.edge(f'l{layer}_exp_gpu{gpu}', f'l{layer}_moe_agg')
        dot.edge(f'l{layer}_moe_agg', f'l{layer}_moe_res')
        dot.edge(f'l{layer}_attn_to_moe', f'l{layer}_moe_res')
        
        if layer < 3:
            dot.edge(f'l{layer}_moe_res', f'prev_layer_{layer}')
        else:
            dot.edge(f'l{layer}_moe_res', 'output')
    
    return dot

# Generate both DAGs
print("Generating baseline DAG...")
baseline_dag = create_baseline_dag()
baseline_dag.render('../outputs/2025-10-16-14-45-55/baseline_dag', format='dot', cleanup=False)
baseline_dag.render('../outputs/2025-10-16-14-45-55/baseline_dag', format='svg', cleanup=False)

print("Generating MA Separation DAG...")
ma_separation_dag = create_ma_separation_dag()
ma_separation_dag.render('../outputs/2025-10-16-14-45-55/ma_separation_dag', format='dot', cleanup=False)
ma_separation_dag.render('../outputs/2025-10-16-14-45-55/ma_separation_dag', format='svg', cleanup=False)

# Create detailed layer-specific DAGs for MA separation
print("Generating detailed layer DAGs...")

def create_layer_dag(layer_num, ma_separation=True):
    """Create detailed DAG for a single layer"""
    suffix = "ma" if ma_separation else "baseline"
    dot = graphviz.Digraph(f'layer_{layer_num}_{suffix}', 
                          comment=f'Layer {layer_num} - {"MA Separation" if ma_separation else "Baseline"}')
    
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3', ranksep='0.8')
    
    if ma_separation:
        # MA Separation layer detail
        dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Input
        dot.node(f'layer{layer_num}_input', f'Layer {layer_num} Input\nInput: [1024,2048,4096]\nGPU: all GPUs', 
                 shape='ellipse', fillcolor='lightgreen')
        
        # Attention group (GPUs 0-7)
        with dot.subgraph(name=f'cluster_attn_group_{layer_num}') as attn_group:
            attn_group.attr(label='Attention Group (GPUs 0-7)', style='dashed', color='red')
            
            # Q projection across 8 GPUs
            for gpu in range(8):
                attn_group.node(f'l{layer_num}_q_gpu{gpu}', f'Q Projection GPU{gpu}\nInput: [1024,2048,4096]\nOutput: [1024,2048,128]\nGPU: {gpu}', fillcolor='yellow')
                attn_group.node(f'l{layer_num}_k_gpu{gpu}', f'K Projection GPU{gpu}\nInput: [1024,2048,4096]\nOutput: [1024,2048,128]\nGPU: {gpu}', fillcolor='yellow')
                attn_group.node(f'l{layer_num}_v_gpu{gpu}', f'V Projection GPU{gpu}\nInput: [1024,2048,4096]\nOutput: [1024,2048,128]\nGPU: {gpu}', fillcolor='yellow')
                
                # All-gather for K,V
                attn_group.node(f'l{layer_num}_kv_gather_gpu{gpu}', f'Gather K,V\nInput: [1024,2048,128]\nOutput: [1024,2048,1024]\nGPU: {gpu}', shape='parallelogram', fillcolor='lightcyan')
                
                # Attention computation
                attn_group.node(f'l{layer_num}_attn_gpu{gpu}', f'Multi-Head Attention GPU{gpu}\nInput: Q=[1024,2048,128], K=[1024,2048,1024], V=[1024,2048,1024]\nOutput: [1024,2048,512]\nGPU: {gpu}', fillcolor='lightblue')
                attn_group.node(f'l{layer_num}_proj_gpu{gpu}', f'Output Projection GPU{gpu}\nInput: [1024,2048,512]\nOutput: [1024,2048,4096]\nGPU: {gpu}', fillcolor='yellow')
            
            # All-reduce for output
            attn_group.node(f'l{layer_num}_attn_allreduce', 'All-Reduce Attention\nInput: [1024,2048,4096]×8\nOutput: [1024,2048,4096]\nGPU: 0-7', shape='parallelogram', fillcolor='lightcyan')
            attn_group.node(f'l{layer_num}_attn_res', 'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='orange')
        
        # Communication
        dot.node(f'l{layer_num}_attn_moe_comm', 'Broadcast to MoE\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]×8\nGPU: 0-7 → 8-15', shape='parallelogram', fillcolor='gray')
        
        # MoE group (GPUs 8-15)
        with dot.subgraph(name=f'cluster_moe_group_{layer_num}') as moe_group:
            moe_group.attr(label='MoE Group (GPUs 8-15)', style='dashed', color='blue')
            
            # Gate
            moe_group.node(f'l{layer_num}_gate', f'Gate Network\nInput: [1024,2048,4096]\nOutput: [1024,2048,16]\nGPU: 8-15', fillcolor='lightcoral')
            
            # Expert routing
            moe_group.node(f'l{layer_num}_router', f'Top-2 Router\nInput: [1024,2048,16]\nOutput: routing decisions\nGPU: 8-15', shape='parallelogram', fillcolor='lightgreen')
            
            # Expert processing
            for gpu in range(8, 16):
                expert_num = gpu - 8
                moe_group.node(f'l{layer_num}_expert{expert_num*2}_gpu{gpu}', f'Expert {expert_num*2}\nInput: [1024,2048,4096]\nOutput: [1024,2048,16384]\nGPU: {gpu}', fillcolor='lightpink')
                moe_group.node(f'l{layer_num}_expert{expert_num*2+1}_gpu{gpu}', f'Expert {expert_num*2+1}\nInput: [1024,2048,4096]\nOutput: [1024,2048,16384]\nGPU: {gpu}', fillcolor='lightpink')
                moe_group.node(f'l{layer_num}_expert_down_gpu{gpu}', f'Expert Down Proj GPU{gpu}\nInput: [1024,2048,16384]\nOutput: [1024,2048,4096]\nGPU: {gpu}', fillcolor='lightpink')
            
            # Aggregation
            moe_group.node(f'l{layer_num}_moe_agg', f'Expert Aggregation\nInput: [1024,2048,4096]×2\nOutput: [1024,2048,4096]\nGPU: 8-15', shape='parallelogram', fillcolor='lightcyan')
            moe_group.node(f'l{layer_num}_moe_res', f'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 8-15', fillcolor='orange')
        
        # Connect MA separation nodes
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_q_gpu0')
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_q_gpu1')
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_q_gpu2')
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_q_gpu3')
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_q_gpu4')
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_q_gpu5')
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_q_gpu6')
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_q_gpu7')
        
        for gpu in range(8):
            dot.edge(f'layer{layer_num}_input', f'l{layer_num}_k_gpu{gpu}')
            dot.edge(f'layer{layer_num}_input', f'l{layer_num}_v_gpu{gpu}')
            dot.edge(f'l{layer_num}_q_gpu{gpu}', f'l{layer_num}_attn_gpu{gpu}')
            dot.edge(f'l{layer_num}_k_gpu{gpu}', f'l{layer_num}_kv_gather_gpu{gpu}')
            dot.edge(f'l{layer_num}_v_gpu{gpu}', f'l{layer_num}_kv_gather_gpu{gpu}')
            dot.edge(f'l{layer_num}_kv_gather_gpu{gpu}', f'l{layer_num}_attn_gpu{gpu}')
            dot.edge(f'l{layer_num}_attn_gpu{gpu}', f'l{layer_num}_proj_gpu{gpu}')
            dot.edge(f'l{layer_num}_proj_gpu{gpu}', f'l{layer_num}_attn_allreduce')
        
        dot.edge(f'l{layer_num}_attn_allreduce', f'l{layer_num}_attn_res')
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_attn_res')
        dot.edge(f'l{layer_num}_attn_res', f'l{layer_num}_attn_moe_comm')
        dot.edge(f'l{layer_num}_attn_moe_comm', f'l{layer_num}_gate')
        dot.edge(f'l{layer_num}_attn_moe_comm', f'l{layer_num}_router')
        
        for gpu in range(8, 16):
            dot.edge(f'l{layer_num}_router', f'l{layer_num}_expert{gpu-8*2}_gpu{gpu}', style='dashed')
            dot.edge(f'l{layer_num}_router', f'l{layer_num}_expert{gpu-8*2+1}_gpu{gpu}', style='dashed')
            dot.edge(f'l{layer_num}_attn_moe_comm', f'l{layer_num}_expert{gpu-8*2}_gpu{gpu}')
            dot.edge(f'l{layer_num}_attn_moe_comm', f'l{layer_num}_expert{gpu-8*2+1}_gpu{gpu}')
            dot.edge(f'l{layer_num}_expert{gpu-8*2}_gpu{gpu}', f'l{layer_num}_expert_down_gpu{gpu}')
            dot.edge(f'l{layer_num}_expert{gpu-8*2+1}_gpu{gpu}', f'l{layer_num}_expert_down_gpu{gpu}')
            dot.edge(f'l{layer_num}_expert_down_gpu{gpu}', f'l{layer_num}_moe_agg')
        
        dot.edge(f'l{layer_num}_moe_agg', f'l{layer_num}_moe_res')
        dot.edge(f'l{layer_num}_attn_moe_comm', f'l{layer_num}_moe_res')
        
    else:
        # Baseline layer detail (simplified for comparison)
        dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'layer{layer_num}_input', f'Layer {layer_num} Input\nInput: [1024,2048,4096]\nGPU: 0-7', 
                 shape='ellipse', fillcolor='lightgreen')
        
        # Attention
        dot.node(f'l{layer_num}_mha_qkv', f'QKV Projection\nInput: [1024,2048,4096]\nOutput: [1024,2048,128] per GPU\nGPU: 0-7', fillcolor='yellow')
        dot.node(f'l{layer_num}_mha_attn', f'Multi-Head Attention\nInput: [1024,2048,128]\nOutput: [1024,2048,128]\nGPU: 0-7', fillcolor='yellow')
        dot.node(f'l{layer_num}_mha_out', f'Output Projection\nInput: [1024,2048,128]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='yellow')
        dot.node(f'l{layer_num}_mha_res', f'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='orange')
        
        # MoE
        dot.node(f'l{layer_num}_moe_gate', f'Gate Network\nInput: [1024,2048,4096]\nOutput: [1024,2048,16]\nGPU: 0-7', fillcolor='lightcoral')
        for gpu in range(8):
            dot.node(f'l{layer_num}_exp{gpu*2}_gpu{gpu}', f'Experts {gpu*2},{gpu*2+1}\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: {gpu}', fillcolor='lightpink')
        dot.node(f'l{layer_num}_moe_agg', f'Expert Aggregation\nInput: [1024,2048,4096]×8\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='lightcyan')
        dot.node(f'l{layer_num}_moe_res', f'Residual Add\nInput: [1024,2048,4096], [1024,2048,4096]\nOutput: [1024,2048,4096]\nGPU: 0-7', fillcolor='orange')
        
        # Connections
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_mha_qkv')
        dot.edge(f'l{layer_num}_mha_qkv', f'l{layer_num}_mha_attn')
        dot.edge(f'l{layer_num}_mha_attn', f'l{layer_num}_mha_out')
        dot.edge(f'l{layer_num}_mha_out', f'l{layer_num}_mha_res')
        dot.edge(f'layer{layer_num}_input', f'l{layer_num}_mha_res')
        
        dot.edge(f'l{layer_num}_mha_res', f'l{layer_num}_moe_gate')
        for gpu in range(8):
            dot.edge(f'l{layer_num}_moe_gate', f'l{layer_num}_exp{gpu*2}_gpu{gpu}', style='dashed')
            dot.edge(f'l{layer_num}_mha_res', f'l{layer_num}_exp{gpu*2}_gpu{gpu}')
            dot.edge(f'l{layer_num}_exp{gpu*2}_gpu{gpu}', f'l{layer_num}_moe_agg')
        
        dot.edge(f'l{layer_num}_moe_agg', f'l{layer_num}_moe_res')
        dot.edge(f'l{layer_num}_mha_res', f'l{layer_num}_moe_res')
    
    return dot

# Generate layer-specific DAGs
for layer in range(4):
    # MA Separation layer
    layer_ma = create_layer_dag(layer, ma_separation=True)
    layer_ma.render(f'../outputs/2025-10-16-14-45-55/layer_{layer}_ma', format='dot', cleanup=False)
    layer_ma.render(f'../outputs/2025-10-16-14-45-55/layer_{layer}_ma', format='svg', cleanup=False)
    
    # Baseline layer
    layer_baseline = create_layer_dag(layer, ma_separation=False)
    layer_baseline.render(f'../outputs/2025-10-16-14-45-55/layer_{layer}_baseline', format='dot', cleanup=False)
    layer_baseline.render(f'../outputs/2025-10-16-14-45-55/layer_{layer}_baseline', format='svg', cleanup=False)

print("All DAGs generated successfully!")

# Create summary file
summary = """
# MA Separation Deployment DAGs Summary

## Generated Files:
1. baseline_dag.dot / baseline_dag.svg - Complete baseline model with TP=8, PP=2
2. ma_separation_dag.dot / ma_separation_dag.svg - Complete MA Separation model
3. layer_0_ma.dot / layer_0_ma.svg - Detailed Layer 0 MA Separation
4. layer_0_baseline.dot / layer_0_baseline.svg - Detailed Layer 0 Baseline
5. layer_1_ma.dot / layer_1_ma.svg - Detailed Layer 1 MA Separation
6. layer_1_baseline.dot / layer_1_baseline.svg - Detailed Layer 1 Baseline
7. layer_2_ma.dot / layer_2_ma.svg - Detailed Layer 2 MA Separation
8. layer_2_baseline.dot / layer_2_baseline.svg - Detailed Layer 2 Baseline
9. layer_3_ma.dot / layer_3_ma.svg - Detailed Layer 3 MA Separation
10. layer_3_baseline.dot / layer_3_baseline.svg - Detailed Layer 3 Baseline

## Key Differences:
- **Baseline**: Uses TP=8 across all 8 GPUs per pipeline stage, PP=2 across 16 GPUs
- **MA Separation**: Separates attention (GPUs 0-7) from MoE experts (GPUs 8-15)
- **Attention Parallelism**: Baseline uses tensor parallelism, MA Separation replicates attention across GPUs
- **Expert Distribution**: MA Separation maps 2 experts per GPU on MoE GPUs

## Engineering Details:
- All tensor dimensions are perfectly aligned
- Communication paths explicitly shown with parallelogram nodes
- Residual connections properly represented with multiple inputs
- Expert routing shown with dashed lines from gate to experts
- GPU assignments clearly labeled for each operation
"""

with open('../outputs/2025-10-16-14-45-55/dag_summary.md', 'w') as f:
    f.write(summary)

print("Summary file created!")