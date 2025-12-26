#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_parallelism_dag():
    """
    Generate a complete DAG for the hybrid expert-tensor-pipeline parallelism strategy.
    This represents a 64-GPU deployment with:
    - 8-way Expert Parallelism
    - 4-way Tensor Parallelism  
    - 2-way Pipeline Parallelism
    """
    
    # Create the main graph
    dot = Digraph(comment='Hybrid Expert-Tensor-Pipeline Parallelism DAG')
    dot.attr(rankdir='TB', size='200,100', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define styles for different node types
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='box', style='filled', fillcolor='lightgreen')     # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]', 
             shape='ellipse', fillcolor='lightgray')
    
    # Pipeline Stage 0: Layers 0-7
    with dot.subgraph(name='cluster_pipeline_0') as c:
        c.attr(label='Pipeline Stage 0 (Layers 0-7)', style='rounded,filled', fillcolor='lightcyan')
        
        # Expert Groups 0-3 (Pipeline Stage 0)
        for expert_group in range(4):
            with c.subgraph(name=f'cluster_expert_{expert_group}_p0') as ec:
                ec.attr(label=f'Expert Group {expert_group} (GPU {expert_group*4}-{expert_group*4+3})', 
                       style='rounded,filled', fillcolor='lightpink')
                
                # Expert routing (gate)
                ec.node(f'gate_e{expert_group}_p0', 
                       f'Gate Selection\\nGPU: {expert_group*4}-{expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]', 
                       shape='parallelogram')
                
                # Hierarchical All-to-All Communication
                ec.node(f'all2all_local_e{expert_group}_p0',
                       f'Local All-to-All\\nGPU: {expert_group*4}-{expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=2560, hidden=512]',
                       shape='ellipse')
                
                # Tensor parallel groups within expert
                for tensor_group in range(4):
                    gpu_id = expert_group * 4 + tensor_group
                    
                    # Attention computation
                    ec.node(f'attention_e{expert_group}_t{tensor_group}_p0',
                           f'Attention\\nGPU: {gpu_id}\\nInput: [batch_size=32, seq_len=2560, heads=4, d_k=32]\\nOutput: [batch_size=32, seq_len=2560, heads=4, d_k=32]',
                           shape='box')
                    
                    # MoE computation (2 experts per GPU)
                    for expert_id in range(2):
                        ec.node(f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p0',
                               f'MoE Expert {expert_id}\\nGPU: {gpu_id}\\nInput: [batch_size=16, seq_len=2560, hidden=256]\\nOutput: [batch_size=16, seq_len=2560, hidden=256]',
                               shape='box')
                    
                    # Tensor reduction
                    ec.node(f'tensor_reduce_e{expert_group}_t{tensor_group}_p0',
                           f'Tensor Reduction\\nGPU: {gpu_id}\\nInput: [batch_size=32, seq_len=2560, hidden=256]\\nOutput: [batch_size=32, seq_len=2560, hidden=256]',
                           shape='ellipse')
        
        # Expert Groups 4-7 (Pipeline Stage 0)
        for expert_group in range(4, 8):
            with c.subgraph(name=f'cluster_expert_{expert_group}_p0') as ec:
                ec.attr(label=f'Expert Group {expert_group} (GPU {expert_group*4}-{expert_group*4+3})', 
                       style='rounded,filled', fillcolor='lightpink')
                
                # Expert routing (gate)
                ec.node(f'gate_e{expert_group}_p0', 
                       f'Gate Selection\\nGPU: {expert_group*4}-{expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]', 
                       shape='parallelogram')
                
                # Hierarchical All-to-All Communication
                ec.node(f'all2all_local_e{expert_group}_p0',
                       f'Local All-to-All\\nGPU: {expert_group*4}-{expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=2560, hidden=512]',
                       shape='ellipse')
                
                # Tensor parallel groups within expert
                for tensor_group in range(4):
                    gpu_id = expert_group * 4 + tensor_group
                    
                    # Attention computation
                    ec.node(f'attention_e{expert_group}_t{tensor_group}_p0',
                           f'Attention\\nGPU: {gpu_id}\\nInput: [batch_size=32, seq_len=2560, heads=4, d_k=32]\\nOutput: [batch_size=32, seq_len=2560, heads=4, d_k=32]',
                           shape='box')
                    
                    # MoE computation (2 experts per GPU)
                    for expert_id in range(2):
                        ec.node(f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p0',
                               f'MoE Expert {expert_id}\\nGPU: {gpu_id}\\nInput: [batch_size=16, seq_len=2560, hidden=256]\\nOutput: [batch_size=16, seq_len=2560, hidden=256]',
                               shape='box')
                    
                    # Tensor reduction
                    ec.node(f'tensor_reduce_e{expert_group}_t{tensor_group}_p0',
                           f'Tensor Reduction\\nGPU: {gpu_id}\\nInput: [batch_size=32, seq_len=2560, hidden=256]\\nOutput: [batch_size=32, seq_len=2560, hidden=256]',
                           shape='ellipse')
    
    # Pipeline Stage 1: Layers 8-15
    with dot.subgraph(name='cluster_pipeline_1') as c:
        c.attr(label='Pipeline Stage 1 (Layers 8-15)', style='rounded,filled', fillcolor='lightsteelblue')
        
        # Expert Groups 0-3 (Pipeline Stage 1)
        for expert_group in range(4):
            with c.subgraph(name=f'cluster_expert_{expert_group}_p1') as ec:
                ec.attr(label=f'Expert Group {expert_group} (GPU {32+expert_group*4}-{32+expert_group*4+3})', 
                       style='rounded,filled', fillcolor='lightcoral')
                
                # Expert routing (gate)
                ec.node(f'gate_e{expert_group}_p1', 
                       f'Gate Selection\\nGPU: {32+expert_group*4}-{32+expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]', 
                       shape='parallelogram')
                
                # Hierarchical All-to-All Communication
                ec.node(f'all2all_local_e{expert_group}_p1',
                       f'Local All-to-All\\nGPU: {32+expert_group*4}-{32+expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=2560, hidden=512]',
                       shape='ellipse')
                
                # Tensor parallel groups within expert
                for tensor_group in range(4):
                    gpu_id = 32 + expert_group * 4 + tensor_group
                    
                    # Attention computation
                    ec.node(f'attention_e{expert_group}_t{tensor_group}_p1',
                           f'Attention\\nGPU: {gpu_id}\\nInput: [batch_size=32, seq_len=2560, heads=4, d_k=32]\\nOutput: [batch_size=32, seq_len=2560, heads=4, d_k=32]',
                           shape='box')
                    
                    # MoE computation (2 experts per GPU)
                    for expert_id in range(2):
                        ec.node(f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p1',
                               f'MoE Expert {expert_id}\\nGPU: {gpu_id}\\nInput: [batch_size=16, seq_len=2560, hidden=256]\\nOutput: [batch_size=16, seq_len=2560, hidden=256]',
                               shape='box')
                    
                    # Tensor reduction
                    ec.node(f'tensor_reduce_e{expert_group}_t{tensor_group}_p1',
                           f'Tensor Reduction\\nGPU: {gpu_id}\\nInput: [batch_size=32, seq_len=2560, hidden=256]\\nOutput: [batch_size=32, seq_len=2560, hidden=256]',
                           shape='ellipse')
        
        # Expert Groups 4-7 (Pipeline Stage 1)
        for expert_group in range(4, 8):
            with c.subgraph(name=f'cluster_expert_{expert_group}_p1') as ec:
                ec.attr(label=f'Expert Group {expert_group} (GPU {32+expert_group*4}-{32+expert_group*4+3})', 
                       style='rounded,filled', fillcolor='lightseagreen')
                
                # Expert routing (gate)
                ec.node(f'gate_e{expert_group}_p1', 
                       f'Gate Selection\\nGPU: {32+expert_group*4}-{32+expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=从一个文件读取内容。然后使用 Python 将其转换为图可视化代码。根据给定的并行策略部署方法文件，我需要生成一个完整的模型部署 DAG（有向无环图），并满足以下条件：

1. **完全反映并行策略**：包括 8 路专家并行、4 路张量并行和 2 路流水线并行。
2. **按 GPU 划分边界**：每个节点都要标注对应的 GPU。
3. **详细到算子级别**：每个层都要细化到具体的算子。
4. **图形表示**：椭圆表示通信，矩形表示计算，平行四边形表示路由/聚合。
5. **节点属性**：每个节点都要有输入和输出维度。
6. **完整通信行为**：所有 GPU 间通信行为都要完整反映。
7. **门控选择**：用虚线表示门控选择过程。

我已经读取了并行策略部署方法文件，现在将生成一个 Python 脚本来创建符合所有要求的 Graphviz DAG。