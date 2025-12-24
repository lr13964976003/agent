#!/usr/bin/env python3

import graphviz
import os

def create_llm_deployment_dag():
    """
    Generate a comprehensive DAG for LLM deployment with parallel strategies.
    Based on the deployment plan: PP=4, EP=4, TP=2, SP=2 (prefill), SP=1 (decode)
    """
    
    # Create main DAG
    dot = graphviz.Digraph(comment='LLM Parallel Strategy Deployment DAG')
    dot.attr(rankdir='TB', size='50,30', dpi='300')
    dot.attr('node', fontsize='10', margin='0.1,0.05')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # GPU assignment: PP(4) × EP(4) × TP(2) × SP(2) = 64 GPUs for prefill
    # For decode: PP(4) × EP(4) × TP(2) = 32 GPUs
    
    # Let's create the DAG structure
    # I'll organize by pipeline stages, then show the parallelism within each stage
    
    # === PREFILL PHASE ===
    dot.node('prefill_start', 'PREFILL PHASE\\nPP=4, EP=4, TP=2, SP=2\\n64 GPUs Total', 
             shape='box', style='filled', fillcolor='red', fontsize='14')
    
    # Input node
    dot.node('input', 'INPUT\\nInput: [batch_size=4, seq_len=10240, hidden=512]\\nOutput: [batch_size=4, seq_len=10240, hidden=512]', 
             shape='ellipse', style='filled', fillcolor='white', penwidth='3')
    
    # === PIPELINE STAGE 1 (Layers 1-4) ===
    dot.node('stage1_label', 'PIPELINE STAGE 1\\nLayers 1-4\\nGPUs: 0-15', 
             shape='box', style='filled', fillcolor='purple', fontsize='12')
    
    # Sequence Parallel Split
    dot.node('sp_split_1', 'SP SPLIT\\nInput: [batch=4, seq=10240, hidden=512]\\nOutput: [batch=4, seq=5120, hidden=512]\\nGPU: 0-15', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Layer 1 - Attention with TP and SP
    dot.node('layernorm1_1', 'LayerNorm\\nInput: [batch=4, seq=5120, hidden=512]\\nOutput: [batch=4, seq=5120, hidden=512]\\nGPU: 0-15', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('attn_qkv1_1', 'Attention QKV Proj\\nTP=2\\nInput: [batch=4, seq=5120, hidden=512]\\nOutput: [batch=4, seq=5120, hidden=256]\\nGPU: 0-7', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_qkv1_2', 'Attention QKV Proj\\nTP=2\\nInput: [batch=4, seq=5120, hidden=512]\\nOutput: [batch=4, seq=5120, hidden=256]\\nGPU: 8-15', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Attention computation with SP
    dot.node('attn_compute1_1', 'Attention Compute\\nSP=2\\nInput: [batch=4, seq=5120, hidden=256]\\nOutput: [batch=4, seq=5120, hidden=256]\\nGPU: 0-7', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_compute1_2', 'Attention Compute\\nSP=2\\nInput: [batch=4, seq=5120, hidden=256]\\nOutput: [batch=4, seq=5120, hidden=256]\\nGPU: 8-15', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # All-gather for attention output
    dot.node('attn_gather1', 'All-Gather\\nTP Attention\\nInput: [batch=4, seq=5120, hidden=256]\\nOutput: [batch=4, seq=5120, hidden=512]\\nGPU: 0-15', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # MLP with MoE
    dot.node('mlp_gate1', 'MLP Gate\\nInput: [batch=4, seq=5120, hidden=512]\\nOutput: [batch=4, seq=5120, num_experts=16]\\nGPU: 0-15', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Expert routing (dashed line for gate selection)
    dot.node('expert_route1', 'Expert Routing\\nTop-2 Selection\\nGPU: 0-15', 
             shape='parallelogram', style='filled, dashed', fillcolor='orange')
    
    # Expert computation with EP=4
    dot.node('expert1_1', 'Expert 0-3\\nEP=4\\nInput: [batch=4, seq=1280, hidden=512]\\nOutput: [batch=4, seq=1280, hidden=512]\\nGPU: 0-3', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert1_2', 'Expert 4-7\\nEP=4\\nInput: [batch=4, seq=1280, hidden=512]\\nOutput: [batch=4, seq=1280, hidden=512]\\nGPU: 4-7', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert1_3', 'Expert 8-11\\nEP=4\\nInput: [batch=4, seq=1280, hidden=512]\\nOutput: [batch=4, seq=1280, hidden=512]\\nGPU: 8-11', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert1_4', 'Expert 12-15\\nEP=4\\nInput: [batch=4, seq=1280, hidden=512]\\nOutput: [batch=4, seq=1280, hidden=512]\\nGPU: 12-15', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # All-to-All for expert aggregation
    dot.node('expert_all2all1', 'All-to-All\\nExpert Aggregation\\nInput: [batch=4, seq=1280, hidden=512]\\nOutput: [batch=4, seq=5120, hidden=512]\\nGPU: 0-15', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # SP Gather at end of stage
    dot.node('sp_gather_1', 'SP GATHER\\nInput: [batch=4, seq=5120, hidden=512]\\nOutput: [batch=4, seq=10240, hidden=512]\\nGPU: 0-15', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # === PIPELINE STAGE 2 (Layers 5-8) ===
    dot.node('stage2_label', 'PIPELINE STAGE 2\\nLayers 5-8\\nGPUs: 16-31', 
             shape='box', style='filled', fillcolor='purple', fontsize='12')
    
    # Similar structure for stage 2
    dot.node('sp_split_2', 'SP SPLIT\\nGPU: 16-31', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    dot.node('layernorm2', 'LayerNorm\\nGPU: 16-31', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('attn_qkv2_1', 'Attention QKV Proj\\nTP=2\\nGPU: 16-23', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_qkv2_2', 'Attention QKV Proj\\nTP=2\\nGPU: 24-31', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('attn_compute2_1', 'Attention Compute\\nSP=2\\nGPU: 16-23', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_compute2_2', 'Attention Compute\\nSP=2\\nGPU: 24-31', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('attn_gather2', 'All-Gather\\nTP Attention\\nGPU: 16-31', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # MLP with MoE for stage 2
    dot.node('mlp_gate2', 'MLP Gate\\nGPU: 16-31', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('expert_route2', 'Expert Routing\\nTop-2 Selection\\nGPU: 16-31', 
             shape='parallelogram', style='filled, dashed', fillcolor='orange')
    
    dot.node('expert2_1', 'Expert 0-3\\nEP=4\\nGPU: 16-19', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert2_2', 'Expert 4-7\\nEP=4\\nGPU: 20-23', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert2_3', 'Expert 8-11\\nEP=4\\nGPU: 24-27', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert2_4', 'Expert 12-15\\nEP=4\\nGPU: 28-31', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('expert_all2all2', 'All-to-All\\nExpert Aggregation\\nGPU: 16-31', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    dot.node('sp_gather_2', 'SP GATHER\\nGPU: 16-31', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # === PIPELINE STAGE 3 (Layers 9-12) ===
    dot.node('stage3_label', 'PIPELINE STAGE 3\\nLayers 9-12\\nGPUs: 32-47', 
             shape='box', style='filled', fillcolor='purple', fontsize='12')
    
    dot.node('sp_split_3', 'SP SPLIT\\nGPU: 32-47', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dot.node('layernorm3', 'LayerNorm\\nGPU: 32-47', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_qkv3_1', 'Attention QKV Proj\\nTP=2\\nGPU: 32-39', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_qkv3_2', 'Attention QKV Proj\\nTP=2\\nGPU: 40-47', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_compute3_1', 'Attention Compute\\nSP=2\\nGPU: 32-39', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_compute3_2', 'Attention Compute\\nSP=2\\nGPU: 40-47', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_gather3', 'All-Gather\\nTP Attention\\nGPU: 32-47', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    dot.node('mlp_gate3', 'MLP Gate\\nGPU: 32-47', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert_route3', 'Expert Routing\\nTop-2 Selection\\nGPU: 32-47', 
             shape='parallelogram', style='filled, dashed', fillcolor='orange')
    dot.node('expert3_1', 'Expert 0-3\\nEP=4\\nGPU: 32-35', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert3_2', 'Expert 4-7\\nEP=4\\nGPU: 36-39', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert3_3', 'Expert 8-11\\nEP=4\\nGPU: 40-43', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert3_4', 'Expert 12-15\\nEP=4\\nGPU: 44-47', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert_all2all3', 'All-to-All\\nExpert Aggregation\\nGPU: 32-47', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    dot.node('sp_gather_3', 'SP GATHER\\nGPU: 32-47', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # === PIPELINE STAGE 4 (Layers 13-16) ===
    dot.node('stage4_label', 'PIPELINE STAGE 4\\nLayers 13-16\\nGPUs: 48-63', 
             shape='box', style='filled', fillcolor='purple', fontsize='12')
    
    dot.node('sp_split_4', 'SP SPLIT\\nGPU: 48-63', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dot.node('layernorm4', 'LayerNorm\\nGPU: 48-63', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_qkv4_1', 'Attention QKV Proj\\nTP=2\\nGPU: 48-55', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_qkv4_2', 'Attention QKV Proj\\nTP=2\\nGPU: 56-63', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_compute4_1', 'Attention Compute\\nSP=2\\nGPU: 48-55', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_compute4_2', 'Attention Compute\\nSP=2\\nGPU: 56-63', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('attn_gather4', 'All-Gather\\nTP Attention\\nGPU: 48-63', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    dot.node('mlp_gate4', 'MLP Gate\\nGPU: 48-63', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert_route4', 'Expert Routing\\nTop-2 Selection\\nGPU: 48-63', 
             shape='parallelogram', style='filled, dashed', fillcolor='orange')
    dot.node('expert4_1', 'Expert 0-3\\nEP=4\\nGPU: 48-51', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert4_2', 'Expert 4-7\\nEP=4\\nGPU: 52-55', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert4_3', 'Expert 8-11\\nEP=4\\nGPU: 56-59', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert4_4', 'Expert 12-15\\nEP=4\\nGPU: 60-63', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('expert_all2all4', 'All-to-All\\nExpert Aggregation\\nGPU: 48-63', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    dot.node('sp_gather_4', 'SP GATHER\\nGPU: 48-63', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Final output
    dot.node('final_output', 'FINAL OUTPUT\\nInput: [batch=4, seq=10240, hidden=512]\\nOutput: [batch=4, seq=10240, hidden=512]', 
             shape='ellipse', style='filled', fillcolor='white', penwidth='3')
    
    # === DECODE PHASE ===
    dot.node('decode_start', 'DECODE PHASE\\nPP=4, EP=4, TP=2, SP=1\\n32 GPUs Total', 
             shape='box', style='filled', fillcolor='red', fontsize='14')
    
    # Decode input (single token)
    dot.node('decode_input', 'DECODE INPUT\\nSingle Token\\nInput: [batch=4, seq=1, hidden=512]\\nOutput: [batch=4, seq=1, hidden=512]', 
             shape='ellipse', style='filled', fillcolor='pink', penwidth='3')
    
    # Decode stage 1 (similar structure but no SP)
    dot.node('decode_stage1', 'DECODE STAGE 1\\nLayers 1-4\\nGPUs: 0-15', 
             shape='box', style='filled', fillcolor='purple', fontsize='12')
    
    dot.node('decode_layernorm1', 'LayerNorm\\nInput: [batch=4, seq=1, hidden=512]\\nOutput: [batch=4, seq=1, hidden=512]\\nGPU: 0-15', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('decode_attn_qkv1_1', 'Attention QKV Proj\\nTP=2\\nGPU: 0-7', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('decode_attn_qkv1_2', 'Attention QKV Proj\\nTP=2\\nGPU: 8-15', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('decode_attn1_1', 'Attention Compute\\nNo SP\\nGPU: 0-7', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('decode_attn1_2', 'Attention Compute\\nNo SP\\nGPU: 8-15', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('decode_attn_gather1', 'All-Gather\\nTP Attention\\nGPU: 0-15', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Decode MLP (same experts but smaller workload)
    dot.node('decode_mlp_gate1', 'MLP Gate\\nGPU: 0-15', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('decode_expert_route1', 'Expert Routing\\nTop-2 Selection\\nGPU: 0-15', 
             shape='parallelogram', style='filled, dashed', fillcolor='orange')
    
    dot.node('decode_expert1_1', 'Expert 0-3\\nEP=4\\nGPU: 0-3', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('decode_expert1_2', 'Expert 4-7\\nEP=4\\nGPU: 4-7', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('decode_expert1_3', 'Expert 8-11\\nEP=4\\nGPU: 8-11', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.node('decode_expert1_4', 'Expert 12-15\\nEP=4\\nGPU: 12-15', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('decode_expert_all2all1', 'All-to-All\\nExpert Aggregation\\nGPU: 0-15', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    dot.node('decode_output1', 'STAGE 1 OUTPUT\\nGPU: 0-15', 
             shape='ellipse', style='filled', fillcolor='lightcyan')
    
    # === EDGES ===
    # Input to prefill
    dot.edge('input', 'sp_split_1')
    
    # Stage 1 flow
    dot.edge('sp_split_1', 'layernorm1_1')
    dot.edge('layernorm1_1', 'attn_qkv1_1')
    dot.edge('layernorm1_1', 'attn_qkv1_2')
    dot.edge('attn_qkv1_1', 'attn_compute1_1')
    dot.edge('attn_qkv1_2', 'attn_compute1_2')
    dot.edge('attn_compute1_1', 'attn_gather1')
    dot.edge('attn_compute1_2', 'attn_gather1')
    dot.edge('attn_gather1', 'mlp_gate1')
    dot.edge('mlp_gate1', 'expert_route1')
    dot.edge('expert_route1', 'expert1_1', style='dashed')
    dot.edge('expert_route1', 'expert1_2', style='dashed')
    dot.edge('expert_route1', 'expert1_3', style='dashed')
    dot.edge('expert_route1', 'expert1_4', style='dashed')
    dot.edge('expert1_1', 'expert_all2all1')
    dot.edge('expert1_2', 'expert_all2all1')
    dot.edge('expert1_3', 'expert_all2all1')
    dot.edge('expert1_4', 'expert_all2all1')
    dot.edge('expert_all2all1', 'sp_gather_1')
    
    # Pipeline stages
    dot.edge('sp_gather_1', 'sp_split_2', label='Pipeline\\nStage 1→2', penwidth='2')
    dot.edge('sp_gather_2', 'sp_split_3', label='Pipeline\\nStage 2→3', penwidth='2')
    dot.edge('sp_gather_3', 'sp_split_4', label='Pipeline\\nStage 3→4', penwidth='2')
    dot.edge('sp_gather_4', 'final_output')
    
    # Decode flow
    dot.edge('decode_input', 'decode_layernorm1')
    dot.edge('decode_layernorm1', 'decode_attn_qkv1_1')
    dot.edge('decode_layernorm1', 'decode_attn_qkv1_2')
    dot.edge('decode_attn_qkv1_1', 'decode_attn1_1')
    dot.edge('decode_attn_qkv1_2', 'decode_attn1_2')
    dot.edge('decode_attn1_1', 'decode_attn_gather1')
    dot.edge('decode_attn1_2', 'decode_attn_gather1')
    dot.edge('decode_attn_gather1', 'decode_mlp_gate1')
    dot.edge('decode_mlp_gate1', 'decode_expert_route1')
    dot.edge('decode_expert_route1', 'decode_expert1_1', style='dashed')
    dot.edge('decode_expert_route1', 'decode_expert1_2', style='dashed')
    dot.edge('decode_expert_route1', 'decode_expert1_3', style='dashed')
    dot.edge('decode_expert_route1', 'decode_expert1_4', style='dashed')
    dot.edge('decode_expert1_1', 'decode_expert_all2all1')
    dot.edge('decode_expert1_2', 'decode_expert_all2all1')
    dot.edge('decode_expert1_3', 'decode_expert_all2all1')
    dot.edge('decode_expert1_4', 'decode_expert_all2all1')
    dot.edge('decode_expert_all2all1', 'decode_output1')
    
    return dot

def main():
    # Create the DAG
    dag = create_llm_deployment_dag()
    
    # Save as DOT file
    dot_file = '../outputs/2025-12-24-12-00-33/llm_parallel_deployment.dot'
    dag.save(dot_file)
    
    # Render as SVG
    svg_file = '../outputs/2025-12-24-12-00-33/llm_parallel_deployment.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG saved as: {dot_file}")
    print(f"SVG saved as: {svg_file}")
    
    # Create a simpler version for better readability
    create_simplified_dag()

def create_simplified_dag():
    """Create a simplified version showing key concepts"""
    dot = graphviz.Digraph(comment='LLM Parallel Strategy - Simplified Overview')
    dot.attr(rankdir='LR', size='20,15', dpi='300')
    
    # High-level overview
    dot.node('input', 'Input\\n[batch=4, seq=10240, hidden=512]', 
             shape='ellipse', style='filled', fillcolor='white')
    
    dot.node('prefill', 'PREFILL PHASE\\nPP=4, EP=4, TP=2, SP=2\\n64 GPUs', 
             shape='box', style='filled', fillcolor='lightblue')
    
    dot.node('decode', 'DECODE PHASE\\nPP=4, EP=4, TP=2, SP=1\\n32 GPUs', 
             shape='box', style='filled', fillcolor='lightgreen')
    
    dot.node('output', 'Output\\nGenerated Tokens', 
             shape='ellipse', style='filled', fillcolor='white')
    
    # Communication patterns
    dot.node('comm', 'Communication Patterns:\\n• All-Reduce (TP)\\n• All-Gather (SP)\\n• All-to-All (EP)\\n• Pipeline (PP)', 
             shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Edges
    dot.edge('input', 'prefill')
    dot.edge('prefill', 'decode', label='KV Cache\\nTransfer')
    dot.edge('decode', 'output')
    
    # Save simplified version
    dot.save('../outputs/2025-12-24-12-00-33/llm_parallel_simplified.dot')
    dot.render('../outputs/2025-12-24-12-00-33/llm_parallel_simplified', format='svg', cleanup=True)

if __name__ == '__main__':
    main()