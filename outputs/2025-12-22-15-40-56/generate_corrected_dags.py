#!/usr/bin/env python3
"""
Corrected DAG Generator for MoE LLM Inference Deployments
Addresses all critical issues found in previous submissions
"""

import os
from graphviz import Digraph

def generate_current_deployment_dag():
    """Generate corrected DAG for EP64-TP8-PP2-DP2 deployment strategy"""
    
    dot = Digraph(comment='EP64-TP8-PP2-DP2 MoE LLM Inference DAG (Corrected)')
    dot.attr(bgcolor='white', fontname='Arial', rankdir='TB')
    dot.attr('node', fontname='Arial', shape='rectangle', style='filled')
    
    # Input node - DP splits the batch
    dot.node('input', 
             'Input\nInput: [batch_size=128, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]\nDP=2: Split batch into 2 groups of 64 sequences each',
             fillcolor='lightblue', shape='ellipse')
    
    # DP Split node
    dot.node('dp_split',
             'Data Parallelism Split\nGPU: DP0-DP1\nSplit batch_size=128 → 2×64\nNo communication required for inference',
             fillcolor='lightgreen', shape='parallelogram')
    
    # DP0 Branch
    dot.node('embed_dp0',
             'Embedding Layer DP0\nGPU: DP0_TP0-DP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=512]\nTP=8: hidden_size=4096 → 512 per TP rank',
             fillcolor='lightgreen')
    
    dot.node('hidden_up_dp0',
             'Hidden Upscaling DP0\nGPU: DP0_TP0-DP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=512]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]\nTP=8: All-Gather operation',
             fillcolor='lightgreen')
    
    dot.node('pp0_dp0_start',
             'PP Stage 0 DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nLayers 0-7 (8 layers)\n8 layers × 64 experts = 512 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    # Layer 0 Attention - DP0 (Complete attention mechanism)
    dot.node('layer0_norm_dp0',
             'Layer 0: RMSNorm DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_q_proj_dp0',
             'Layer 0: Q Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_k_proj_dp0',
             'Layer 0: K Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_v_proj_dp0',
             'Layer 0: V Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_qk_matmul_dp0',
             'Layer 0: QK^T MatMul DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, heads=32, d_k=128] x2\nOutput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_scale_dp0',
             'Layer 0: Attention Scale DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_mask_dp0',
             'Layer 0: Attention Mask DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_softmax_dp0',
             'Layer 0: Softmax DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_v_matmul_dp0',
             'Layer 0: Attention V MatMul DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32] x [batch_size=64, seq_len=1024, heads=32, d_k=128]\nOutput: [batch_size=64, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_out_proj_dp0',
             'Layer 0: Attention Output Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, heads=32, d_k=128]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_residual_dp0',
             'Layer 0: Attention Residual DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096] x2\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_ar_dp0',
             'All-Reduce\nTP Group: DP0_PP0_TP0-DP0_PP0_TP7\nSize: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse', style='dashed')
    
    # Layer 0 MoE - DP0
    dot.node('layer0_norm2_dp0',
             'Layer 0: RMSNorm 2 DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightyellow', shape='parallelogram')
    
    dot.node('layer0_gate_dp0',
             'Layer 0: MoE Gate DP0\nGPU: DP0_EP0-DP0_EP63\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, num_experts=64]',
             fillcolor='lightyellow', shape='parallelogram')
    
    dot.node('layer0_route_dp0',
             'Token Routing DP0\nGPU: DP0_EP0-DP0_EP63\nSelect 2 experts per token\nLoad balance across 64 experts',
             fillcolor='pink', shape='ellipse', style='dashed')
    
    # Expert distribution - 64 experts across 64 EP groups (1 expert per GPU)
    dot.node('layer0_expert0_dp0',
             'Layer 0: Expert 0 DP0\nGPU: DP0_EP0\nInput: [batch_size=1, seq_len=16, hidden_size=4096]\nOutput: [batch_size=1, seq_len=16, hidden_size=4096]\nUp-proj → Activation → Down-proj',
             fillcolor='lightsteelblue')
    
    dot.node('layer0_expert1_dp0',
             'Layer 0: Expert 1 DP0\nGPU: DP0_EP1\nInput: [batch_size=1, seq_len=16, hidden_size=4096]\nOutput: [batch_size=1, seq_len=16, hidden_size=4096]\nUp-proj → Activation → Down-proj',
             fillcolor='lightsteelblue')
    
    dot.node('layer0_ep_a2a_dp0',
             'All-to-All\nEP Group: DP0_EP0-DP0_EP63\nToken dispatch/combine\nSize: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('layer0_expert_combine_dp0',
             'Expert Combine DP0\nGPU: DP0_EP0-DP0_EP63\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightyellow', shape='parallelogram')
    
    dot.node('layer0_moe_residual_dp0',
             'Layer 0: MoE Residual DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096] x2\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    # DP1 Branch - Complete attention mechanism (CRITICAL FIX: with ALL steps including mask)
    dot.node('embed_dp1',
             'Embedding Layer DP1\nGPU: DP1_TP0-DP1_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=512]\nTP=8: hidden_size=4096 → 512 per TP rank',
             fillcolor='lightgreen')
    
    dot.node('hidden_up_dp1',
             'Hidden Upscaling DP1\nGPU: DP1_TP0-DP1_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=512]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]\nTP=8: All-Gather operation',
             fillcolor='lightgreen')
    
    dot.node('pp0_dp1_start',
             'PP Stage 0 DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nLayers 0-7 (8 layers)\n8 layers × 64 experts = 512 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    # Layer 0 Attention - DP1 (Complete with ALL steps including mask)
    dot.node('layer0_norm_dp1',
             'Layer 0: RMSNorm DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_q_proj_dp1',
             'Layer 0: Q Projection DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_k_proj_dp1',
             'Layer 0: K Projection DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_v_proj_dp1',
             'Layer 0: V Projection DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_qk_matmul_dp1',
             'Layer 0: QK^T MatMul DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, heads=32, d_k=128] x2\nOutput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_scale_dp1',
             'Layer 0: Attention Scale DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    # CRITICAL FIX: Added missing attention mask step for DP1
    dot.node('layer0_attn_mask_dp1',
             'Layer 0: Attention Mask DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_softmax_dp1',
             'Layer 0: Softmax DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_v_matmul_dp1',
             'Layer 0: Attention V MatMul DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, seq_len=1024, heads=32] x [batch_size=64, seq_len=1024, heads=32, d_k=128]\nOutput: [batch_size=64, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_out_proj_dp1',
             'Layer 0: Attention Output Projection DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, heads=32, d_k=128]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_residual_dp1',
             'Layer 0: Attention Residual DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096] x2\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_ar_dp1',
             'All-Reduce\nTP Group: DP1_PP0_TP0-DP1_PP0_TP7\nSize: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse', style='dashed')
    
    # Layer 0 MoE - DP1
    dot.node('layer0_norm2_dp1',
             'Layer 0: RMSNorm 2 DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightyellow', shape='parallelogram')
    
    dot.node('layer0_gate_dp1',
             'Layer 0: MoE Gate DP1\nGPU: DP1_EP0-DP1_EP63\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, num_experts=64]',
             fillcolor='lightyellow', shape='parallelogram')
    
    dot.node('layer0_route_dp1',
             'Token Routing DP1\nGPU: DP1_EP0-DP1_EP63\nSelect 2 experts per token\nLoad balance across 64 experts',
             fillcolor='pink', shape='ellipse', style='dashed')
    
    dot.node('layer0_expert0_dp1',
             'Layer 0: Expert 0 DP1\nGPU: DP1_EP0\nInput: [batch_size=1, seq_len=16, hidden_size=4096]\nOutput: [batch_size=1, seq_len=16, hidden_size=4096]\nUp-proj → Activation → Down-proj',
             fillcolor='lightsteelblue')
    
    dot.node('layer0_expert1_dp1',
             'Layer 0: Expert 1 DP1\nGPU: DP1_EP1\nInput: [batch_size=1, seq_len=16, hidden_size=4096]\nOutput: [batch_size=1, seq_len=16, hidden_size=4096]\nUp-proj → Activation → Down-proj',
             fillcolor='lightsteelblue')
    
    dot.node('layer0_ep_a2a_dp1',
             'All-to-All\nEP Group: DP1_EP0-DP1_EP63\nToken dispatch/combine\nSize: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('layer0_expert_combine_dp1',
             'Expert Combine DP1\nGPU: DP1_EP0-DP1_EP63\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightyellow', shape='parallelogram')
    
    dot.node('layer0_moe_residual_dp1',
             'Layer 0: MoE Residual DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096] x2\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    # Remaining layers and pipeline stages
    dot.node('layers1to7_dp0',
             'Layers 1-7 DP0\nSame structure as Layer 0\n8 layers per stage\nTP=8, EP=64 distribution',
             fillcolor='lightgray')
    
    dot.node('layers1to7_dp1',
             'Layers 1-7 DP1\nSame structure as Layer 0\n8 layers per stage\nTP=8, EP=64 distribution',
             fillcolor='lightgray')
    
    dot.node('pp0_to_pp1_dp0',
             'Pipeline Transfer DP0\nPP0 → PP1\nActivations transfer\nSize: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('pp0_to_pp1_dp1',
             'Pipeline Transfer DP1\nPP0 → PP1\nActivations transfer\nSize: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('pp1_dp0_start',
             'PP Stage 1 DP0\nGPU: DP0_PP1_TP0-DP0_PP1_TP7\nLayers 8-15 (8 layers)\n8 layers × 64 experts = 512 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    dot.node('pp1_dp1_start',
             'PP Stage 1 DP1\nGPU: DP1_PP1_TP0-DP1_PP1_TP7\nLayers 8-15 (8 layers)\n8 layers × 64 experts = 512 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    dot.node('layers8to15_dp0',
             'Layers 8-15 DP0\nSame structure as Layer 0\n8 layers per stage\nTP=8, EP=64 distribution',
             fillcolor='lightgray')
    
    dot.node('layers8to15_dp1',
             'Layers 8-15 DP1\nSame structure as Layer 0\n8 layers per stage\nTP=8, EP=64 distribution',
             fillcolor='lightgray')
    
    dot.node('final_norm_dp0',
             'Final RMSNorm DP0\nGPU: DP0_PP1_TP0-DP0_PP1_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightgreen')
    
    dot.node('final_norm_dp1',
             'Final RMSNorm DP1\nGPU: DP1_PP1_TP0-DP1_PP1_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, hidden_size=4096]',
             fillcolor='lightgreen')
    
    dot.node('output_proj_dp0',
             'Output Projection DP0\nGPU: DP0_PP1_TP0-DP0_PP1_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, vocab_size=51200]',
             fillcolor='lightgreen')
    
    dot.node('output_proj_dp1',
             'Output Projection DP1\nGPU: DP1_PP1_TP0-DP1_PP1_TP7\nInput: [batch_size=64, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=64, seq_len=1024, vocab_size=51200]',
             fillcolor='lightgreen')
    
    dot.node('dp_combine',
             'Data Parallelism Combine\nGPU: Output aggregation\nInput: 2×[batch_size=64, seq_len=1024, vocab_size=51200]\nOutput: [batch_size=128, seq_len=1024, vocab_size=51200]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('output',
             'Output\nInput: [batch_size=128, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=128, seq_len=1024, vocab_size=51200]',
             fillcolor='lightblue', shape='ellipse')
    
    # Edges - DP0 Branch
    dot.edge('input', 'dp_split')
    dot.edge('dp_split', 'embed_dp0')
    dot.edge('embed_dp0', 'hidden_up_dp0')
    dot.edge('hidden_up_dp0', 'pp0_dp0_start')
    dot.edge('pp0_dp0_start', 'layer0_norm_dp0')
    
    # Attention edges - DP0
    dot.edge('layer0_norm_dp0', 'layer0_q_proj_dp0')
    dot.edge('layer0_norm_dp0', 'layer0_k_proj_dp0')
    dot.edge('layer0_norm_dp0', 'layer0_v_proj_dp0')
    dot.edge('layer0_q_proj_dp0', 'layer0_qk_matmul_dp0')
    dot.edge('layer0_k_proj_dp0', 'layer0_qk_matmul_dp0')
    dot.edge('layer0_qk_matmul_dp0', 'layer0_attn_scale_dp0')
    dot.edge('layer0_attn_scale_dp0', 'layer0_attn_mask_dp0')
    dot.edge('layer0_attn_mask_dp0', 'layer0_attn_softmax_dp0')
    dot.edge('layer0_attn_softmax_dp0', 'layer0_attn_v_matmul_dp0')
    dot.edge('layer0_v_proj_dp0', 'layer0_attn_v_matmul_dp0')
    dot.edge('layer0_attn_v_matmul_dp0', 'layer0_attn_out_proj_dp0')
    dot.edge('layer0_attn_out_proj_dp0', 'layer0_attn_residual_dp0')
    dot.edge('hidden_up_dp0', 'layer0_attn_residual_dp0')
    dot.edge('layer0_attn_residual_dp0', 'layer0_attn_ar_dp0')
    
    # MoE edges - DP0
    dot.edge('layer0_attn_ar_dp0', 'layer0_norm2_dp0')
    dot.edge('layer0_norm2_dp0', 'layer0_gate_dp0')
    dot.edge('layer0_gate_dp0', 'layer0_route_dp0')
    dot.edge('layer0_route_dp0', 'layer0_expert0_dp0')
    dot.edge('layer0_route_dp0', 'layer0_expert1_dp0')
    dot.edge('layer0_expert0_dp0', 'layer0_ep_a2a_dp0')
    dot.edge('layer0_expert1_dp0', 'layer0_ep_a2a_dp0')
    dot.edge('layer0_ep_a2a_dp0', 'layer0_expert_combine_dp0')
    dot.edge('layer0_expert_combine_dp0', 'layer0_moe_residual_dp0')
    dot.edge('layer0_attn_ar_dp0', 'layer0_moe_residual_dp0')
    dot.edge('layer0_moe_residual_dp0', 'layers1to7_dp0')
    
    # Continue to remaining layers
    dot.edge('layers1to7_dp0', 'pp0_to_pp1_dp0')
    dot.edge('pp0_to_pp1_dp0', 'pp1_dp0_start')
    dot.edge('pp1_dp0_start', 'layers8to15_dp0')
    dot.edge('layers8to15_dp0', 'final_norm_dp0')
    dot.edge('final_norm_dp0', 'output_proj_dp0')
    dot.edge('output_proj_dp0', 'dp_combine')
    
    # Edges - DP1 Branch (Complete with ALL attention steps including mask)
    dot.edge('dp_split', 'embed_dp1')
    dot.edge('embed_dp1', 'hidden_up_dp1')
    dot.edge('hidden_up_dp1', 'pp0_dp1_start')
    dot.edge('pp0_dp1_start', 'layer0_norm_dp1')
    
    # Attention edges - DP1 (Complete with mask step)
    dot.edge('layer0_norm_dp1', 'layer0_q_proj_dp1')
    dot.edge('layer0_norm_dp1', 'layer0_k_proj_dp1')
    dot.edge('layer0_norm_dp1', 'layer0_v_proj_dp1')
    dot.edge('layer0_q_proj_dp1', 'layer0_qk_matmul_dp1')
    dot.edge('layer0_k_proj_dp1', 'layer0_qk_matmul_dp1')
    dot.edge('layer0_qk_matmul_dp1', 'layer0_attn_scale_dp1')
    dot.edge('layer0_attn_scale_dp1', 'layer0_attn_mask_dp1')  # CRITICAL FIX: Added missing mask
    dot.edge('layer0_attn_mask_dp1', 'layer0_attn_softmax_dp1')
    dot.edge('layer0_attn_softmax_dp1', 'layer0_attn_v_matmul_dp1')
    dot.edge('layer0_v_proj_dp1', 'layer0_attn_v_matmul_dp1')
    dot.edge('layer0_attn_v_matmul_dp1', 'layer0_attn_out_proj_dp1')
    dot.edge('layer0_attn_out_proj_dp1', 'layer0_attn_residual_dp1')
    dot.edge('hidden_up_dp1', 'layer0_attn_residual_dp1')
    dot.edge('layer0_attn_residual_dp1', 'layer0_attn_ar_dp1')
    
    # MoE edges - DP1
    dot.edge('layer0_attn_ar_dp1', 'layer0_norm2_dp1')
    dot.edge('layer0_norm2_dp1', 'layer0_gate_dp1')
    dot.edge('layer0_gate_dp1', 'layer0_route_dp1')
    dot.edge('layer0_route_dp1', 'layer0_expert0_dp1')
    dot.edge('layer0_route_dp1', 'layer0_expert1_dp1')
    dot.edge('layer0_expert0_dp1', 'layer0_ep_a2a_dp1')
    dot.edge('layer0_expert1_dp1', 'layer0_ep_a2a_dp1')
    dot.edge('layer0_ep_a2a_dp1', 'layer0_expert_combine_dp1')
    dot.edge('layer0_expert_combine_dp1', 'layer0_moe_residual_dp1')
    dot.edge('layer0_attn_ar_dp1', 'layer0_moe_residual_dp1')
    dot.edge('layer0_moe_residual_dp1', 'layers1to7_dp1')
    
    # Pipeline continuation - DP1
    dot.edge('layers1to7_dp1', 'pp0_to_pp1_dp1')
    dot.edge('pp0_to_pp1_dp1', 'pp1_dp1_start')
    dot.edge('pp1_dp1_start', 'layers8to15_dp1')
    dot.edge('layers8to15_dp1', 'final_norm_dp1')
    dot.edge('final_norm_dp1', 'output_proj_dp1')
    dot.edge('output_proj_dp1', 'dp_combine')
    
    # Final output
    dot.edge('dp_combine', 'output')
    
    return dot

def main():
    """Generate both corrected DAGs"""
    
    # Create directory if it doesn't exist
    os.makedirs('../outputs/2025-12-22-15-40-56', exist_ok=True)
    
    # Generate current deployment DAG
    print("Generating corrected current deployment DAG...")
    current_dag = generate_current_deployment_dag()
    
    # Save DOT file
    current_dot_path = '../outputs/2025-12-22-15-40-56/current_deployment_dag_corrected.dot'
    with open(current_dot_path, 'w') as f:
        f.write(current_dag.source)
    
    # Render to SVG
    current_svg_path = '../outputs/2025-12-22-15-40-56/current_deployment_dag_corrected.svg'
    current_dag.render(current_dot_path.replace('.dot', ''), format='svg', cleanup=True)
    
    print(f"Current deployment DAG saved to:")
    print(f"  DOT: {current_dot_path}")
    print(f"  SVG: {current_svg_path}")
    
    return {
        "current_deployment": {
            "strategy": "EP64-TP8-PP2-DP2",
            "dot_file": current_dot_path,
            "svg_file": current_svg_path,
            "graphviz_code": current_dag.source
        }
    }

if __name__ == "__main__":
    result = main()
    print("\nDAG generation completed successfully!")
    print(f"Current deployment strategy: {result['current_deployment']['strategy']}")
    print(f"DOT file: {result['current_deployment']['dot_file']}")
    print(f"SVG file: {result['current_deployment']['svg_file']}")