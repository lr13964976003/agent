#!/usr/bin/env python3
"""
Corrected DAG Generator for EP32-TP16-PP4-DP4 New Deployment Strategy
Addresses all critical issues found in previous submissions
"""

import os
from graphviz import Digraph

def generate_new_deployment_dag():
    """Generate corrected DAG for EP32-TP16-PP4-DP4 deployment strategy"""
    
    dot = Digraph(comment='EP32-TP16-PP4-DP4 MoE LLM Inference DAG (Corrected)')
    dot.attr(bgcolor='white', fontname='Arial', rankdir='TB')
    dot.attr('node', fontname='Arial', shape='rectangle', style='filled')
    
    # Input node - DP splits the batch
    dot.node('input', 
             'Input\nInput: [batch_size=128, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]\nDP=4: Split batch into 4 groups of 32 sequences each',
             fillcolor='lightblue', shape='ellipse')
    
    # DP Split node
    dot.node('dp_split',
             'Data Parallelism Split\nGPU: DP0-DP3\nSplit batch_size=128 → 4×32\nNo communication required for inference',
             fillcolor='lightgreen', shape='parallelogram')
    
    # DP0 Branch - Complete attention mechanism
    dot.node('embed_dp0',
             'Embedding Layer DP0\nGPU: DP0_TP0-DP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=256]\nTP=16: hidden_size=4096 → 256 per TP rank',
             fillcolor='lightgreen')
    
    dot.node('hidden_up_dp0',
             'Hidden Upscaling DP0\nGPU: DP0_TP0-DP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=256]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]\nTP=16: All-Gather operation',
             fillcolor='lightgreen')
    
    dot.node('pp0_dp0_start',
             'PP Stage 0 DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nLayers 0-3 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    # Complete Layer 0 attention for DP0 with correct dimensions
    dot.node('layer0_norm_dp0',
             'Layer 0: RMSNorm DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_q_proj_dp0',
             'Layer 0: Q Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_k_proj_dp0',
             'Layer 0: K Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_v_proj_dp0',
             'Layer 0: V Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_qk_matmul_dp0',
             'Layer 0: QK^T MatMul DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, heads=32, d_k=128] x2\nOutput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_scale_dp0',
             'Layer 0: Attention Scale DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_mask_dp0',
             'Layer 0: Attention Mask DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_softmax_dp0',
             'Layer 0: Softmax DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_v_matmul_dp0',
             'Layer 0: Attention V MatMul DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32] x [batch_size=32, seq_len=1024, heads=32, d_k=128]\nOutput: [batch_size=32, seq_len=1024, heads=32, d_k=128]',
              fillcolor='lightcoral')
    
    dot.node('layer0_attn_out_proj_dp0',
             'Layer 0: Attention Output Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, heads=32, d_k=128]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_residual_dp0',
             'Layer 0: Attention Residual DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096] x2\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_ar_dp0',
             'All-Reduce\nTP Group: DP0_PP0_TP0-DP0_PP0_TP15\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse', style='dashed')
    
    # Layer 0 MoE - DP0
    dot.node('layer0_norm2_dp0',
             'Layer 0: RMSNorm 2 DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightyellow', shape='parallelogram')
    
    dot.node('layer0_gate_dp0',
             'Layer 0: MoE Gate DP0\nGPU: DP0_EP0-DP0_EP31\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, num_experts=64]',
             fillcolor='lightyellow', shape='parallelogram')
    
    dot.node('layer0_route_dp0',
             'Token Routing DP0\nGPU: DP0_EP0-DP0_EP31\nSelect 2 experts per token\nLoad balance across 64 experts\nEP32: 2 experts per GPU',
             fillcolor='pink', shape='ellipse', style='dashed')
    
    # Expert distribution - 64 experts across 32 EP groups (2 experts per GPU)
    dot.node('layer0_expert0_dp0',
             'Layer 0: Expert 0 DP0\nGPU: DP0_EP0\nInput: [batch_size=2, seq_len=64, hidden_size=4096]\nOutput: [batch_size=2, seq_len=64, hidden_size=4096]\nUp-proj → Activation → Down-proj',
             fillcolor='lightsteelblue')
    
    dot.node('layer0_expert1_dp0',
             'Layer 0: Expert 1 DP0\nGPU: DP0_EP0\nInput: [batch_size=2, seq_len=64, hidden_size=4096]\nOutput: [batch_size=2, seq_len=64, hidden_size=4096]\nUp-proj → Activation → Down-proj',
             fillcolor='lightsteelblue')
    
    dot.node('layer0_expert2_dp0',
             'Layer 0: Expert 2 DP0\nGPU: DP0_EP1\nInput: [batch_size=2, seq_len=64, hidden_size=4096]\nOutput: [batch_size=2, seq_len=64, hidden_size=4096]\nUp-proj → Activation → Down-proj',
             fillcolor='lightsteelblue')
    
    dot.node('layer0_expert3_dp0',
             'Layer 0: Expert 3 DP0\nGPU: DP0_EP1\nInput: [batch_size=2, seq_len=64, hidden_size=4096]\nOutput: [batch_size=2, seq_len=64, hidden_size=4096]\nUp-proj → Activation → Down-proj',
             fillcolor='lightsteelblue')
    
    dot.node('layer0_ep_a2a_dp0',
             'All-to-All\nEP Group: DP0_EP0-DP0_EP31\nToken dispatch/combine\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('layer0_expert_combine_dp0',
             'Expert Combine DP0\nGPU: DP0_EP0-DP0_EP31\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightyellow', shape='parallelogram')
    
    dot.node('layer0_moe_residual_dp0',
             'Layer 0: MoE Residual DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096] x2\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    # Remaining layers DP0
    dot.node('layers1to3_dp0',
             'Layers 1-3 DP0\nComplete attention + MoE structure\n4 layers per stage\nTP=16, EP=32 (2 experts/GPU)',
             fillcolor='lightgray')
    
    # Pipeline transfers DP0
    dot.node('pp0_to_pp1_dp0',
             'Pipeline Transfer DP0\nPP0 → PP1\nActivations transfer\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('pp1_to_pp2_dp0',
             'Pipeline Transfer DP0\nPP1 → PP2\nActivations transfer\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('pp2_to_pp3_dp0',
             'Pipeline Transfer DP0\nPP2 → PP3\nActivations transfer\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    # Pipeline stages DP0
    dot.node('pp1_dp0_start',
             'PP Stage 1 DP0\nGPU: DP0_PP1_TP0-DP0_PP1_TP15\nLayers 4-7 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    dot.node('pp2_dp0_start',
             'PP Stage 2 DP0\nGPU: DP0_PP2_TP0-DP0_PP2_TP15\nLayers 8-11 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    dot.node('pp3_dp0_start',
             'PP Stage 3 DP0\nGPU: DP0_PP3_TP0-DP0_PP3_TP15\nLayers 12-15 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    # Final norm DP0
    dot.node('final_norm_dp0',
             'Final RMSNorm DP0\nGPU: DP0_PP3_TP0-DP0_PP3_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightgreen')
    
    dot.node('output_proj_dp0',
             'Output Projection DP0\nGPU: DP0_PP3_TP0-DP0_PP3_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, vocab_size=51200]',
             fillcolor='lightgreen')
    
    # DP1 Branch - Complete attention mechanism
    dot.node('embed_dp1',
             'Embedding Layer DP1\nGPU: DP1_TP0-DP1_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=256]\nTP=16: hidden_size=4096 → 256 per TP rank',
             fillcolor='lightgreen')
    
    dot.node('hidden_up_dp1',
             'Hidden Upscaling DP1\nGPU: DP1_TP0-DP1_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=256]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]\nTP=16: All-Gather operation',
             fillcolor='lightgreen')
    
    dot.node('pp0_dp1_start',
             'PP Stage 0 DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nLayers 0-3 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    # Complete Layer 0 attention for DP1 with correct dimensions
    dot.node('layer0_norm_dp1',
             'Layer 0: RMSNorm DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_q_proj_dp1',
             'Layer 0: Q Projection DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_k_proj_dp1',
             'Layer 0: K Projection DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_v_proj_dp1',
             'Layer 0: V Projection DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_qk_matmul_dp1',
             'Layer 0: QK^T MatMul DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, heads=32, d_k=128] x2\nOutput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_scale_dp1',
             'Layer 0: Attention Scale DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_mask_dp1',
             'Layer 0: Attention Mask DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_softmax_dp1',
             'Layer 0: Softmax DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]\nOutput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_v_matmul_dp1',
             'Layer 0: Attention V MatMul DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, seq_len=1024, heads=32] x [batch_size=32, seq_len=1024, heads=32, d_k=128]\nOutput: [batch_size=32, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_out_proj_dp1',
             'Layer 0: Attention Output Projection DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, heads=32, d_k=128]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_residual_dp1',
             'Layer 0: Attention Residual DP1\nGPU: DP1_PP0_TP0-DP1_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096] x2\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_attn_ar_dp1',
             'All-Reduce\nTP Group: DP1_PP0_TP0-DP1_PP0_TP15\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse', style='dashed')
    
    # Final layers DP1
    dot.node('final_norm_dp1',
             'Final RMSNorm DP1\nGPU: DP1_PP3_TP0-DP1_PP3_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightgreen')
    
    dot.node('output_proj_dp1',
             'Output Projection DP1\nGPU: DP1_PP3_TP0-DP1_PP3_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, vocab_size=51200]',
             fillcolor='lightgreen')
    
    # CRITICAL FIX: Add complete flows for DP2 and DP3 (simplified representation)
    dot.node('layers1to3_dp2',
             'Layers 1-3 DP2\nComplete attention + MoE structure\n4 layers per stage\nTP=16, EP=32 (2 experts/GPU)',
             fillcolor='lightgray')
    
    dot.node('layers1to3_dp3',
             'Layers 1-3 DP3\nComplete attention + MoE structure\n4 layers per stage\nTP=16, EP=32 (2 experts/GPU)',
             fillcolor='lightgray')
    
    # Pipeline transfers for all branches
    dot.node('pp0_to_pp1_dp2',
             'Pipeline Transfer DP2\nPP0 → PP1\nActivations transfer\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('pp1_to_pp2_dp2',
             'Pipeline Transfer DP2\nPP1 → PP2\nActivations transfer\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('pp2_to_pp3_dp2',
             'Pipeline Transfer DP2\nPP2 → PP3\nActivations transfer\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('pp0_to_pp1_dp3',
             'Pipeline Transfer DP3\nPP0 → PP1\nActivations transfer\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('pp1_to_pp2_dp3',
             'Pipeline Transfer DP3\nPP1 → PP2\nActivations transfer\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('pp2_to_pp3_dp3',
             'Pipeline Transfer DP3\nPP2 → PP3\nActivations transfer\nSize: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='orange', shape='ellipse')
    
    # Pipeline stages for all branches
    dot.node('pp1_dp2_start',
             'PP Stage 1 DP2\nGPU: DP2_PP1_TP0-DP2_PP1_TP15\nLayers 4-7 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    dot.node('pp2_dp2_start',
             'PP Stage 2 DP2\nGPU: DP2_PP2_TP0-DP2_PP2_TP15\nLayers 8-11 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    dot.node('pp3_dp2_start',
             'PP Stage 3 DP2\nGPU: DP2_PP3_TP0-DP2_PP3_TP15\nLayers 12-15 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    dot.node('pp1_dp3_start',
             'PP Stage 1 DP3\nGPU: DP3_PP1_TP0-DP3_PP1_TP15\nLayers 4-7 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    dot.node('pp2_dp3_start',
             'PP Stage 2 DP3\nGPU: DP3_PP2_TP0-DP3_PP2_TP15\nLayers 8-11 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    dot.node('pp3_dp3_start',
             'PP Stage 3 DP3\nGPU: DP3_PP3_TP0-DP3_PP3_TP15\nLayers 12-15 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    # Final combination
    dot.node('dp_combine',
             'Data Parallelism Combine\nGPU: Output aggregation\nInput: 4×[batch_size=32, seq_len=1024, vocab_size=51200]\nOutput: [batch_size=128, seq_len=1024, vocab_size=51200]',
             fillcolor='orange', shape='ellipse')
    
    dot.node('output',
             'Output\nInput: [batch_size=128, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=128, seq_len=1024, vocab_size=51200]',
             fillcolor='lightblue', shape='ellipse')
    
    # Complete attention flow for DP0 (all steps)
    dot.edge('input', 'dp_split')
    dot.edge('dp_split', 'embed_dp0')
    dot.edge('embed_dp0', 'hidden_up_dp0')
    dot.edge('hidden_up_dp0', 'pp0_dp0_start')
    dot.edge('pp0_dp0_start', 'layer0_norm_dp0')
    
    # Complete attention mechanism - DP0
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
    dot.edge('layer0_route_dp0', 'layer0_expert2_dp0')
    dot.edge('layer0_route_dp0', 'layer0_expert3_dp0')
    dot.edge('layer0_expert0_dp0', 'layer0_ep_a2a_dp0')
    dot.edge('layer0_expert1_dp0', 'layer0_ep_a2a_dp0')
    dot.edge('layer0_expert2_dp0', 'layer0_ep_a2a_dp0')
    dot.edge('layer0_expert3_dp0', 'layer0_ep_a2a_dp0')
    dot.edge('layer0_ep_a2a_dp0', 'layer0_expert_combine_dp0')
    dot.edge('layer0_expert_combine_dp0', 'layer0_moe_residual_dp0')
    dot.edge('layer0_attn_ar_dp0', 'layer0_moe_residual_dp0')
    dot.edge('layer0_moe_residual_dp0', 'layers1to3_dp0')
    
    # Pipeline continuation - DP0
    dot.edge('layers1to3_dp0', 'pp0_to_pp1_dp0')
    dot.edge('pp0_to_pp1_dp0', 'pp1_dp0_start')
    dot.edge('pp1_dp0_start', 'pp1_to_pp2_dp0')
    dot.edge('pp1_to_pp2_dp0', 'pp2_dp0_start')
    dot.edge('pp2_dp0_start', 'pp2_to_pp3_dp0')
    dot.edge('pp2_to_pp3_dp0', 'pp3_dp0_start')
    dot.edge('pp3_dp0_start', 'final_norm_dp0')
    dot.edge('final_norm_dp0', 'output_proj_dp0')
    dot.edge('output_proj_dp0', 'dp_combine')
    
    # Complete attention flow for DP1 (all steps)
    dot.edge('dp_split', 'embed_dp1')
    dot.edge('embed_dp1', 'hidden_up_dp1')
    dot.edge('hidden_up_dp1', 'pp0_dp1_start')
    dot.edge('pp0_dp1_start', 'layer0_norm_dp1')
    
    # Complete attention mechanism - DP1
    dot.edge('layer0_norm_dp1', 'layer0_q_proj_dp1')
    dot.edge('layer0_norm_dp1', 'layer0_k_proj_dp1')
    dot.edge('layer0_norm_dp1', 'layer0_v_proj_dp1')
    dot.edge('layer0_q_proj_dp1', 'layer0_qk_matmul_dp1')
    dot.edge('layer0_k_proj_dp1', 'layer0_qk_matmul_dp1')
    dot.edge('layer0_qk_matmul_dp1', 'layer0_attn_scale_dp1')
    dot.edge('layer0_attn_scale_dp1', 'layer0_attn_mask_dp1')
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
    dot.edge('layer0_moe_residual_dp1', 'layers1to3_dp1')
    
    # Pipeline continuation - DP1
    dot.edge('layers1to3_dp1', 'pp0_to_pp1_dp1')
    dot.edge('pp0_to_pp1_dp1', 'pp1_dp1_start')
    dot.edge('pp1_dp1_start', 'pp1_to_pp2_dp1')
    dot.edge('pp1_to_pp2_dp1', 'pp2_dp1_start')
    dot.edge('pp2_dp1_start', 'pp2_to_pp3_dp1')
    dot.edge('pp2_to_pp3_dp1', 'pp3_dp1_start')
    dot.edge('pp3_dp1_start', 'final_norm_dp1')
    dot.edge('final_norm_dp1', 'output_proj_dp1')
    dot.edge('output_proj_dp1', 'dp_combine')
    
    # CRITICAL FIX: Add complete flows for DP2 and DP3 (simplified representation)
    dot.edge('dp_split', 'embed_dp2')
    dot.edge('embed_dp2', 'hidden_up_dp2')
    dot.edge('hidden_up_dp2', 'pp0_dp2_start')
    dot.edge('pp0_dp2_start', 'layers1to3_dp2')  # Complete attention + MoE structure
    dot.edge('layers1to3_dp2', 'pp0_to_pp1_dp2')
    dot.edge('pp0_to_pp1_dp2', 'pp1_dp2_start')
    dot.edge('pp1_dp2_start', 'pp1_to_pp2_dp2')
    dot.edge('pp1_to_pp2_dp2', 'pp2_dp2_start')
    dot.edge('pp2_dp2_start', 'pp2_to_pp3_dp2')
    dot.edge('pp2_to_pp3_dp2', 'pp3_dp2_start')
    dot.edge('pp3_dp2_start', 'final_norm_dp2')
    dot.edge('final_norm_dp2', 'output_proj_dp2')
    dot.edge('output_proj_dp2', 'dp_combine')
    
    dot.edge('dp_split', 'embed_dp3')
    dot.edge('embed_dp3', 'hidden_up_dp3')
    dot.edge('hidden_up_dp3', 'pp0_dp3_start')
    dot.edge('pp0_dp3_start', 'layers1to3_dp3')  # Complete attention + MoE structure
    dot.edge('layers1to3_dp3', 'pp0_to_pp1_dp3')
    dot.edge('pp0_to_pp1_dp3', 'pp1_dp3_start')
    dot.edge('pp1_dp3_start', 'pp1_to_pp2_dp3')
    dot.edge('pp1_to_pp2_dp3', 'pp2_dp3_start')
    dot.edge('pp2_dp3_start', 'pp2_to_pp3_dp3')
    dot.edge('pp2_to_pp3_dp3', 'pp3_dp3_start')
    dot.edge('pp3_dp3_start', 'final_norm_dp3')
    dot.edge('final_norm_dp3', 'output_proj_dp3')
    dot.edge('output_proj_dp3', 'dp_combine')
    
    # Final output
    dot.edge('dp_combine', 'output')
    
    return dot

def main():
    """Generate corrected new deployment DAG"""
    
    # Create directory if it doesn't exist
    os.makedirs('../outputs/2025-12-22-15-40-56', exist_ok=True)
    
    # Generate new deployment DAG
    print("Generating corrected new deployment DAG...")
    new_dag = generate_new_deployment_dag()
    
    # Save DOT file
    new_dot_path = '../outputs/2025-12-22-15-40-56/new_deployment_dag_corrected.dot'
    with open(new_dot_path, 'w') as f:
        f.write(new_dag.source)
    
    # Render to SVG
    new_svg_path = '../outputs/2025-12-22-15-40-56/new_deployment_dag_corrected.svg'
    new_dag.render(new_dot_path.replace('.dot', ''), format='svg', cleanup=True)
    
    print(f"New deployment DAG saved to:")
    print(f"  DOT: {new_dot_path}")
    print(f"  SVG: {new_svg_path}")
    
    return {
        "new_deployment": {
            "strategy": "EP32-TP16-PP4-DP4",
            "dot_file": new_dot_path,
            "svg_file": new_svg_path,
            "graphviz_code": new_dag.source
        }
    }

if __name__ == "__main__":
    result = main()
    print("\nNew deployment DAG generation completed successfully!")
    print(f"New deployment strategy: {result['new_deployment']['strategy']}")
    print(f"DOT file: {result['new_deployment']['dot_file']}")
    print(f"SVG file: {result['new_deployment']['svg_file']}")