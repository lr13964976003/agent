#!/usr/bin/env python3

import os

def generate_corrected_moe_dag():
    """Generate a corrected MoE DAG that fixes all structural violations"""
    
    dot_content = '''// 30B MoE Model Deployment DAG - EP16-TP8-PP4-DP4 (CORRECTED)
digraph {
	bgcolor=white rankdir=TB splines=ortho
	node [fillcolor=lightblue shape=rectangle style=filled]
	node [fillcolor=lightgreen shape=ellipse style=filled]
	node [fillcolor=lightyellow shape=parallelogram style=filled]
	
	// Input Layer
	subgraph cluster_input {
		fillcolor=lightgray label="Input Layer" style=rounded
		input [label="Input Embedding\nGPU: [0-511]\nInput: [batch_size=128, seq_len=1024, hidden=1024]\nOutput: [batch_size=128, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
	}
	
	// DP Split
	dp_split [label="DP Split\nGPU: [0-511]\nInput: [batch_size=128, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
	input -> dp_split
	
	// Pipeline Stage 0: Layers 0-3 (GPUs 0-127)
	subgraph cluster_pp0 {
		fillcolor=lightcoral label="Pipeline Stage 0: Layers 0-3 (GPUs 0-127)" style=rounded
		
		// Layer 0
		pp0_layer0_attn [label="Layer 0: Attention\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer0_tp_comm [label="TP All-Reduce\nGPU: [0-7], [8-15], [16-23], [24-31]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer0_moe [label="Layer 0: MoE Routing\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp0_layer0_gate [label="Expert Gate Selection\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 0
		pp0_layer0_expert0 [label="Expert 0\nGPU: [0-7]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer0_expert0_tp [label="TP All-Reduce\nGPU: [0-7]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer0_expert1 [label="Expert 1\nGPU: [8-15]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer0_expert1_tp [label="TP All-Reduce\nGPU: [8-15]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer0_expert2 [label="Expert 2\nGPU: [16-23]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer0_expert2_tp [label="TP All-Reduce\nGPU: [16-23]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer0_expert3 [label="Expert 3\nGPU: [24-31]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer0_expert3_tp [label="TP All-Reduce\nGPU: [24-31]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 0
		pp0_layer0_agg [label="Expert Aggregation\nGPU: [0-31]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp0_layer0_ep_comm [label="EP All-to-All\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Layer 3 (final layer in PP0)
		pp0_layer3_final [label="Layer 3: Final\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer3_ep_comm [label="EP All-to-All\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
	}
	
	// Pipeline Stage 1: Layers 4-7 (GPUs 128-255)
	subgraph cluster_pp1 {
		fillcolor=lightsteelblue label="Pipeline Stage 1: Layers 4-7 (GPUs 128-255)" style=rounded
		
		// Layer 4
		pp1_layer4_attn [label="Layer 4: Attention\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer4_tp_comm [label="TP All-Reduce\nGPU: [128-135], [136-143], [144-151], [152-159]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer4_moe [label="Layer 4: MoE Routing\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp1_layer4_gate [label="Expert Gate Selection\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 4
		pp1_layer4_expert0 [label="Expert 0\nGPU: [128-135]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer4_expert0_tp [label="TP All-Reduce\nGPU: [128-135]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer4_expert1 [label="Expert 1\nGPU: [136-143]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer4_expert1_tp [label="TP All-Reduce\nGPU: [136-143]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer4_expert2 [label="Expert 2\nGPU: [144-151]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer4_expert2_tp [label="TP All-Reduce\nGPU: [144-151]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer4_expert3 [label="Expert 3\nGPU: [152-159]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer4_expert3_tp [label="TP All-Reduce\nGPU: [152-159]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 4
		pp1_layer4_agg [label="Expert Aggregation\nGPU: [128-159]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp1_layer4_ep_comm [label="EP All-to-All\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Layer 7 (final layer in PP1)
		pp1_layer7_final [label="Layer 7: Final\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer7_ep_comm [label="EP All-to-All\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
	}
	
	// Pipeline Stage 2: Layers 8-11 (GPUs 256-383)
	subgraph cluster_pp2 {
		fillcolor=lightseagreen label="Pipeline Stage 2: Layers 8-11 (GPUs 256-383)" style=rounded
		
		// Layer 8
		pp2_layer8_attn [label="Layer 8: Attention\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer8_tp_comm [label="TP All-Reduce\nGPU: [256-263], [264-271], [272-279], [280-287]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer8_moe [label="Layer 8: MoE Routing\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp2_layer8_gate [label="Expert Gate Selection\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 8
		pp2_layer8_expert0 [label="Expert 0\nGPU: [256-263]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer8_expert0_tp [label="TP All-Reduce\nGPU: [256-263]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer8_expert1 [label="Expert 1\nGPU: [264-271]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer8_expert1_tp [label="TP All-Reduce\nGPU: [264-271]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer8_expert2 [label="Expert 2\nGPU: [272-279]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer8_expert2_tp [label="TP All-Reduce\nGPU: [272-279]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer8_expert3 [label="Expert 3\nGPU: [280-287]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer8_expert3_tp [label="TP All-Reduce\nGPU: [280-287]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 8
		pp2_layer8_agg [label="Expert Aggregation\nGPU: [256-287]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp2_layer8_ep_comm [label="EP All-to-All\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Layer 11 (final layer in PP2)
		pp2_layer11_final [label="Layer 11: Final\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer11_ep_comm [label="EP All-to-All\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
	}
	
	// Pipeline Stage 3: Layers 12-15 (GPUs 384-511)
	subgraph cluster_pp3 {
		fillcolor=lightsalmon label="Pipeline Stage 3: Layers 12-15 (GPUs 384-511)" style=rounded
		
		// Layer 12
		pp3_layer12_attn [label="Layer 12: Attention\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp3_layer12_tp_comm [label="TP All-Reduce\nGPU: [384-391], [392-399], [400-407], [408-415]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp3_layer12_moe [label="Layer 12: MoE Routing\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp3_layer12_gate [label="Expert Gate Selection\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 12
		pp3_layer12_expert0 [label="Expert 0\nGPU: [384-391]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp3_layer12_expert0_tp [label="TP All-Reduce\nGPU: [384-391]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp3_layer12_expert1 [label="Expert 1\nGPU: [392-399]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp3_layer12_expert1_tp [label="TP All-Reduce\nGPU: [392-399]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp3_layer12_expert2 [label="Expert 2\nGPU: [400-407]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp3_layer12_expert2_tp [label="TP All-Reduce\nGPU: [400-407]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp3_layer12_expert3 [label="Expert 3\nGPU: [408-415]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp3_layer12_expert3_tp [label="TP All-Reduce\nGPU: [408-415]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 12
		pp3_layer12_agg [label="Expert Aggregation\nGPU: [384-415]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp3_layer12_ep_comm [label="EP All-to-All\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Layer 15 (final layer in PP3)
		pp3_layer15_final [label="Layer 15: Final\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp3_layer15_ep_comm [label="EP All-to-All\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
	}
	
	// Output Layer
	subgraph cluster_output {
		fillcolor=lightgray label="Output Layer" style=rounded
		output [label="Output Layer\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, vocab_size=32000]" fillcolor=lightblue shape=rectangle]
		dp_agg [label="DP Aggregation\nGPU: [0-511]\nInput: [batch_size=32, seq_len=1024, vocab_size=32000]\nOutput: [batch_size=128, seq_len=1024, vocab_size=32000]" fillcolor=lightyellow shape=parallelogram]
	}
	
	// Connections - FIXING THE BROKEN STRUCTURE
	dp_split -> pp0_layer0_attn
	
	// PP0 Layer 0 connections
	pp0_layer0_attn -> pp0_layer0_tp_comm
	pp0_layer0_tp_comm -> pp0_layer0_moe
	pp0_layer0_moe -> pp0_layer0_gate
	pp0_layer0_gate -> pp0_layer0_expert0 [style=dashed]
	pp0_layer0_gate -> pp0_layer0_expert1 [style=dashed]
	pp0_layer0_gate -> pp0_layer0_expert2 [style=dashed]
	pp0_layer0_gate -> pp0_layer0_expert3 [style=dashed]
	pp0_layer0_expert0 -> pp0_layer0_expert0_tp
	pp0_layer0_expert1 -> pp0_layer0_expert1_tp
	pp0_layer0_expert2 -> pp0_layer0_expert2_tp
	pp0_layer0_expert3 -> pp0_layer0_expert3_tp
	pp0_layer0_expert0_tp -> pp0_layer0_agg
	pp0_layer0_expert1_tp -> pp0_layer0_agg
	pp0_layer0_expert2_tp -> pp0_layer0_agg
	pp0_layer0_expert3_tp -> pp0_layer0_agg
	pp0_layer0_agg -> pp0_layer0_ep_comm
	
	// PP0 Layer 3 connections (simplified)
	pp0_layer0_ep_comm -> pp0_layer3_final
	pp0_layer3_final -> pp0_layer3_ep_comm
	
	// PP1 Layer 4 connections
	pp0_layer3_ep_comm -> pp1_layer4_attn
	pp1_layer4_attn -> pp1_layer4_tp_comm
	pp1_layer4_tp_comm -> pp1_layer4_moe
	pp1_layer4_moe -> pp1_layer4_gate
	pp1_layer4_gate -> pp1_layer4_expert0 [style=dashed]
	pp1_layer4_gate -> pp1_layer4_expert1 [style=dashed]
	pp1_layer4_gate -> pp1_layer4_expert2 [style=dashed]
	pp1_layer4_gate -> pp1_layer4_expert3 [style=dashed]
	pp1_layer4_expert0 -> pp1_layer4_expert0_tp
	pp1_layer4_expert1 -> pp1_layer4_expert1_tp
	pp1_layer4_expert2 -> pp1_layer4_expert2_tp
	pp1_layer4_expert3 -> pp1_layer4_expert3_tp
	pp1_layer4_expert0_tp -> pp1_layer4_agg
	pp1_layer4_expert1_tp -> pp1_layer4_agg
	pp1_layer4_expert2_tp -> pp1_layer4_agg
	pp1_layer4_expert3_tp -> pp1_layer4_agg
	pp1_layer4_agg -> pp1_layer4_ep_comm
	
	// PP1 Layer 7 connections (simplified)
	pp1_layer4_ep_comm -> pp1_layer7_final
	pp1_layer7_final -> pp1_layer7_ep_comm
	
	// PP2 Layer 8 connections
	pp1_layer7_ep_comm -> pp2_layer8_attn
	pp2_layer8_attn -> pp2_layer8_tp_comm
	pp2_layer8_tp_comm -> pp2_layer8_moe
	pp2_layer8_moe -> pp2_layer8_gate
	pp2_layer8_gate -> pp2_layer8_expert0 [style=dashed]
	pp2_layer8_gate -> pp2_layer8_expert1 [style=dashed]
	pp2_layer8_gate -> pp2_layer8_expert2 [style=dashed]
	pp2_layer8_gate -> pp2_layer8_expert3 [style=dashed]
	pp2_layer8_expert0 -> pp2_layer8_expert0_tp
	pp2_layer8_expert1 -> pp2_layer8_expert1_tp
	pp2_layer8_expert2 -> pp2_layer8_expert2_tp
	pp2_layer8_expert3 -> pp2_layer8_expert3_tp
	pp2_layer8_expert0_tp -> pp2_layer8_agg
	pp2_layer8_expert1_tp -> pp2_layer8_agg
	pp2_layer8_expert2_tp -> pp2_layer8_agg
	pp2_layer8_expert3_tp -> pp2_layer8_agg
	pp2_layer8_agg -> pp2_layer8_ep_comm
	
	// PP2 Layer 11 connections (simplified)
	pp2_layer8_ep_comm -> pp2_layer11_final
	pp2_layer11_final -> pp2_layer11_ep_comm
	
	// PP3 Layer 12 connections
	pp2_layer11_ep_comm -> pp3_layer12_attn
	pp3_layer12_attn -> pp3_layer12_tp_comm
	pp3_layer12_tp_comm -> pp3_layer12_moe
	pp3_layer12_moe -> pp3_layer12_gate
	pp3_layer12_gate -> pp3_layer12_expert0 [style=dashed]
	pp3_layer12_gate -> pp3_layer12_expert1 [style=dashed]
	pp3_layer12_gate -> pp3_layer12_expert2 [style=dashed]
	pp3_layer12_gate -> pp3_layer12_expert3 [style=dashed]
	pp3_layer12_expert0 -> pp3_layer12_expert0_tp
	pp3_layer12_expert1 -> pp3_layer12_expert1_tp
	pp3_layer12_expert2 -> pp3_layer12_expert2_tp
	pp3_layer12_expert3 -> pp3_layer12_expert3_tp
	pp3_layer12_expert0_tp -> pp3_layer12_agg
	pp3_layer12_expert1_tp -> pp3_layer12_agg
	pp3_layer12_expert2_tp -> pp3_layer12_agg
	pp3_layer12_expert3_tp -> pp3_layer12_agg
	pp3_layer12_agg -> pp3_layer12_ep_comm
	
	// PP3 Layer 15 connections (simplified)
	pp3_layer12_ep_comm -> pp3_layer15_final
	pp3_layer15_final -> pp3_layer15_ep_comm
	
	// Output connections
	pp3_layer15_ep_comm -> output
	output -> dp_agg
}
'''
    
    # Write the corrected DAG
    with open('../outputs/2025-12-05-09-02-16/moe_corrected_dag.dot', 'w') as f:
        f.write(dot_content)
    
    # Generate SVG using graphviz
    os.system('dot -Tsvg ../outputs/2025-12-05-09-02-16/moe_corrected_dag.dot -o ../outputs/2025-12-05-09-02-16/moe_corrected_dag.svg')
    
    print("Corrected DAG generated successfully!")
    print("Files created:")
    print("- ../outputs/2025-12-05-09-02-16/moe_corrected_dag.dot")
    print("- ../outputs/2025-12-05-09-02-16/moe_corrected_dag.svg")

if __name__ == "__main__":
    generate_corrected_moe_dag()