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
		
		// Layer 1
		pp0_layer1_attn [label="Layer 1: Attention\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer1_tp_comm [label="TP All-Reduce\nGPU: [0-7], [8-15], [16-23], [24-31]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer1_moe [label="Layer 1: MoE Routing\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp0_layer1_gate [label="Expert Gate Selection\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 1
		pp0_layer1_expert0 [label="Expert 0\nGPU: [0-7]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer1_expert0_tp [label="TP All-Reduce\nGPU: [0-7]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer1_expert1 [label="Expert 1\nGPU: [8-15]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer1_expert1_tp [label="TP All-Reduce\nGPU: [8-15]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer1_expert2 [label="Expert 2\nGPU: [16-23]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer1_expert2_tp [label="TP All-Reduce\nGPU: [16-23]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer1_expert3 [label="Expert 3\nGPU: [24-31]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer1_expert3_tp [label="TP All-Reduce\nGPU: [24-31]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 1
		pp0_layer1_agg [label="Expert Aggregation\nGPU: [0-31]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp0_layer1_ep_comm [label="EP All-to-All\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Layer 2
		pp0_layer2_attn [label="Layer 2: Attention\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer2_tp_comm [label="TP All-Reduce\nGPU: [0-7], [8-15], [16-23], [24-31]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer2_moe [label="Layer 2: MoE Routing\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp0_layer2_gate [label="Expert Gate Selection\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 2
		pp0_layer2_expert0 [label="Expert 0\nGPU: [0-7]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer2_expert0_tp [label="TP All-Reduce\nGPU: [0-7]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer2_expert1 [label="Expert 1\nGPU: [8-15]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer2_expert1_tp [label="TP All-Reduce\nGPU: [8-15]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer2_expert2 [label="Expert 2\nGPU: [16-23]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer2_expert2_tp [label="TP All-Reduce\nGPU: [16-23]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer2_expert3 [label="Expert 3\nGPU: [24-31]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer2_expert3_tp [label="TP All-Reduce\nGPU: [24-31]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 2
		pp0_layer2_agg [label="Expert Aggregation\nGPU: [0-31]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp0_layer2_ep_comm [label="EP All-to-All\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Layer 3
		pp0_layer3_attn [label="Layer 3: Attention\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer3_tp_comm [label="TP All-Reduce\nGPU: [0-7], [8-15], [16-23], [24-31]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer3_moe [label="Layer 3: MoE Routing\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp0_layer3_gate [label="Expert Gate Selection\nGPU: [0-31]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 3
		pp0_layer3_expert0 [label="Expert 0\nGPU: [0-7]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer3_expert0_tp [label="TP All-Reduce\nGPU: [0-7]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer3_expert1 [label="Expert 1\nGPU: [8-15]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer3_expert1_tp [label="TP All-Reduce\nGPU: [8-15]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer3_expert2 [label="Expert 2\nGPU: [16-23]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer3_expert2_tp [label="TP All-Reduce\nGPU: [16-23]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp0_layer3_expert3 [label="Expert 3\nGPU: [24-31]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp0_layer3_expert3_tp [label="TP All-Reduce\nGPU: [24-31]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 3
		pp0_layer3_agg [label="Expert Aggregation\nGPU: [0-31]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
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
		
		// Layer 5
		pp1_layer5_attn [label="Layer 5: Attention\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer5_tp_comm [label="TP All-Reduce\nGPU: [128-135], [136-143], [144-151], [152-159]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer5_moe [label="Layer 5: MoE Routing\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp1_layer5_gate [label="Expert Gate Selection\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 5
		pp1_layer5_expert0 [label="Expert 0\nGPU: [128-135]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer5_expert0_tp [label="TP All-Reduce\nGPU: [128-135]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer5_expert1 [label="Expert 1\nGPU: [136-143]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer5_expert1_tp [label="TP All-Reduce\nGPU: [136-143]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer5_expert2 [label="Expert 2\nGPU: [144-151]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer5_expert2_tp [label="TP All-Reduce\nGPU: [144-151]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer5_expert3 [label="Expert 3\nGPU: [152-159]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer5_expert3_tp [label="TP All-Reduce\nGPU: [152-159]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 5
		pp1_layer5_agg [label="Expert Aggregation\nGPU: [128-159]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp1_layer5_ep_comm [label="EP All-to-All\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Layer 6
		pp1_layer6_attn [label="Layer 6: Attention\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer6_tp_comm [label="TP All-Reduce\nGPU: [128-135], [136-143], [144-151], [152-159]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer6_moe [label="Layer 6: MoE Routing\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp1_layer6_gate [label="Expert Gate Selection\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 6
		pp1_layer6_expert0 [label="Expert 0\nGPU: [128-135]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer6_expert0_tp [label="TP All-Reduce\nGPU: [128-135]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer6_expert1 [label="Expert 1\nGPU: [136-143]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer6_expert1_tp [label="TP All-Reduce\nGPU: [136-143]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer6_expert2 [label="Expert 2\nGPU: [144-151]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer6_expert2_tp [label="TP All-Reduce\nGPU: [144-151]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer6_expert3 [label="Expert 3\nGPU: [152-159]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer6_expert3_tp [label="TP All-Reduce\nGPU: [152-159]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 6
		pp1_layer6_agg [label="Expert Aggregation\nGPU: [128-159]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp1_layer6_ep_comm [label="EP All-to-All\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Layer 7
		pp1_layer7_attn [label="Layer 7: Attention\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer7_tp_comm [label="TP All-Reduce\nGPU: [128-135], [136-143], [144-151], [152-159]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer7_moe [label="Layer 7: MoE Routing\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp1_layer7_gate [label="Expert Gate Selection\nGPU: [128-159]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 7
		pp1_layer7_expert0 [label="Expert 0\nGPU: [128-135]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer7_expert0_tp [label="TP All-Reduce\nGPU: [128-135]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer7_expert1 [label="Expert 1\nGPU: [136-143]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer7_expert1_tp [label="TP All-Reduce\nGPU: [136-143]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer7_expert2 [label="Expert 2\nGPU: [144-151]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer7_expert2_tp [label="TP All-Reduce\nGPU: [144-151]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp1_layer7_expert3 [label="Expert 3\nGPU: [152-159]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp1_layer7_expert3_tp [label="TP All-Reduce\nGPU: [152-159]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 7
		pp1_layer7_agg [label="Expert Aggregation\nGPU: [128-159]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
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
		
		// Layer 9
		pp2_layer9_attn [label="Layer 9: Attention\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer9_tp_comm [label="TP All-Reduce\nGPU: [256-263], [264-271], [272-279], [280-287]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer9_moe [label="Layer 9: MoE Routing\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp2_layer9_gate [label="Expert Gate Selection\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 9
		pp2_layer9_expert0 [label="Expert 0\nGPU: [256-263]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer9_expert0_tp [label="TP All-Reduce\nGPU: [256-263]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer9_expert1 [label="Expert 1\nGPU: [264-271]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer9_expert1_tp [label="TP All-Reduce\nGPU: [264-271]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer9_expert2 [label="Expert 2\nGPU: [272-279]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer9_expert2_tp [label="TP All-Reduce\nGPU: [272-279]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer9_expert3 [label="Expert 3\nGPU: [280-287]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer9_expert3_tp [label="TP All-Reduce\nGPU: [280-287]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 9
		pp2_layer9_agg [label="Expert Aggregation\nGPU: [256-287]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp2_layer9_ep_comm [label="EP All-to-All\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Layer 10
		pp2_layer10_attn [label="Layer 10: Attention\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer10_tp_comm [label="TP All-Reduce\nGPU: [256-263], [264-271], [272-279], [280-287]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer10_moe [label="Layer 10: MoE Routing\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp2_layer10_gate [label="Expert Gate Selection\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 10
		pp2_layer10_expert0 [label="Expert 0\nGPU: [256-263]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer10_expert0_tp [label="TP All-Reduce\nGPU: [256-263]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer10_expert1 [label="Expert 1\nGPU: [264-271]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer10_expert1_tp [label="TP All-Reduce\nGPU: [264-271]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer10_expert2 [label="Expert 2\nGPU: [272-279]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer10_expert2_tp [label="TP All-Reduce\nGPU: [272-279]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer10_expert3 [label="Expert 3\nGPU: [280-287]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer10_expert3_tp [label="TP All-Reduce\nGPU: [280-287]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 10
		pp2_layer10_agg [label="Expert Aggregation\nGPU: [256-287]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp2_layer10_ep_comm [label="EP All-to-All\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Layer 11
		pp2_layer11_attn [label="Layer 11: Attention\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer11_tp_comm [label="TP All-Reduce\nGPU: [256-263], [264-271], [272-279], [280-287]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer11_moe [label="Layer 11: MoE Routing\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp2_layer11_gate [label="Expert Gate Selection\nGPU: [256-287]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 11
		pp2_layer11_expert0 [label="Expert 0\nGPU: [256-263]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer11_expert0_tp [label="TP All-Reduce\nGPU: [256-263]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer11_expert1 [label="Expert 1\nGPU: [264-271]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer11_expert1_tp [label="TP All-Reduce\nGPU: [264-271]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer11_expert2 [label="Expert 2\nGPU: [272-279]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer11_expert2_tp [label="TP All-Reduce\nGPU: [272-279]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp2_layer11_expert3 [label="Expert 3\nGPU: [280-287]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp2_layer11_expert3_tp [label="TP All-Reduce\nGPU: [280-287]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		
		// Expert Aggregation and EP Communication for Layer 11
		pp2_layer11_agg [label="Expert Aggregation\nGPU: [256-287]\nInput: [batch_size=~2, seq_len=1024, hidden=1024] × 4\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
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
		
		// Layer 13
		pp3_layer13_attn [label="Layer 13: Attention\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp3_layer13_tp_comm [label="TP All-Reduce\nGPU: [384-391], [392-399], [400-407], [408-415]\nInput: [batch_size=32, seq_len=1024, hidden=128]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp3_layer13_moe [label="Layer 13: MoE Routing\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: [batch_size=32, seq_len=1024, hidden=1024]" fillcolor=lightyellow shape=parallelogram]
		pp3_layer13_gate [label="Expert Gate Selection\nGPU: [384-415]\nInput: [batch_size=32, seq_len=1024, hidden=1024]\nOutput: routing decisions" fillcolor=lightyellow shape=parallelogram style=dashed]
		
		// Expert 0-3 for Layer 13
		pp3_layer13_expert0 [label="Expert 0\nGPU: [384-391]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp3_layer13_expert0_tp [label="TP All-Reduce\nGPU: [384-391]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp3_layer13_expert1 [label="Expert 1\nGPU: [392-399]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp3_layer13_expert1_tp [label="TP All-Reduce\nGPU: [392-399]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp3_layer13_expert2 [label="Expert 2\nGPU: [400-407]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightblue shape=rectangle]
		pp3_layer13_expert2_tp [label="TP All-Reduce\nGPU: [400-407]\nInput: [batch_size=~2, seq_len=1024, hidden=256]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor=lightgreen shape=ellipse]
		pp3_layer13_expert3 [label="Expert 3\nGPU: [408-415]\nInput: [batch_size=~2,LM: [408-415]\nInput: [batch_size=~2, seq_len=1024, hidden=1024]\nOutput: [batch_size=~2, seq_len=1024, hidden=1024]" fillcolor