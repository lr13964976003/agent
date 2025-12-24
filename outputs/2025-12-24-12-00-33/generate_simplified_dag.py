#!/usr/bin/env python3

def generate_simplified_dag():
    """Generate simplified LLM parallel strategy deployment DAG for overview visualization"""
    
    dot_content = '''// Simplified LLM Parallel Strategy Deployment DAG
// Focus on high-level communication patterns and phase transitions
digraph {
	dpi=300 rankdir=TB size="30,20"
	node [fontsize=12 margin="0.2,0.1"]
	
	// Styling
	node [fillcolor=lightblue shape=ellipse style=filled]
	node [fillcolor=lightgreen shape=rectangle style=filled]
	node [fillcolor=lightyellow shape=parallelogram style=filled]
	
	// INPUT
	input [label="INPUT\\n[batch=4, seq=10240, hidden=512]" fillcolor=white penwidth=3 shape=ellipse style=filled]
	
	// PREFILL PHASE - High Level
	prefill_start [label="PREFILL PHASE\\nPP=4, EP=4, TP=2, SP=2\\n64 GPUs" fillcolor=red fontsize=14 shape=box style=filled]
	
	// Pipeline stages as single nodes for overview
	prefill_stage1 [label="PIPELINE STAGE 1\\nLayers 1-4\\nGPUs 0-15\\nAttention + MoE" fillcolor=lightgreen shape=rectangle style=filled]
	prefill_stage2 [label="PIPELINE STAGE 2\\nLayers 5-8\\nGPUs 16-31\\nAttention + MoE" fillcolor=lightgreen shape=rectangle style=filled]
	prefill_stage3 [label="PIPELINE STAGE 3\\nLayers 9-12\\nGPUs 32-47\\nAttention + MoE" fillcolor=lightgreen shape=rectangle style=filled]
	prefill_stage4 [label="PIPELINE STAGE 4\\nLayers 13-16\\nGPUs 48-63\\nAttention + MoE" fillcolor=lightgreen shape=rectangle style=filled]
	
	// Communication nodes for prefill
	sp_comm [label="SP Communication\\nAll-Gather/Split\\nAcross 64 GPUs" fillcolor=lightblue shape=ellipse style=filled]
	tp_comm [label="TP Communication\\nAll-Gather\\nAcross TP pairs" fillcolor=lightblue shape=ellipse style=filled]
	ep_comm [label="EP Communication\\nAll-to-All\\nExpert routing" fillcolor=lightblue shape=ellipse style=filled]
	pp_comm [label="PP Communication\\nPipeline bubbles\\nStage-to-stage" fillcolor=lightblue shape=ellipse style=filled]
	
	// DECODE PHASE
	decode_start [label="DECODE PHASE\\nPP=4, EP=4, TP=2, SP=1\\n32 GPUs" fillcolor=red fontsize=14 shape=box style=filled]
	decode_input [label="DECODE INPUT\\nSingle Token\\n[batch=4, seq=1, hidden=512]" fillcolor=pink penwidth=3 shape=ellipse style=filled]
	
	// Decode stages
	decode_stage1 [label="DECODE STAGE 1\\nLayers 1-4\\nGPUs 0-15\\nAttention + MoE" fillcolor=lightgreen shape=rectangle style=filled]
	decode_stage2 [label="DECODE STAGE 2\\nLayers 5-8\\nGPUs 16-31\\nAttention + MoE" fillcolor=lightgreen shape=rectangle style=filled]
	decode_stage3 [label="DECODE STAGE 3\\nLayers 9-12\\nGPUs 32-47\\nAttention + MoE" fillcolor=lightgreen shape=rectangle style=filled]
	decode_stage4 [label="DECODE STAGE 4\\nLayers 13-16\\nGPUs 48-63\\nAttention + MoE" fillcolor=lightgreen shape=rectangle style=filled]
	
	// Communication for decode
	decode_tp_comm [label="TP Communication\\nAll-Gather\\nAcross TP pairs" fillcolor=lightblue shape=ellipse style=filled]
	decode_ep_comm [label="EP Communication\\nAll-to-All\\nExpert routing" fillcolor=lightblue shape=ellipse style=filled]
	decode_pp_comm [label="PP Communication\\nPipeline bubbles\\nStage-to-stage" fillcolor=lightblue shape=ellipse style=filled]
	
	// OUTPUT
	final_output [label="FINAL OUTPUT\\n[batch=4, seq=10240, hidden=512]" fillcolor=white penwidth=3 shape=ellipse style=filled]
	decode_output [label="DECODE OUTPUT\\n[batch=4, seq=1, hidden=512]" fillcolor=pink penwidth=3 shape=ellipse style=filled]
	
	// High-level connections
	input -> prefill_stage1
	prefill_stage1 -> sp_comm -> prefill_stage2
	prefill_stage2 -> sp_comm -> prefill_stage3
	prefill_stage3 -> sp_comm -> prefill_stage4
	prefill_stage4 -> final_output
	
	// Communication patterns within stages
	prefill_stage1 -> tp_comm -> prefill_stage1
	prefill_stage1 -> ep_comm -> prefill_stage1
	prefill_stage2 -> tp_comm -> prefill_stage2
	prefill_stage2 -> ep_comm -> prefill_stage2
	prefill_stage3 -> tp_comm -> prefill_stage3
	prefill_stage3 -> ep_comm -> prefill_stage3
	prefill_stage4 -> tp_comm -> prefill_stage4
	prefill_stage4 -> ep_comm -> prefill_stage4
	
	// Decode connections
	decode_input -> decode_stage1
	decode_stage1 -> decode_stage2
	decode_stage2 -> decode_stage3
	decode_stage3 -> decode_stage4
	decode_stage4 -> decode_output
	
	// Decode communication
	decode_stage1 -> decode_tp_comm -> decode_stage1
	decode_stage1 -> decode_ep_comm -> decode_stage1
	decode_stage2 -> decode_tp_comm -> decode_stage2
	decode_stage2 -> decode_ep_comm -> decode_stage2
	decode_stage3 -> decode_tp_comm -> decode_stage3
	decode_stage3 -> decode_ep_comm -> decode_stage3
	decode_stage4 -> decode_tp_comm -> decode_stage4
	decode_stage4 -> decode_ep_comm -> decode_stage4
	
	// Phase relationship (dashed)
	final_output -> decode_input [style=dashed label="Next token"]
}
'''
    
    # Write the simplified DAG to file
    dot_file_path = "../outputs/2025-12-24-12-00-33/llm_parallel_simplified_fixed.dot"
    with open(dot_file_path, 'w') as f:
        f.write(dot_content)
    
    print(f"Simplified DAG written to: {dot_file_path}")
    
    # Generate SVG using graphviz
    try:
        import subprocess
        svg_file_path = "../outputs/2025-12-24-12-00-33/llm_parallel_simplified_fixed.svg"
        subprocess.run(['dot', '-Tsvg', dot_file_path, '-o', svg_file_path], check=True)
        print(f"Simplified SVG image generated: {svg_file_path}")
        return dot_file_path, svg_file_path
    except subprocess.CalledProcessError as e:
        print(f"Error generating simplified SVG: {e}")
        return dot_file_path, None
    except FileNotFoundError:
        print("Graphviz not found. Install with: apt-get install graphviz")
        return dot_file_path, None

if __name__ == "__main__":
    dot_path, svg_path = generate_simplified_dag()
    print(f"Simplified files generated:\nDOT: {dot_path}\nSVG: {svg_path}")