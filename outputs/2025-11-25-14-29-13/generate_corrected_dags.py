#!/usr/bin/env python3
"""
Generate corrected DAGs for MoE Baseline and Proposed Models
Addresses redundancy issues and adds proper annotations
"""

import os

class DagGenerator:
    def __init__(self):
        self.output_dir = "../outputs/2025-11-25-14-29-13"
        
    def generate_baseline_dag(self):
        """Generate concise baseline DAG with TP=8, PP=2"""
        dag_content = '''digraph MoE_Baseline_TP8_PP2 {
    rankdir=TB;
    node [shape=rectangle, style="rounded,filled"];
    
    // Global graph attributes
    label="MoE Baseline Model DAG (TP=8, PP=2): 16 Layers, Shared Experts";
    labelloc="t";
    fontsize=20;
    
    // Input node
    input [label="Input\\nShape: [batch_size=128, seq_len=10000, d_model=4096]\\nGPU: All GPUs", 
           shape=ellipse, fillcolor=lightblue];
    
    // Stage 0: Layers 0-7 (GPUs 0-7)
    subgraph cluster_stage0 {
        label="Stage 0\\nLayers 0-7\\nGPUs: 0-7";
        style="dashed";
        
        // Layer template for Stage 0
        subgraph cluster_layer_template_s0 {
            label="Layer Template (All Layers 0-7)";
            style="dotted";
            
            ln1_s0 [label="LayerNorm\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 0-7", 
                   fillcolor=lightyellow];
            
            attn_s0 [label="Multi-Head Attention\\nHeads: 32, d_k: 128\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 0-7", 
                    fillcolor=lightcoral];
            
            residual1_s0 [label="Residual Add\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 0-7", 
                          shape=parallelogram, fillcolor=lightgray];
            
            ln2_s0 [label="LayerNorm\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 0-7", 
                   fillcolor=lightyellow];
            
            // Expert processing (shared across GPUs)
            gate_s0 [label="Expert Gate\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 0-7", 
                     shape=parallelogram, fillcolor=lightgreen];
            
            experts_s0 [label="16 Shared Experts\\nShape: [128, 10000, 16384]→[128, 10000, 4096]\\nGPU: All GPUs 0-7\\nMemory: 256MB/expert", 
                        fillcolor=lightpink];
            
            residual2_s0 [label="Residual Add\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 0-7", 
                          shape=parallelogram, fillcolor=lightgray];
            
            // Connections within layer
            ln1_s0 -> attn_s0;
            attn_s0 -> residual1_s0;
            residual1_s0 -> ln2_s0;
            ln2_s0 -> gate_s0;
            gate_s0 -> experts_s0;
            experts_s0 -> residual2_s0;
        }
        
        note_s0 [label="16 Layers (0-7)\\nEach Layer: Same Structure\\nTensor Parallelism: 8-way", 
                shape=note, style="dotted"];
    }
    
    // Stage 1: Layers 8-15 (GPUs 8-15)
    subgraph cluster_stage1 {
        label="Stage 1\\nLayers 8-15\\nGPUs: 8-15";
        style="dashed";
        
        // Layer template for Stage 1
        subgraph cluster_layer_template_s1 {
            label="Layer Template (All Layers 8-15)";
            style="dotted";
            
            ln1_s1 [label="LayerNorm\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 8-15", 
                   fillcolor=lightyellow];
            
            attn_s1 [label="Multi-Head Attention\\nHeads: 32, d_k: 128\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 8-15", 
                    fillcolor=lightcoral];
            
            residual1_s1 [label="Residual Add\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 8-15", 
                          shape=parallelogram, fillcolor=lightgray];
            
            ln2_s1 [label="LayerNorm\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 8-15", 
                   fillcolor=lightyellow];
            
            // Expert processing (shared across GPUs)
            gate_s1 [label="Expert Gate\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 8-15", 
                     shape=parallelogram, fillcolor=lightgreen];
            
            experts_s1 [label="16 Shared Experts\\nShape: [128, 10000, 16384]→[128, 10000, 4096]\\nGPU: All GPUs 8-15\\nMemory: 256MB/expert", 
                        fillcolor=lightpink];
            
            residual2_s1 [label="Residual Add\\nShape: [128, 10000, 4096]\\nGPU: All GPUs 8-15", 
                          shape=parallelogram, fillcolor=lightgray];
            
            // Connections within layer
            ln1_s1 -> attn_s1;
            attn_s1 -> residual1_s1;
            residual1_s1 -> ln2_s1;
            ln2_s1 -> gate_s1;
            gate_s1 -> experts_s1;
            experts_s1 -> residual2_s1;
        }
        
        note_s1 [label="16 Layers (8-15)\\nEach Layer: Same Structure\\nTensor Parallelism: 8-way", 
                shape=note, style="dotted"];
    }
    
    // Pipeline connections
    pipeline_comm [label="Pipeline Communication\\nShape: [128, 10000, 4096]\\nBetween Stage 0 & 1", 
                  shape=ellipse, fillcolor=orange];
    
    // Output node
    output [label="Output\\nShape: [batch_size=128, seq_len=10000, d_model=4096]\\nGPU: All GPUs", 
            shape=ellipse, fillcolor=lightblue];
    
    // High-level connections
    input -> ln1_s0;
    residual2_s0 -> pipeline_comm [label="After 8 layers"];
    pipeline_comm -> ln1_s1;
    residual2_s1 -> output [label="After 16 layers"];
    
    // Constraint for rank
    {rank=same; input; output;}
}'''
        
        with open(f"{self.output_dir}/baseline_model_corrected.dot", "w") as f:
            f.write(dag_content)
            
    def generate_proposed_dag(self):
        """Generate concise proposed DAG with EP=16"""
        dag_content = '''digraph MoE_Proposed_EP16 {
    rankdir=TB;
    node [shape=rectangle, style="rounded,filled"];
    
    // Global graph attributes
    label="MoE Proposed Model DAG (EP=16): 16 Layers, Expert Parallelism";
    labelloc="t";
    fontsize=20;
    
    // Input node
    input [label="Input\\nShape: [batch_size=128, seq_len=10000, d_model=4096]\\nGPU: All GPUs", 
           shape=ellipse, fillcolor=lightblue];
    
    // Layer template (one representative layer shown)
    subgraph cluster_layer_template {
        label="Layer Template (All 16 Layers)";
        style="dashed";
        
        ln1 [label="LayerNorm\\nShape: [128, 10000, 4096]\\nGPU: All GPUs", 
             fillcolor=lightyellow];
        
        attn [label="Multi-Head Attention\\nHeads: 32, d_k: 128\\nShape: [128, 10000, 4096]\\nGPU: All GPUs", 
              fillcolor=lightcoral];
        
        residual1 [label="Residual Add\\nShape: [128, 10000, 4096]\\nGPU: All GPUs", 
                   shape=parallelogram, fillcolor=lightgray];
        
        ln2 [label="LayerNorm\\nShape: [128, 10000, 4096]\\nGPU: All GPUs", 
             fillcolor=lightyellow];
        
        gate [label="Expert Gate\\nShape: [128, 10000, 4096]\\nGPU: All GPUs\\nTop-K=2", 
              shape=parallelogram, fillcolor=lightgreen];
        
        // Expert distribution across GPUs
        subgraph cluster_expert_distribution {
            label="Expert Distribution (16 Experts Across GPUs)";
            style="dotted";
            
            expert_0 [label="Expert 0\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 0\\nMemory: 4GB", 
                     fillcolor=lightpink];
            expert_1 [label="Expert 1\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 1\\nMemory: 4GB", 
                     fillcolor=lightpink];
            expert_2 [label="Expert 2\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 2\\nMemory: 4GB", 
                     fillcolor=lightpink];
            expert_3 [label="Expert 3\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 3\\nMemory: 4GB", 
                     fillcolor=lightpink];
            expert_4 [label="Expert 4\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 4\\nMemory: 4GB", 
                     fillcolor=lightpink];
            expert_5 [label="Expert 5\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 5\\nMemory: 4GB", 
                     fillcolor=lightpink];
            expert_6 [label="Expert 6\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 6\\nMemory: 4GB", 
                     fillcolor=lightpink];
            expert_7 [label="Expert 7\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 7\\nMemory: 4GB", 
                     fillcolor=lightpink];
            expert_8 [label="Expert 8\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 8\\nMemory: 4GB", 
                     fillcolor=lightpink];
            expert_9 [label="Expert 9\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 9\\nMemory: 4GB", 
                     fillcolor=lightpink];
            expert_10 [label="Expert 10\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 10\\nMemory: 4GB", 
                      fillcolor=lightpink];
            expert_11 [label="Expert 11\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 11\\nMemory: 4GB", 
                      fillcolor=lightpink];
            expert_12 [label="Expert 12\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 12\\nMemory: 4GB", 
                      fillcolor=lightpink];
            expert_13 [label="Expert 13\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 13\\nMemory: 4GB", 
                      fillcolor=lightpink];
            expert_14 [label="Expert 14\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 14\\nMemory: 4GB", 
                      fillcolor=lightpink];
            expert_15 [label="Expert 15\\nShape: [tokens, 4096]→[tokens, 4096]\\nGPU: 15\\nMemory: 4GB", 
                      fillcolor=lightpink];
        }
        
        // Routing and aggregation
        route_tokens [label="Token Routing\\nShape: [tokens, 4096]\\nGPU: All GPUs\\nAsync: Enabled", 
                      shape=ellipse, fillcolor=orange];
        
        aggregate_tokens [label="Token Aggregation\\nShape: [128, 10000, 4096]\\nGPU: All GPUs", 
                          shape=ellipse, fillcolor=orange];
        
        residual2 [label="Residual Add\\nShape: [128, 10000, 4096]\\nGPU: All GPUs", 
                   shape=parallelogram, fillcolor=lightgray];
        
        // Connections within layer
        ln1 -> attn;
        attn -> residual1;
        residual1 -> ln2;
        ln2 -> gate;
        gate -> route_tokens;
        
        // Routing to experts (dashed lines for dynamic routing)
        route_tokens -> expert_0 [style=dashed, label="Top-K routing"];
        route_tokens -> expert_1 [style=dashed];
        route_tokens -> expert_2 [style=dashed];
        route_tokens -> expert_3 [style=dashed];
        route_tokens -> expert_4 [style=dashed];
        route_tokens -> expert_5 [style=dashed];
        route_tokens -> expert_6 [style=dashed];
        route_tokens -> expert_7 [style=dashed];
        route_tokens -> expert_8 [style=dashed];
        route_tokens -> expert_9 [style=dashed];
        route_tokens -> expert_10 [style=dashed];
        route_tokens -> expert_11 [style=dashed];
        route_tokens -> expert_12 [style=dashed];
        route_tokens -> expert_13 [style=dashed];
        route_tokens -> expert_14 [style=dashed];
        route_tokens -> expert_15 [style=dashed];
        
        // Aggregation back from experts
        expert_0 -> aggregate_tokens [style=dashed];
        expert_1 -> aggregate_tokens [style=dashed];
        expert_2 -> aggregate_tokens [style=dashed];
        expert_3 -> aggregate_tokens [style=dashed];
        expert_4 -> aggregate_tokens [style=dashed];
        expert_5 -> aggregate_tokens [style=dashed];
        expert_6 -> aggregate_tokens [style=dashed];
        expert_7 -> aggregate_tokens [style=dashed];
        expert_8 -> aggregate_tokens [style=dashed];
        expert_9 -> aggregate_tokens [style=dashed];
        expert_10 -> aggregate_tokens [style=dashed];
        expert_11 -> aggregate_tokens [style=dashed];
        expert_12 -> aggregate_tokens [style=dashed];
        expert_13 -> aggregate_tokens [style=dashed];
        expert_14 -> aggregate_tokens [style=dashed];
        expert_15 -> aggregate_tokens [style=dashed];
        
        aggregate_tokens -> residual2;
    }
    
    layer_note [label="16 Layers Total\\nEach Layer: Same EP-16 Structure\\n1 Expert per GPU per Layer", 
                shape=note, style="dotted"];
    
    // Output node
    output [label="Output\\nShape: [batch_size=128, seq_len=10000, d_model=4096]\\nGPU: All GPUs", 
            shape=ellipse, fillcolor=lightblue];
    
    // High-level connections
    input -> ln1;
    residual2 -> output [label="After 16 layers"];
    
    // Constraint for rank
    {rank=same; input; output;}
}'''
        
        with open(f"{self.output_dir}/proposed_model_corrected.dot", "w") as f:
            f.write(dag_content)
            
    def generate_svg_files(self):
        """Generate SVG files using Graphviz"""
        commands = [
            f"dot -Tsvg {self.output_dir}/baseline_model_corrected.dot -o {self.output_dir}/baseline_model_corrected.svg",
            f"dot -Tsvg {self.output_dir}/proposed_model_corrected.dot -o {self.output_dir}/proposed_model_corrected.svg"
        ]
        
        for cmd in commands:
            os.system(cmd)
            
    def run(self):
        """Generate all corrected DAGs"""
        self.generate_baseline_dag()
        self.generate_proposed_dag()
        self.generate_svg_files()
        print("Corrected DAGs generated successfully!")

if __name__ == "__main__":
    generator = DagGenerator()
    generator.run()