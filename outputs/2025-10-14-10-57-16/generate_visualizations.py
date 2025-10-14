#!/usr/bin/env python3

import graphviz
import os
from pathlib import Path

def create_dag_visualization():
    """Create DAG visualizations for the improved tensor parallelism strategy"""
    
    # Create output directory
    output_dir = Path("./outputs/2025-10-14-10-57-16")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the DOT file
    with open(output_dir / "improved_tensor_parallel_dag.dot", "r") as f:
        dot_content = f.read()
    
    # Create Graphviz source
    src = graphviz.Source(dot_content)
    
    # Render as SVG
    src.render(output_dir / "improved_tensor_parallel_dag", format='svg', cleanup=False)
    
    print(f"Generated SVG visualization: {output_dir}/improved_tensor_parallel_dag.svg")
    
    # Also create a high-level overview DAG
    create_overview_dag(output_dir)
    
    # Create detailed layer breakdown
    create_layer_breakdown(output_dir)

def create_overview_dag(output_dir):
    """Create a high-level overview of the deployment strategy"""
    
    overview_dot = """
    digraph {
        rankdir=LR
        size="20,10"
        node [shape=box, style=filled, fillcolor=lightblue]
        
        // Input
        input [label="Model Input\n[1024, 10000, 8192]"]
        
        // Stage 0: GPUs 0-7
        subgraph cluster_stage0 {
            label="Stage 0: Layers 0-7\nGPUs 0-7 (8-way Tensor Parallel)"
            style=filled
            fillcolor=lightyellow
            
            layer0 [label="Layer 0\nTensor Parallel"]
            layer1 [label="Layer 1\nTensor Parallel"]
            layer2 [label="Layer 2\nTensor Parallel"]
            layer3 [label="Layer 3\nTensor Parallel"]
            layer4 [label="Layer 4\nTensor Parallel"]
            layer5 [label="Layer 5\nTensor Parallel"]
            layer6 [label="Layer 6\nTensor Parallel"]
            layer7 [label="Layer 7\nTensor Parallel"]
            
            layer0 -> layer1 -> layer2 -> layer3 -> layer4 -> layer5 -> layer6 -> layer7
        }
        
        // Stage 1: GPUs 8-15
        subgraph cluster_stage1 {
            label="Stage 1: Layers 8-15\nGPUs 8-15 (8-way Tensor Parallel)"
            style=filled
            fillcolor=lightgreen
            
            layer8 [label="Layer 8\nTensor Parallel"]
            layer9 [label="Layer 9\nTensor Parallel"]
            layer10 [label="Layer 10\nTensor Parallel"]
            layer11 [label="Layer 11\nTensor Parallel"]
            layer12 [label="Layer 12\nTensor Parallel"]
            layer13 [label="Layer 13\nTensor Parallel"]
            layer14 [label="Layer 14\nTensor Parallel"]
            layer15 [label="Layer 15\nTensor Parallel"]
            
            layer8 -> layer9 -> layer10 -> layer11 -> layer12 -> layer13 -> layer14 -> layer15
        }
        
        // Output
        output [label="Model Output\n[1024, 10000, 8192]"]
        
        // Connections
        input -> layer0
        layer7 -> layer8 [label="Pipeline Communication"]
        layer15 -> output
    }
    """
    
    src = graphviz.Source(overview_dot)
    src.render(output_dir / "deployment_overview", format='svg', cleanup=False)
    print(f"Generated overview visualization: {output_dir}/deployment_overview.svg")

def create_layer_breakdown(output_dir):
    """Create detailed breakdown of a single layer"""
    
    layer_dot = """
    digraph {
        rankdir=TB
        size="25,20"
        node [shape=box, style=filled]
        
        // Layer input
        input [label="Layer Input\n[1024, 10000, 8192]", fillcolor=lightblue]
        
        // LayerNorm
        ln1 [label="LayerNorm1\n[1024, 10000, 8192]", fillcolor=lightgreen]
        
        // QKV Linear - Column Parallel
        subgraph cluster_qkv {
            label="QKV Linear (Column Parallel)"
            style=filled
            fillcolor=lightyellow
            
            split_qkv [label="Split Input\n8-way parallel", shape=parallelogram]
            q_linear [label="Query Linear\n[1024, 10000, 1024]", fillcolor=orange]
            k_linear [label="Key Linear\n[1024, 10000, 1024]", fillcolor=orange]
            v_linear [label="Value Linear\n[1024, 10000, 1024]", fillcolor=orange]
        }
        
        // Attention
        subgraph cluster_attn {
            label="Multi-Head Attention"
            style=filled
            fillcolor=lightcyan
            
            attn_score [label="Q·K^T\n[1024, 2, 10000, 10000]", fillcolor=lightgreen]
            softmax [label="Softmax\n[1024, 2, 10000, 10000]", fillcolor=lightgreen]
            attn_out [label="Attention·V\n[1024, 10000, 1024]", fillcolor=lightgreen]
        }
        
        // Gather and Project
        gather_attn [label="Gather Outputs\n8-way → 1", shape=parallelogram]
        split_proj [label="Split for Projection\n8-way parallel", shape=parallelogram]
        attn_proj [label="Output Projection\n[1024, 10000, 1024]", fillcolor=orange]
        allreduce_proj [label="All-Reduce Sum\n8-way → 1", shape=parallelogram, fillcolor=red]
        
        // Residual 1
        res1 [label="Residual Add 1\n[1024, 10000, 8192]", fillcolor=lightgreen]
        
        // LayerNorm2
        ln2 [label="LayerNorm2\n[1024, 10000, 8192]", fillcolor=lightgreen]
        
        // MLP - Column + Row Parallel
        subgraph cluster_mlp {
            label="MLP (Column + Row Parallel)"
            style=filled
            fillcolor=lightpink
            
            split_mlp [label="Split Input\n8-way parallel", shape=parallelogram]
            mlp1 [label="Linear1 (Column)\n[1024, 10000, 4096]", fillcolor=orange]
            gelu [label="GELU\n[1024, 10000, 4096]", fillcolor=lightgreen]
            mlp2 [label="Linear2 (Row)\n[1024, 10000, 1024]", fillcolor=orange]
            allreduce_mlp [label="All-Reduce Sum\n8-way → 1", shape=parallelogram, fillcolor=red]
        }
        
        // Final residual
        res2 [label="Residual Add 2\n[1024, 10000, 8192]", fillcolor=lightgreen]
        
        // Output
        output [label="Layer Output\n[1024, 10000, 8192]", fillcolor=lightblue]
        
        // Connections
        input -> ln1
        ln1 -> split_qkv
        split_qkv -> q_linear
        split_qkv -> k_linear
        split_qkv -> v_linear
        q_linear -> attn_score
        k_linear -> attn_score
        attn_score -> softmax
        softmax -> attn_out
        v_linear -> attn_out
        attn_out -> gather_attn
        gather_attn -> split_proj
        split_proj -> attn_proj
        attn_proj -> allreduce_proj
        allreduce_proj -> res1
        input -> res1 [style=dashed]
        res1 -> ln2
        ln2 -> split_mlp
        split_mlp -> mlp1
        mlp1 -> gelu
        gelu -> mlp2
        mlp2 -> allreduce_mlp
        allreduce_mlp -> res2
        res1 -> res2 [style=dashed]
        res2 -> output
    }
    """
    
    src = graphviz.Source(layer_dot)
    src.render(output_dir / "layer_detailed_breakdown", format='svg', cleanup=False)
    print(f"Generated layer breakdown visualization: {output_dir}/layer_detailed_breakdown.svg")

if __name__ == "__main__":
    create_dag_visualization()