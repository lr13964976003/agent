#!/usr/bin/env python3
"""
Generate operator-level DAG for the hybrid tensor-parallel pipeline deployment.
Stages:
  Stage 0 (GPU0): Input -> Embedding -> Split for pipeline
  Stage 1 (GPU1+GPU2): Tensor-parallel Expert (column-parallel first linear, row-parallel second linear)
  Stage 2 (GPU0): Aggregate -> Output
All dimensions are fully specified and aligned.
"""

import subprocess
import os

dot_content = """
digraph LLM_Hybrid_Tensor_Pipeline_Deployment {
    rankdir=TB;
    splines=polyline;
    node [shape=rectangle, style=filled, fillcolor=lightblue];
    
    // ---------- Input ----------
    Input [shape=ellipse, label="Input\\nGPU: any\\nInput: [batch_size=1, seq_len=1024]\\nOutput: [batch_size=1, seq_len=1024]"];
    
    // ---------- Stage 0: GPU0 Embedding ----------
    Embedding [shape=rectangle, label="Embedding\\nGPU: 0\\nInput: [batch_size=1, seq_len=1024]\\nOutput: [batch_size=1, seq_len=1024, hidden=4096]"];
    
    // ---------- Pipeline Send (GPU0 -> GPU1/2) ----------
    PipeSend0 [shape=parallelogram, label="PipelineSend\\nGPU: 0->1/2\\nInput: [batch_size=1, seq_len=1024, hidden=4096]\\nOutput: [batch_size=1, seq_len=1024, hidden=4096]"];
    
    // ---------- Stage 1: GPU1 & GPU2 Tensor-Parallel Expert ----------
    // Gate on GPU1 (dashed selection line)
    Gate [shape=rectangle, label="Gate\\nGPU: 1\\nInput: [batch_size=1, seq_len=1024, hidden=4096]\\nOutput: [batch_size=1, seq_len=1024, hidden=4096]"];
    
    // Split hidden for column-parallel first linear
    SplitHidden [shape=parallelogram, label="SplitHiddenDim\\nGPU: 1&2\\nInput: [batch_size=1, seq_len=1024, hidden=4096]\\nOutput: [batch_size=1, seq_len=1024, hidden=2048]"];
    
    // Column-parallel first linear (GPU1 & GPU2)
    Linear1_GPU1 [shape=rectangle, label="Linear1_GPU1\\nGPU: 1\\nInput: [batch_size=1, seq_len=1024, hidden=2048]\\nOutput: [batch_size=1, seq_len=1024, ffn=2048]"];
    Linear1_GPU2 [shape=rectangle, label="Linear1_GPU2\\nGPU: 2\\nInput: [batch_size=1, seq_len=1024, hidden=2048]\\nOutput: [batch_size=1, seq_len=1024, ffn=2048]"];
    
    // All-gather intermediate to both GPUs
    AllGatherIntermediate [shape=ellipse, label="AllGather\\nGPU: 1<->2\\nInput: [batch_size=1, seq_len=1024, ffn=2048]\\nOutput: [batch_size=1, seq_len=1024, ffn=4096]"];
    
    // GELU activation (identical on both GPUs)
    GELU_GPU1 [shape=rectangle, label="GELU_GPU1\\nGPU: 1\\nInput: [batch_size=1, seq_len=1024, ffn=4096]\\nOutput: [batch_size=1, seq_len=1024, ffn=4096]"];
    GELU_GPU2 [shape=rectangle, label="GELU_GPU2\\nGPU: 2\\nInput: [batch_size=1, seq_len=1024, ffn=4096]\\nOutput: [batch_size=1, seq_len=1024, ffn=4096]"];
    
    // Split ffn for row-parallel second linear
    SplitFFN [shape=parallelogram, label="SplitFFNDim\\nGPU: 1&2\\nInput: [batch_size=1, seq_len=1024, ffn=4096]\\nOutput: [batch_size=1, seq_len=1024, ffn=2048]"];
    
    // Row-parallel second linear (GPU1 & GPU2)
    Linear2_GPU1 [shape=rectangle, label="Linear2_GPU1\\nGPU: 1\\nInput: [batch_size=1, seq_len=1024, ffn=2048]\\nOutput: [batch_size=1, seq_len=1024, hidden=4096]"];
    Linear2_GPU2 [shape=rectangle, label="Linear2_GPU2\\nGPU: 2\\nInput: [batch_size=1, seq_len=1024, ffn=2048]\\nOutput: [batch_size=1, seq_len=1024, hidden=4096]"];
    
    // All-reduce sum for row-parallel output
    AllReduceOutput [shape=ellipse, label="AllReduceSum\\nGPU: 1<->2\\nInput: [batch_size=1, seq_len=1024, hidden=4096]\\nOutput: [batch_size=1, seq_len=1024, hidden=4096]"];
    
    // Residual add (two inputs: branch and main)
    ResidualAdd [shape=rectangle, label="ResidualAdd\\nGPU: 1&2\\nInput: [batch_size=1, seq_len=1024, hidden=4096], [batch_size=1, seq_len=1024, hidden=4096]\\nOutput: [batch_size=1, seq_len=1024, hidden=4096]"];
    
    // ---------- Pipeline Send (GPU1/2 -> GPU0) ----------
    PipeSend1 [shape=parallelogram, label="PipelineSend\\nGPU: 1/2->0\\nInput: [batch_size=1, seq_len=1024, hidden=4096]\\nOutput: [batch_size=1, seq_len=1024, hidden=4096]"];
    
    // ---------- Stage 2: GPU0 Aggregation + Output ----------
    Aggregate [shape=parallelogram, label="Aggregate\\nGPU: 0\\nInput: [batch_size=1, seq_len=1024, hidden=4096]\\nOutput: [batch_size=1, seq_len=1024, hidden=4096]"];
    
    Output [shape=ellipse, label="Output\\nGPU: 0\\nInput: [batch_size=1, seq_len=1024, hidden=4096]\\nOutput: [batch_size=1, seq_len=1024, vocab=32000]"];
    
    // ---------- Edges ----------
    Input -> Embedding;
    Embedding -> PipeSend0;
    PipeSend0 -> Gate [label="GPU1"];
    PipeSend0 -> SplitHidden [label="GPU2"];
    
    Gate -> SplitHidden [style=dashed, label="selection"];
    SplitHidden -> Linear1_GPU1;
    SplitHidden -> Linear1_GPU2;
    
    Linear1_GPU1 -> AllGatherIntermediate;
    Linear1_GPU2 -> AllGatherIntermediate;
    
    AllGatherIntermediate -> GELU_GPU1;
    AllGatherIntermediate -> GELU_GPU2;
    
    GELU_GPU1 -> SplitFFN;
    GELU_GPU2 -> SplitFFN;
    
    SplitFFN -> Linear2_GPU1;
    SplitFFN -> Linear2_GPU2;
    
    Linear2_GPU1 -> AllReduceOutput;
    Linear2_GPU2 -> AllReduceOutput;
    
    // Residual connections (two inputs)
    AllReduceOutput -> ResidualAdd;
    PipeSend0 -> ResidualAdd [label="residual"];
    
    ResidualAdd -> PipeSend1;
    PipeSend1 -> Aggregate;
    Aggregate -> Output;
}
"""

# Ensure output directory exists
os.makedirs("../outputs/2025-12-03-11-11-34", exist_ok=True)

# Write DOT file
dot_path = "../outputs/2025-12-03-11-11-34/hybrid_tensor_pipeline_deployment.dot"
with open(dot_path, "w") as f:
    f.write(dot_content)

# Generate SVG via dot command
svg_path = "../outputs/2025-12-03-11-11-34/hybrid_tensor_pipeline_deployment.svg"
subprocess.run(["dot", "-Tsvg", dot_path, "-o", svg_path], check=True)

print("Generated files:")
print("  DOT:", dot_path)
print("  SVG:", svg_path)
