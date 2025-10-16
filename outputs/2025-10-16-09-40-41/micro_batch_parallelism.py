dag_dir="./outputs/2025-10-16-09-40-41"
import subprocess
import os

# Set the directory
dag_dir = "./outputs/2025-10-16-09-40-41"

# Create micro-batch parallelism DAG for improved TPS
micro_batch_dag = '''
digraph micro_batch_parallel_model {
    rankdir=LR
    size="40,30"
    
    node [fontname="Arial", fontsize=10]
    
    // Input with micro-batching
    input [label="Model Input\\n[batch=4096, seq=10000, dim=8192]\\nSplit into 4×1024", shape=parallelogram, fillcolor=lightgreen, style=filled]
    
    // Micro-batch split
    split [label="Micro-batch Split\\n[4096→4×1024]\\nGPU: 0-15", shape=parallelogram, fillcolor=lightsteelblue, style=filled]
    
    // 4 micro-batches processed in parallel
    subgraph cluster_microbatch0 {
        label="Micro-batch 0 (GPUs 0-3)";
        style=dashed;
        color=green;
        
        mba0 [label="MBA-0\\n[1024×10000×8192]\\nGPU: 0", shape=ellipse, fillcolor=lightcyan, style=filled]
        ln0_0 [label="LayerNorm\\nLayer 0\\nGPU: 0", shape=rectangle, fillcolor=lightyellow, style=filled]
        mha0_0 [label="MHA\\nOptimized\\nGPU: 0", shape=rectangle, fillcolor=lightpink, style=filled]
        mlp0_0 [label="MLP\\nTensor Parallel\\nGPU: 0", shape=rectangle, fillcolor=lightcoral, style=filled]
        
        ln1_0 [label="LayerNorm\\nLayer 1\\nGPU: 0", shape=rectangle, fillcolor=lightyellow, style=filled]
        mha1_0 [label="MHA\\nOptimized\\nGPU: 0", shape=rectangle, fillcolor=lightpink, style=filled]
        mlp1_0 [label="MLP\\nTensor Parallel\\nGPU: 0", shape=rectangle, fillcolor=lightcoral, style=filled]
    }
    
    subgraph cluster_microbatch1 {
        label="Micro-batch 1 (GPUs 4-7)";
        style=dashed;
        color=blue;
        
        mba1 [label="MBA-1\\n[1024×10000×8192]\\nGPU: 4", shape=ellipse, fillcolor=lightcyan, style=filled]
        ln0_1 [label="LayerNorm\\nLayer 0\\nGPU: 4", shape=rectangle, fillcolor=lightyellow, style=filled]
        mha0_1 [label="MHA\\nOptimized\\nGPU: 4", shape=rectangle, fillcolor=lightpink, style=filled]
        mlp0_1 [label="MLP\\nTensor Parallel\\nGPU: 4", shape=rectangle, fillcolor=lightcoral, style=filled]
        
        ln1_1 [label="LayerNorm\\nLayer 1\\nGPU: 4", shape=rectangle, fillcolor=lightyellow, style=filled]
        mha1_1 [label="MHA\\nOptimized\\nGPU: 4", shape=rectangle, fillcolor=lightpink, style=filled]
        mlp1_1 [label="MLP\\nTensor Parallel\\nGPU: 4", shape=rectangle, fillcolor=lightcoral, style=filled]
    }
    
    subgraph cluster_microbatch2 {
        label="Micro-batch 2 (GPUs 8-11)";
        style=dashed;
        color=red;
        
        mba2 [label="MBA-2\\n[1024×10000×8192]\\nGPU: 8", shape=ellipse, fillcolor=lightcyan, style=filled]
        ln0_2 [label="LayerNorm\\nLayer 0\\nGPU: 8", shape=rectangle, fillcolor=lightyellow, style=filled]
        mha0_2 [label="MHA\\nOptimized\\nGPU: 8", shape=rectangle, fillcolor=lightpink, style=filled]
        mlp0_2 [label="MLP\\nTensor Parallel\\nGPU: 8", shape=rectangle, fillcolor=lightcoral, style=filled]
        
        ln1_2 [label="LayerNorm\\nLayer 1\\nGPU: 8", shape=rectangle, fillcolor=lightyellow, style=filled]
        mha1_2 [label="MHA\\nOptimized\\nGPU: 8", shape=rectangle, fillcolor=lightpink, style=filled]
        mlp1_2 [label="MLP\\nTensor Parallel\\nGPU: 8", shape=rectangle, fillcolor=lightcoral, style=filled]
    }
    
    subgraph cluster_microbatch3 {
        label="Micro-batch 3 (GPUs 12-15)";
        style=dashed;
        color=purple;
        
        mba3 [label="MBA-3\\n[1024×10000×8192]\\nGPU: 12", shape=ellipse, fillcolor=lightcyan, style=filled]
        ln0_3 [label="LayerNorm\\nLayer 0\\nGPU: 12", shape=rectangle, fillcolor=lightyellow, style=filled]
        mha0_3 [label="MHA\\nOptimized\\nGPU: 12", shape=rectangle, fillcolor=lightpink, style=filled]
        mlp0_3 [label="MLP\\nTensor Parallel\\nGPU: 12", shape=rectangle, fillcolor=lightcoral, style=filled]
        
        ln1_3 [label="LayerNorm\\nLayer 1\\nGPU: 12", shape=rectangle, fillcolor=lightyellow, style=filled]
        mha1_3 [label="MHA\\nOptimized\\nGPU: 12", shape=rectangle, fillcolor=lightpink, style=filled]
        mlp1_3 [label="MLP\\nTensor Parallel\\nGPU: 12", shape=rectangle, fillcolor=lightcoral, style=filled]
    }
    
    // Aggregation
    gather [label="Gather Results\\n4×[1024×10000×8192]→[4096×10000×8192]\\nGPU: 0-15", shape=parallelogram, fillcolor=lightsteelblue, style=filled]
    output [label="Final Output\\n[4096×10000×8192]\\nGPU: 0-15", shape=parallelogram, fillcolor=lightgreen, style=filled]
    
    // Connections
    input -> split
    split -> mba0
    split -> mba1
    split -> mba2
    split -> mba3
    
    // Micro-batch 0 flow
    mba0 -> ln0_0 -> mha0_0 -> mlp0_0 -> ln1_0 -> mha1_0 -> mlp1_0 -> gather
    
    // Micro-batch 1 flow
    mba1 -> ln0_1 -> mha0_1 -> mlp0_1 -> ln1_1 -> mha1_1 -> mlp1_1 -> gather
    
    // Micro-batch 2 flow
    mba2 -> ln0_2 -> mha0_2 -> mlp0_2 -> ln1_2 -> mha1_2 -> mlp1_2 -> gather
    
    // Micro-batch 3 flow
    mba3 -> ln0_3 -> mha0_3 -> mlp0_3 -> ln1_3 -> mha1_3 -> mlp1_3 -> gather
    
    gather -> output
}
'''

# Write micro-batch parallelism DAG
with open(f"{dag_dir}/micro_batch_parallelism.dot", "w") as f:
    f.write(micro_batch_dag)

subprocess.run(["dot", "-Tsvg", f"{dag_dir}/micro_batch_parallelism.dot", "-o", f"{dag_dir}/micro_batch_parallelism.svg"])

print("Micro-batch parallelism DAG generated successfully")