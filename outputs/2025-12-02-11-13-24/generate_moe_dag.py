#!/usr/bin/env python3

import os

def generate_moe_dag():
    """Generate complete MoE model DAG with hybrid parallelism"""
    
    dot_content = """digraph MoE_Hybrid_Parallel {
    rankdir=TB;
    node [shape=rectangle, style=filled, fillcolor=lightblue];
    edge [fontsize=10];
    
    // Graph styling
    graph [bgcolor=white, fontname="Arial", fontsize=12];
    node [fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=9];
    
    // Define node shapes
    node [shape=ellipse, fillcolor=yellow]; // Communication nodes
    node [shape=parallelogram, fillcolor=lightgreen]; // Routing/aggregation nodes
    node [shape=rectangle, fillcolor=lightblue]; // Computation nodes
    
    // Input node
    Input [shape=rectangle, fillcolor=lightcoral, label="Input\\nInput:[batch_size=8,seq_len=256,token_dim=1024]\\nOutput:[batch_size=8,seq_len=256,token_dim=1024]"];
    
    // Pipeline Stage 0 (Layers 0-1) - GPUs 0-3
    subgraph cluster_stage0 {
        label="Pipeline Stage 0 (Layers 0-1) - GPUs 0-3";
        style=filled;
        fillcolor=lightgray;
        
        // Layer 0 - Attention (Tensor Parallel across GPUs 0-3)
        subgraph cluster_layer0_attention {
            label="Layer 0 - Multi-Head Attention (TP=4)";
            style=filled;
            fillcolor=lightblue;
            
            // Input split for tensor parallelism
            L0_Attention_Split [shape=parallelogram, fillcolor=lightgreen, label="Split Input\\nInput:[batch_size=8,seq_len=256,hidden=4096]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            
            // QKV projections (column parallel)
            L0_Q_Proj_GPU0 [shape=rectangle, fillcolor=lightblue, label="Q Projection GPU0\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_K_Proj_GPU0 [shape=rectangle, fillcolor=lightblue, label="K Projection GPU0\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_V_Proj_GPU0 [shape=rectangle, fillcolor=lightblue, label="V Projection GPU0\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            
            L0_Q_Proj_GPU1 [shape=rectangle, fillcolor=lightblue, label="Q Projection GPU1\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_K_Proj_GPU1 [shape=rectangle, fillcolor=lightblue, label="K Projection GPU1\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_V_Proj_GPU1 [shape=rectangle, fillcolor=lightblue, label="V Projection GPU1\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            
            L0_Q_Proj_GPU2 [shape=rectangle, fillcolor=lightblue, label="Q Projection GPU2\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_K_Proj_GPU2 [shape=rectangle, fillcolor=lightblue, label="K Projection GPU2\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_V_Proj_GPU2 [shape=rectangle, fillcolor=lightblue, label="V Projection GPU2\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            
            L0_Q_Proj_GPU3 [shape=rectangle, fillcolor=lightblue, label="Q Projection GPU3\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_K_Proj_GPU3 [shape=rectangle, fillcolor=lightblue, label="K Projection GPU3\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_V_Proj_GPU3 [shape=rectangle, fillcolor=lightblue, label="V Projection GPU3\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            
            // All-reduce for QKV
            L0_Q_AllReduce [shape=ellipse, fillcolor=yellow, label="All-Reduce Q\\nInput:[batch_size=8,seq_len=256,heads=2,d_k=128]x4\\nOutput:[batch_size=8,seq_len=256,heads=8,d_k=128]"];
            L0_K_AllReduce [shape=ellipse, fillcolor=yellow, label="All-Reduce K\\nInput:[batch_size=8,seq_len=256,heads=2,d_k=128]x4\\nOutput:[batch_size=8,seq_len=256,heads=8,d_k=128]"];
            L0_V_AllReduce [shape=ellipse, fillcolor=yellow, label="All-Reduce V\\nInput:[batch_size=8,seq_len=256,heads=2,d_k=128]x4\\nOutput:[batch_size=8,seq_len=256,heads=8,d_k=128]"];
            
            // Attention computation
            L0_Attention_GPU0 [shape=rectangle, fillcolor=lightblue, label="Attention Compute GPU0\\nInput:[batch_size=8,seq_len=256,heads=8,d_k=128]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_Attention_GPU1 [shape=rectangle, fillcolor=lightblue, label="Attention Compute GPU1\\nInput:[batch_size=8,seq_len=256,heads=8,d_k=128]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_Attention_GPU2 [shape=rectangle, fillcolor=lightblue, label="Attention Compute GPU2\\nInput:[batch_size=8,seq_len=256,heads=8,d_k=128]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            L0_Attention_GPU3 [shape=rectangle, fillcolor=lightblue, label="Attention Compute GPU3\\nInput:[batch_size=8,seq_len=256,heads=8,d_k=128]\\nOutput:[batch_size=8,seq_len=256,heads=2,d_k=128]"];
            
            // Output projection (row parallel)
            L0_O_Proj_GPU0 [shape=rectangle, fillcolor=lightblue, label="O Projection GPU0\\nInput:[batch_size=8,seq_len=256,heads=2,d_k=128]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            L0_O_Proj_GPU1 [shape=rectangle, fillcolor=lightblue, label="O Projection GPU1\\nInput:[batch_size=8,seq_len=256,heads=2,d_k=128]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            L0_O_Proj_GPU2 [shape=rectangle, fillcolor=lightblue, label="O Projection GPU2\\nInput:[batch_size=8,seq_len=256,heads=2,d_k=128]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            L0_O_Proj_GPU3 [shape=rectangle, fillcolor=lightblue, label="O Projection GPU3\\nInput:[batch_size=8,seq_len=256,heads=2,d_k=128]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            
            // All-reduce for output
            L0_O_AllReduce [shape=ellipse, fillcolor=yellow, label="All-Reduce O\\nInput:[batch_size=8,seq_len=256,hidden=1024]x4\\nOutput:[batch_size=8,seq_len=256,hidden=4096]"];
            
            // Residual connection
            L0_Residual_Add [shape=parallelogram, fillcolor=lightgreen, label="Residual Add\\nInput:[batch_size=8,seq_len=256,hidden=4096],[batch_size=8,seq_len=256,hidden=4096]\\nOutput:[batch_size=8,seq_len=256,hidden=4096]"];
            
            // Layer Norm
            L0_LayerNorm [shape=rectangle, fillcolor=lightblue, label="Layer Norm\\nInput:[batch_size=8,seq_len=256,hidden=4096]\\nOutput:[batch_size=8,seq_len=256,hidden=4096]"];
        }
        
        // Layer 0 - MoE (Expert Parallel across GPUs 0-3)
        subgraph cluster_layer0_moe {
            label="Layer 0 - MoE (EP=4)";
            style=filled;
            fillcolor=lightcyan;
            
            // Gate computation (on GPU0)
            L0_Gate [shape=rectangle, fillcolor=lightblue, label="Gate GPU0\\nInput:[batch_size=8,seq_len=256,hidden=4096]\\nOutput:[batch_size=8,seq_len=256,experts=4]"];
            
            // Expert routing (dashed lines indicate selection)
            L0_Router [shape=parallelogram, fillcolor=lightgreen, label="Router\\nInput:[batch_size=8,seq_len=256,experts=4]\\nOutput:Routing decisions"];
            
            // Expert computations (one per GPU)
            L0_Expert0_GPU0 [shape=rectangle, fillcolor=lightblue, label="Expert 0 GPU0\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            L0_Expert1_GPU1 [shape=rectangle, fillcolor=lightblue, label="Expert 1 GPU1\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            L0_Expert2_GPU2 [shape=rectangle, fillcolor=lightblue, label="Expert 2 GPU2\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            L0_Expert3_GPU3 [shape=rectangle, fillcolor=lightblue, label="Expert 3 GPU3\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            
            // Expert FFN details (column-row parallel)
            L0_Expert0_FFN1 [shape=rectangle, fillcolor=lightblue, label="Expert0 FFN1 GPU0\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,ffn=2048]"];
            L0_Expert0_GELU [shape=rectangle, fillcolor=lightblue, label="Expert0 GELU GPU0\\nInput:[batch_size=8,seq_len=256,ffn=2048]\\nOutput:[batch_size=8,seq_len=256,ffn=2048]"];
            L0_Expert0_FFN2 [shape=rectangle, fillcolor=lightblue, label="Expert0 FFN2 GPU0\\nInput:[batch_size=8,seq_len=256,ffn=2048]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            
            L0_Expert1_FFN1 [shape=rectangle, fillcolor=lightblue, label="Expert1 FFN1 GPU1\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,ffn=2048]"];
            L0_Expert1_GELU [shape=rectangle, fillcolor=lightblue, label="Expert1 GELU GPU1\\nInput:[batch_size=8,seq_len=256,ffn=2048]\\nOutput:[batch_size=8,seq_len=256,ffn=2048]"];
            L0_Expert1_FFN2 [shape=rectangle, fillcolor=lightblue, label="Expert1 FFN2 GPU1\\nInput:[batch_size=8,seq_len=256,ffn=2048]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            
            L0_Expert2_FFN1 [shape=rectangle, fillcolor=lightblue, label="Expert2 FFN1 GPU2\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,ffn=2048]"];
            L0_Expert2_GELU [shape=rectangle, fillcolor=lightblue, label="Expert2 GELU GPU2\\nInput:[batch_size=8,seq_len=256,ffn=2048]\\nOutput:[batch_size=8,seq_len=256,ffn=2048]"];
            L0_Expert2_FFN2 [shape=rectangle, fillcolor=lightblue, label="Expert2 FFN2 GPU2\\nInput:[batch_size=8,seq_len=256,ffn=2048]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            
            L0_Expert3_FFN1 [shape=rectangle, fillcolor=lightblue, label="Expert3 FFN1 GPU3\\nInput:[batch_size=8,seq_len=256,hidden=1024]\\nOutput:[batch_size=8,seq_len=256,ffn=2048]"];
            L0_Expert3_GELU [shape=rectangle, fillcolor=lightblue, label="Expert3 GELU GPU3\\nInput:[batch_size=8,seq_len=256,ffn=2048]\\nOutput:[batch_size=8,seq_len=256,ffn=2048]"];
            L0_Expert3_FFN2 [shape=rectangle, fillcolor=lightblue, label="Expert3 FFN2 GPU3\\nInput:[batch_size=8,seq_len=256,ffn=2048]\\nOutput:[batch_size=8,seq_len=256,hidden=1024]"];
            
            // Expert aggregation
            L0_Expert_Aggregate [shape=parallelogram, fillcolor=lightgreen, label="Aggregate Experts\\nInput:[batch_size=8,seq_len=256,hidden=1024]x4\\nOutput:[batch_size=8,seq_len=256,hidden=4096]"];
            
            // Residual connection
            L0_MoE_Residual_Add [shape=parallelogram, fillcolor=lightgreen, label="MoE Residual Add\\nInput:[batch_size=8,seq_len=256,hidden=4096],[batch_size=8,seq_len=256,hidden=4096]\\nOutput:[batch_size=8,seq_len=256,hidden=4096]"];
            
            // Layer Norm
            L0_MoE_LayerNorm [shape=rectangle, fillcolor=lightblue, label="MoE Layer Norm\\nInput:[batch_size=8,seq_len=256,hidden=4096]\\nOutput:[batch_size=8,seq_len=256,hidden=4096]"];
        }
        
        // Communication between stage 0 and stage 1
        Stage0_to_Stage1 [shape=ellipse, fillcolor=yellow, label="Pipeline Comm\\nStage0→Stage1\\nInput:[batch_size=8,seq_len=256,hidden=4096]\\nOutput:[batch_size=8,seq_len=256,hidden=4096]"];
    }
    
    // Final output
    Output [shape=rectangle, fillcolor=lightcoral, label="Output\\nInput:[batch_size=8,seq_len=256,hidden=4096]\\nOutput:[batch_size=8,seq_len=256,hidden=4096]"];
    
    // Edges for Layer 0 Attention
    Input -> L0_Attention_Split;
    L0_Attention_Split -> L0_Q_Proj_GPU0;
    L0_Attention_Split -> L0_K_Proj_GPU0;
    L0_Attention_Split -> L0_V_Proj_GPU0;
    L0_Attention_Split -> L0_Q_Proj_GPU1;
    L0_Attention_Split -> L0_K_Proj_GPU1;
    L0_Attention_Split -> L0_V_Proj_GPU1;
    L0_Attention_Split -> L0_Q_Proj_GPU2;
    L0_Attention_Split -> L0_K_Proj_GPU2;
    L0_Attention_Split -> L0_V_Proj_GPU2;
    L0_Attention_Split -> L0_Q_Proj_GPU3;
    L0_Attention_Split -> L0_K_Proj_GPU3;
    L0_Attention_Split -> L0_V_Proj_GPU3;
    
    L0_Q_Proj_GPU0 -> L0_Q_AllReduce;
    L0_Q_Proj_GPU1 -> L0_Q_AllReduce;
    L0_Q_Proj_GPU2 -> L0_Q_AllReduce;
    L0_Q_Proj_GPU3 -> L0_Q_AllReduce;
    
    L0_K_Proj_GPU0 -> L0_K_AllReduce;
    L0_K_Proj_GPU1 -> L0_K_AllReduce;
    L0_K_Proj_GPU2 -> L0_K_AllReduce;
    L0_K_Proj_GPU3 -> L0_K_AllReduce;
    
    L0_V_Proj_GPU0 -> L0_V_AllReduce;
    L0_V_Proj_GPU1 -> L0_V_AllReduce;
    L0_V_Proj_GPU2 -> L0_V_AllReduce;
    L0_V_Proj_GPU3 -> L0_V_AllReduce;
    
    L0_Q_AllReduce -> L0_Attention_GPU0;
    L0_Q_AllReduce -> L0_Attention_GPU1;
    L0_Q_AllReduce -> L0_Attention_GPU2;
    L0_Q_AllReduce -> L0_Attention_GPU3;
    L0_K_AllReduce -> L0_Attention_GPU0;
    L0_K_AllReduce -> L0_Attention_GPU1;
    L0_K_AllReduce -> L0_Attention_GPU2;
    L0_K_AllReduce -> L0_Attention_GPU3;
    L0_V_AllReduce -> L0_Attention_GPU0;
    L0_V_AllReduce -> L0_Attention_GPU1;
    L0_V_AllReduce -> L0_Attention_GPU2;
    L0_V_AllReduce -> L0_Attention_GPU3;
    
    L0_Attention_GPU0 -> L0_O_Proj_GPU0;
    L0_Attention_GPU1 -> L0_O_Proj_GPU1;
    L0_Attention_GPU2 -> L0_O_Proj_GPU2;
    L0_Attention_GPU3 -> L0_O_Proj_GPU3;
    
    L0_O_Proj_GPU0 -> L0_O_AllReduce;
    L0_O_Proj_GPU1 -> L0_O_AllReduce;
    L0_O_Proj_GPU2 -> L0_O_AllReduce;
    L0_O_Proj_GPU3 -> L0_O_AllReduce;
    
    L0_O_AllReduce -> L0_Residual_Add;
    Input -> L0_Residual_Add; // Residual connection from input
    L0_Residual_Add -> L0_LayerNorm;
    
    // Edges for Layer 0 MoE
    L0_LayerNorm -> L0_Gate;
    L0_Gate -> L0_Router;
    L0_Router -> L0_Expert0_GPU0 [style=dashed];
    L0_Router -> L0_Expert1_GPU1 [style=dashed];
    L0_Router -> L0_Expert2_GPU2 [style=dashed];
    L0_Router -> L0_Expert3_GPU3 [style=dashed];
    
    L0_LayerNorm -> L0_Expert0_GPU0;
    L0_LayerNorm -> L0_Expert1_GPU1;
    L0_LayerNorm -> L0_Expert2_GPU2;
    L0_LayerNorm -> L0_Expert3_GPU3;
    
    L0_Expert0_GPU0 -> L0_Expert0_FFN1;
    L0_Expert0_FFN1 -> L0_Expert0_GELU;
    L0_Expert0_GELU -> L0_Expert0_FFN2;
    
    L0_Expert1_GPU1 -> L0_Expert1_FFN1;
    L0_Expert1_FFN1 -> L0_Expert1_GELU;
    L0_Expert1_GELU -> L0_Expert1_FFN2;
    
    L0_Expert2_GPU2 -> L0_Expert2_FFN1;
    L0_Expert2_FFN1 -> L0_Expert2_GELU;
    L0_Expert2_GELU -> L0_Expert2_FFN2;
    
    L0_Expert3_GPU3 -> L0_Expert3_FFN1;
    L0_Expert3_FFN1 -> L0_Expert3_GELU;
    L0_Expert3_GELU -> L0_Expert3_FFN2;
    
    L0_Expert0_FFN2 -> L0_Expert_Aggregate;
    L0_Expert1_FFN2 -> L0_Expert_Aggregate;
    L0_Expert2_FFN2 -> L0_Expert_Aggregate;
    L0_Expert3_FFN2 -> L0_Expert_Aggregate;
    
    L0_Expert_Aggregate -> L0_MoE_Residual_Add;
    L0_LayerNorm -> L0_MoE_Residual_Add; // Residual connection
    L0_MoE_Residual_Add -> L0_MoE_LayerNorm;
    L0_MoE_LayerNorm -> Stage0_to_Stage1;
    
    // Stage 0 to Stage 1 connection
    Stage0_to_Stage1 -> Output;
    
}"""
    
    return dot_content

def main():
    # Generate the DOT content
    dot_content = generate_moe_dag()
    
    # Write to DOT file
    dot_file = "../outputs/2025-12-02-11-13-24/moe_hybrid_parallel_dag.dot"
    with open(dot_file, 'w') as f:
        f.write(dot_content)
    
    print(f"DOT file written to: {dot_file}")
    
    # Generate SVG image using dot command
    svg_file = "../outputs/2025-12-02-11-13-24/moe_hybrid_parallel_dag.svg"
    
    try:
        import subprocess
        result = subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], 
                              capture_output=True, text=True, check=True)
        print(f"SVG image generated: {svg_file}")
        print(f"DOT file path: {dot_file}")
        print(f"SVG file path: {svg_file}")
        
        # Return the paths in JSON format
        import json
        paths = {
            "dot_file": dot_file,
            "svg_file": svg_file
        }
        print(json.dumps(paths, indent=2))
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating SVG: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        # Still return the DOT file even if SVG generation fails
        import json
        paths = {
            "dot_file": dot_file,
            "svg_file": None
        }
        print(json.dumps(paths, indent=2))
    except FileNotFoundError:
        print("dot command not found. Please install graphviz to generate SVG images.")
        # Still return the DOT file
        import json
        paths = {
            "dot_file": dot_file,
            "svg_file": None
        }
        print(json.dumps(paths, indent=2))

if __name__ == "__main__":
    main()