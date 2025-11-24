#!/usr/bin/env python3
"""
Ring Attention + Sequence Parallel DAG Generator
16-way sequence parallelism across 16 GPUs
"""

import os

def generate_ring_attention_dag():
    """Generate detailed Graphviz DOT code for Ring Attention + Sequence Parallel"""
    
    dot_content = '''digraph ring_attention_sequence_parallel {
    rankdir=TB;
    bgcolor="#f8f9fa";
    
    // Graph attributes
    node [shape=rectangle, style="rounded,filled", fontname="Helvetica"];
    edge [fontname="Helvetica", fontsize=10];
    
    // Input nodes
    Input [label="Input\nSequence\nB=128, L=100K, D=4096", shape=ellipse, fillcolor="#e3f2fd"];
    
    // Sequence split across 16 devices
    subgraph cluster_sequence_split {
        label="Sequence Parallel Split (16 GPUs)";
        style="rounded,dashed";
        fillcolor="#fff3e0";
        
        Split0 [label="Split\nDevice 0: [0:6250]\nB=128, L=6250, D=4096", shape=parallelogram, fillcolor="#ffecb3"];
        Split1 [label="Split\nDevice 1: [6250:12500]\nB=128, L=6250, D=4096", shape=parallelogram, fillcolor="#ffecb3"];
        Split15 [label="Split\n...\nDevice 15: [93750:100000]\nB=128, L=6250, D=4096", shape=parallelogram, fillcolor="#ffecb3"];
    }
    
    // Representative layer (layer 0) on Device 0
    subgraph cluster_device0_layer0 {
        label="Layer 0 - Device 0 (Ring Position 0)";
        style="rounded,dashed";
        fillcolor="#e8f5e9";
        
        // QKV Projection
        D0_L0_QKV_proj [label="QKV Projection\nInput: [128, 6250, 4096]\nOutput: [128, 6250, 4096]\nWeights: [512, 4096]", fillcolor="#c8e6c9"];
        
        // Ring Attention components
        D0_L0_KV_buffer [label="KV Buffer\nSize: [128, 6250, 4096]\nLocal K,V", fillcolor="#fff9c4"];
        D0_L0_Q [label="Q Tensor\nSize: [128, 6250, 4096]\nLocal Q", fillcolor="#fff9c4"];
        
        // Ring communication nodes
        D0_L0_recv_prev [label="Receive\nFrom Device 15\nKV: [128, 6250, 4096]", shape=ellipse, fillcolor="#ff8a65"];
        D0_L0_send_next [label="Send\nTo Device 1\nKV: [128, 6250, 4096]", shape=ellipse, fillcolor="#ff8a65"];
        
        // Attention computation for each stage
        D0_L0_attention_stage0 [label="Attention Stage 0\nQ×Local KV\nInput: Q[128,6250,4096], KV[128,6250,4096]\nOutput: [128,6250,4096]", fillcolor="#c8e6c9"];
        D0_L0_attention_stage1 [label="Attention Stage 1\nQ×Recv KV\nInput: Q[128,6250,4096], KV[128,6250,4096]\nOutput: [128,6250,4096]", fillcolor="#c8e6c9"];
        D0_L0_attention_stage15 [label="Attention Stage 15\nQ×Device 1 KV\nInput: Q[128,6250,4096], KV[128,6250,4096]\nOutput: [128,6250,4096]", fillcolor="#c8e6c9"];
        
        D0_L0_accumulate [label="Accumulate\nAttention Outputs\nSum 16 partial results", shape=diamond, fillcolor="#ffccbc"];
        D0_L0_output_proj [label="Output Projection\nInput: [128, 6250, 4096]\nOutput: [128, 6250, 4096]", fillcolor="#c8e6c9"];
        D0_L0_residual1 [label="Add\nResidual", shape=diamond, fillcolor="#ffccbc"];
        
        // MLP components
        D0_L0_MLP_gate [label="MLP Gate\nInput: [128, 6250, 4096]\nOutput: [128, 6250, 16384]", fillcolor="#c8e6c9"];
        D0_L0_MLP_up [label="MLP Up\nInput: [128, 6250, 4096]\nOutput: [128, 6250, 16384]", fillcolor="#c8e6c9"];
        D0_L0_MLP_down [label="MLP Down\nInput: [128, 6250, 16384]\nOutput: [128, 6250, 4096]", fillcolor="#c8e6c9"];
        D0_L0_residual2 [label="Add\nResidual", shape=diamond, fillcolor="#ffccbc"];
    }
    
    // Ring communication flow
    subgraph cluster_ring_communication {
        label="Ring Communication Pattern";
        style="rounded,dashed";
        fillcolor="#fce4ec";
        
        Ring0 [label="Ring Step 0\nLocal compute", shape=ellipse, fillcolor="#f8bbd0"];
        Ring1 [label="Ring Step 1\nSend/Recv", shape=ellipse, fillcolor="#f8bbd0"];
        Ring15 [label="Ring Step 15\nFinal compute", shape=ellipse, fillcolor="#f8bbd0"];
        
        // Ring flow indicators
        Ring0 -> Ring1 [label="KV transfer"];
        Ring1 -> Ring15 [label="..."];
    }
    
    // Other devices (abbreviated)
    subgraph cluster_other_devices {
        label="Other Devices (1-15)";
        style="rounded,dashed";
        fillcolor="#f3e5f5";
        
        Device1 [label="Device 1\nSame as Device 0\nRing Position 1", shape=note, fillcolor="#f5f5f5"];
        Device15 [label="Device 15\nSame as Device 0\nRing Position 15", shape=note, fillcolor="#f5f5f5"];
    }
    
    // All layers (abbreviated)
    subgraph cluster_all_layers {
        label="All 16 Layers";
        style="rounded,dashed";
        fillcolor="#e0f2f1";
        
        NoteLayers [label="Layers 1-15\n(Same structure as Layer 0)\nEach device processes all 16 layers", shape=note, fillcolor="#f5f5f5"];
    }
    
    // Output aggregation
    subgraph cluster_output {
        label="Output Aggregation";
        style="rounded,dashed";
        fillcolor="#fff8e1";
        
        MergeAll [label="Merge\n16 Device Outputs\nSequence: [0:100000]", shape=parallelogram, fillcolor="#ffecb3"];
        Output [label="Output\nSequence\nB=128, L=100K, D=4096", shape=ellipse, fillcolor="#e3f2fd"];
    }
    
    // Detailed connections for Device 0
    Input -> Split0;
    Split0 -> D0_L0_QKV_proj;
    D0_L0_QKV_proj -> D0_L0_KV_buffer;
    D0_L0_QKV_proj -> D0_L0_Q;
    
    D0_L0_KV_buffer -> D0_L0_attention_stage0;
    D0_L0_Q -> D0_L0_attention_stage0;
    D0_L0_attention_stage0 -> D0_L0_accumulate;
    
    D0_L0_KV_buffer -> D0_L0_send_next;
    D0_L0_recv_prev -> D0_L0_attention_stage15;
    D0_L0_Q -> D0_L0_attention_stage15;
    D0_L0_attention_stage15 -> D0_L0_accumulate;
    
    D0_L0_accumulate -> D0_L0_output_proj;
    D0_L0_output_proj -> D0_L0_residual1;
    D0_L0_residual1 -> D0_L0_MLP_gate;
    D0_L0_MLP_gate -> D0_L0_MLP_up;
    D0_L0_MLP_up -> D0_L0_MLP_down;
    D0_L0_MLP_down -> D0_L0_residual2;
    D0_L0_residual2 -> NoteLayers;
    
    // Ring communication connections
    D0_L0_send_next -> Device1 [style=dashed, label="KV"];
    Device15 -> D0_L0_recv_prev [style=dashed, label="KV"];
    
    // Final output
    NoteLayers -> MergeAll;
    MergeAll -> Output;
    
    // Residual connections
    Split0 -> D0_L0_residual1 [style=dashed, label="Residual"];
    D0_L0_residual1 -> D0_L0_residual2 [style=dashed, label="Residual"];
    
    // Ring flow visualization
    D0_L0_KV_buffer -> Ring0;
    Ring0 -> D0_L0_send_next;
    D0_L0_recv_prev -> Ring15;
}
'''
    
    return dot_content

def generate_detailed_ring_attention_dag():
    """Generate a more detailed DAG showing all 16 devices and ring stages"""
    
    dot_content = '''digraph detailed_ring_attention {
    rankdir=LR;
    bgcolor="#f8f9fa";
    
    // Graph attributes
    node [shape=rectangle, style="rounded,filled", fontname="Helvetica"];
    edge [fontname="Helvetica", fontsize=9];
    
    // Input
    Input [label="Input\nB=128, L=100K, D=4096", shape=ellipse, fillcolor="#e3f2fd"];
    
    // Layer 0 across all 16 devices
    subgraph cluster_layer0_ring {
        label="Layer 0 Ring Attention - 16 Devices";
        style="rounded,dashed";
        fillcolor="#e8f5e9";
        
        // Device 0
        subgraph cluster_d0 {
            label="Device 0 (Seq 0-6250)";
            style="rounded";
            fillcolor="#fff3e0";
            
            D0_QKV [label="QKV Proj\n[128,6250,4096]", fillcolor="#c8e6c9"];
            D0_Q [label="Q\n[128,6250,4096]", fillcolor="#fff9c4"];
            D0_KV0 [label="KV-0\nLocal", fillcolor="#fff9c4"];
            D0_Compute0 [label="Attn-0\nQ×KV0", fillcolor="#c8e6c9"];
            D0_KV1 [label="KV-1\nFrom D15", fillcolor="#ffecb3"];
            D0_Compute1 [label="Attn-1\nQ×KV1", fillcolor="#c8e6c9"];
            D0_KV15 [label="KV-15\nFrom D1", fillcolor="#ffecb3"];
            D0_Compute15 [label="Attn-15\nQ×KV15", fillcolor="#c8e6c9"];
            D0_Accum [label="Sum\n16 partials", shape=diamond, fillcolor="#ffccbc"];
            D0_Out [label="Output Proj\n[128,6250,4096]", fillcolor="#c8e6c9"];
        }
        
        // Device 1
        subgraph cluster_d1 {
            label="Device 1 (Seq 6250-12500)";
            style="rounded";
            fillcolor="#fff3e0";
            
            D1_QKV [label="QKV Proj\n[128,6250,4096]", fillcolor="#c8e6c9"];
            D1_Q [label="Q\n[128,6250,4096]", fillcolor="#fff9c4"];
            // Similar compute stages...
            D1_Accum [label="Sum\n16 partials", shape=diamond, fillcolor="#ffccbc"];
            D1_Out [label="Output Proj\n[128,6250,4096]", fillcolor="#c8e6c9"];
        }
        
        // Device 15
        subgraph cluster_d15 {
            label="Device 15 (Seq 93750-100K)";
            style="rounded";
            fillcolor="#fff3e0";
            
            D15_QKV [label="QKV Proj\n[128,6250,4096]", fillcolor="#c8e6c9"];
            D15_Q [label="Q\n[128,6250,4096]", fillcolor="#fff9c4"];
            D15_Accum [label="Sum\n16 partials", shape=diamond, fillcolor="#ffccbc"];
            D15_Out [label="Output Proj\n[128,6250,4096]", fillcolor="#c8e6c9"];
        }
    }
    
    // Ring communication edges
    subgraph cluster_ring_comm {
        label="Ring Communication";
        style="rounded,dashed";
        fillcolor="#fce4ec";
        
        edge [color="#e91e63", style=dashed];
        // KV ring transfers
        D0_KV0 -> D1_Compute0 [constraint=false];
        D1_KV0 -> D2 [constraint=false];
        D15_KV0 -> D0_Compute15 [constraint=false];
    }
    
    // Input split
    Input -> D0_QKV;
    Input -> D1_QKV;
    Input -> D15_QKV;
    
    // Device 0 connections
    D0_QKV -> D0_Q;
    D0_QKV -> D0_KV0;
    D0_Q -> D0_Compute0;
    D0_KV0 -> D0_Compute0;
    D0_Compute0 -> D0_Accum;
    D0_Accum -> D0_Out;
    
    // Similar connections for all devices...
    
    // Output merge
    D0_Out -> Output;
    D1_Out -> Output;
    D15_Out -> Output;
    
    Output [label="Output\nB=128, L=100K, D=4096", shape=ellipse, fillcolor="#e3f2fd"];
}
'''
    
    return dot_content

if __name__ == "__main__":
    # Generate ring attention DAG
    dag_content = generate_ring_attention_dag()
    
    # Write detailed ring attention DOT file
    with open("../outputs/2025-11-24-16-10-27/ring_attention_sequence_parallel.dot", "w") as f:
        f.write(dag_content)
    
    # Generate SVG
    os.system("dot -Tsvg ../outputs/2025-11-24-16-10-27/ring_attention_sequence_parallel.dot -o ../outputs/2025-11-24-16-10-27/ring_attention_sequence_parallel.svg")
    
    # Generate detailed version
    detailed_content = generate_detailed_ring_attention_dag()
    with open("../outputs/2025-11-24-16-10-27/detailed_ring_attention.dot", "w") as f:
        f.write(detailed_content)
    
    os.system("dot -Tsvg ../outputs/2025-11-24-16-10-27/detailed_ring_attention.dot -o ../outputs/2025-11-24-16-10-27/detailed_ring_attention.svg")
    
    print("Generated ring attention DAG files:")
    print("- ../outputs/2025-11-24-16-10-27/ring_attention_sequence_parallel.dot")
    print("- ../outputs/2025-11-24-16-10-27/ring_attention_sequence_parallel.svg")
    print("- ../outputs/2025-11-24-16-10-27/detailed_ring_attention.dot")
    print("- ../outputs/2025-11-24-16-10-27/detailed_ring_attention.svg")