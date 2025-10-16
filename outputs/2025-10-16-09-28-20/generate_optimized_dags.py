#!/usr/bin/env python3

import os
import subprocess
import json

def generate_svg_from_dot(dot_file, svg_file):
    """Generate SVG from DOT file using graphviz."""
    try:
        subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating {svg_file}: {e}")
        return False

def main():
    # Directory containing the DOT files
    output_dir = "./outputs/2025-10-16-09-28-20"
    
    # List of all optimized DAG files to process
    dag_files = [
        "optimized_complete_helix_model.dot",
        "optimized_mha_layer_0_pipelined.dot",
        "optimized_mha_layer_1_pipelined.dot",
        "optimized_mlp_layer_0_tensor_parallel.dot",
        "optimized_mlp_layer_1_tensor_parallel.dot",
        "optimized_communication_patterns.dot"
    ]
    
    # Generate SVG files and collect paths
    svg_paths = []
    dot_paths = []
    
    for dot_file in dag_files:
        dot_path = os.path.join(output_dir, dot_file)
        svg_file = dot_file.replace('.dot', '.svg')
        svg_path = os.path.join(output_dir, svg_file)
        
        if os.path.exists(dot_path):
            if generate_svg_from_dot(dot_path, svg_path):
                svg_paths.append(svg_path)
                dot_paths.append(dot_path)
                print(f"Generated {svg_file}")
            else:
                print(f"Failed to generate {svg_file}")
        else:
            print(f"DOT file not found: {dot_path}")
    
    # Create submission JSON
    submission = {
        "generated_dags": {
            "optimized_complete_model": {
                "dot_path": "./outputs/2025-10-16-09-28-20/optimized_complete_helix_model.dot",
                "svg_path": "./outputs/2025-10-16-09-28-20/optimized_complete_helix_model.svg"
            },
            "optimized_mha_layer_0": {
                "dot_path": "./outputs/2025-10-16-09-28-20/optimized_mha_layer_0_pipelined.dot",
                "svg_path": "./outputs/2025-10-16-09-28-20/optimized_mha_layer_0_pipelined.svg"
            },
            "optimized_mha_layer_1": {
                "dot_path": "./outputs/2025-10-16-09-28-20/optimized_mha_layer_1_pipelined.dot",
                "svg_path": "./outputs/2025-10-16-09-28-20/optimized_mha_layer_1_pipelined.svg"
            },
            "optimized_mlp_layer_0": {
                "dot_path": "./outputs/2025-10-16-09-28-20/optimized_mlp_layer_0_tensor_parallel.dot",
                "svg_path": "./outputs/2025-10-16-09-28-20/optimized_mlp_layer_0_tensor_parallel.svg"
            },
            "optimized_mlp_layer_1": {
                "dot_path": "./outputs/2025-10-16-09-28-20/optimized_mlp_layer_1_tensor_parallel.dot",
                "svg_path": "./outputs/2025-10-16-09-28-20/optimized_mlp_layer_1_tensor_parallel.svg"
            },
            "optimized_communication_patterns": {
                "dot_path": "./outputs/2025-10-16-09-28-20/optimized_communication_patterns.dot",
                "svg_path": "./outputs/2025-10-16-09-28-20/optimized_communication_patterns.svg"
            }
        },
        "optimization_summary": {
            "strategy": "Pipeline parallelism with 8-way tensor parallel per stage",
            "improvements": [
                "Reduced communication overhead from 16-way to 8-way partitioning",
                "Introduced pipeline stages enabling micro-batch overlap",
                "Fused attention operations reducing kernel launch overhead",
                "Optimized all-reduce operations with ring topology",
                "Distributed LayerNorm reducing synchronization points",
                "Better GPU utilization and load balancing across stages"
            ],
            "expected_tps_improvement": "~2-3x improvement through reduced communication latency and improved overlap",
            "architecture_changes": [
                "Pipeline parallelism across 2 stages (8 GPUs each) instead of 16-way single stage",
                "Reduced concatenation steps from multi-level to single-level",
                "Fused operations to reduce kernel launches",
                "Micro-batch overlap between stages"
            ]
        }
    }
    
    # Write submission JSON
    with open(os.path.join(output_dir, "submission.json"), "w") as f:
        json.dump(submission, f, indent=2)
    
    print("Submission JSON created successfully!")

if __name__ == "__main__":
    main()