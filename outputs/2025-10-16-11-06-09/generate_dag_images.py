#!/usr/bin/env python3
import os
import subprocess
import json

def generate_graphviz_images():
    """Generate SVG images from DOT files for the optimized Helix model."""
    
    dot_files = [
        "optimized_complete_helix_model.dot",
        "optimized_mha_layer_0_pipeline_parallel.dot",
        "optimized_mha_layer_1_pipeline_parallel.dot",
        "optimized_mlp_layer_0_grouped_tensor_parallel.dot",
        "optimized_mlp_layer_1_grouped_tensor_parallel.dot",
        "optimized_communication_patterns.dot"
    ]
    
    results = []
    
    for dot_file in dot_files:
        dot_path = f"./outputs/2025-10-16-11-06-09/{dot_file}"
        svg_path = dot_path.replace('.dot', '.svg')
        
        if os.path.exists(dot_path):
            cmd = ['dot', '-Tsvg', dot_path, '-o', svg_path]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                results.append({
                    "dot_file": dot_path,
                    "svg_file": svg_path,
                    "status": "success"
                })
                print(f"Generated: {svg_path}")
            except subprocess.CalledProcessError as e:
                results.append({
                    "dot_file": dot_path,
                    "svg_file": svg_path,
                    "status": "error",
                    "error": str(e)
                })
                print(f"Error generating {svg_path}: {e}")
        else:
            results.append({
                "dot_file": dot_path,
                "svg_file": svg_path,
                "status": "missing"
            })
            print(f"Missing: {dot_path}")
    
    return results

if __name__ == "__main__":
    results = generate_graphviz_images()
    
    # Save results to JSON
    with open('./outputs/2025-10-16-11-06-09/dag_generation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDAG generation completed. Results saved to dag_generation_results.json")