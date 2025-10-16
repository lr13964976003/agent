#!/usr/bin/env python3
import subprocess
import os

def generate_svg(dot_file, svg_file):
    """Generate SVG from DOT file using Graphviz"""
    try:
        result = subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully generated {svg_file}")
        else:
            print(f"Error generating {svg_file}: {result.stderr}")
    except Exception as e:
        print(f"Exception generating {svg_file}: {e}")

# List of DOT files to convert
dot_files = [
    "optimized_complete_helix_model.dot",
    "optimized_mha_layer_0_pipeline_parallel.dot",
    "optimized_mha_layer_1_pipeline_parallel.dot",
    "optimized_mlp_layer_0_grouped_tensor_parallel.dot",
    "optimized_mlp_layer_1_grouped_tensor_parallel.dot",
    "optimized_communication_patterns.dot"
]

output_dir = "./outputs/2025-10-16-11-06-09"

# Generate SVG for each DOT file
for dot_file in dot_files:
    dot_path = os.path.join(output_dir, dot_file)
    svg_path = os.path.join(output_dir, dot_file.replace('.dot', '.svg'))
    
    if os.path.exists(dot_path):
        print(f"Processing {dot_path}...")
        generate_svg(dot_path, svg_path)
    else:
        print(f"File not found: {dot_path}")

print("Image generation complete")