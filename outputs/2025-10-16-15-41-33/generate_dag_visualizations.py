#!/usr/bin/env python3
"""
Generate visual representations of the FA Pool DAGs
"""

import os
import subprocess
import graphviz

# Directory containing the DAGs
OUTPUT_DIR = "../outputs/2025-10-16-15-41-33"

def render_dot_file(dot_file, output_format='svg'):
    """Render a dot file to SVG format"""
    try:
        # Read the dot file
        with open(os.path.join(OUTPUT_DIR, dot_file), 'r') as f:
            dot_source = f.read()
        
        # Create a graphviz source object
        source = graphviz.Source(dot_source)
        
        # Get base filename without extension
        base_name = os.path.splitext(dot_file)[0]
        
        # Render to SVG
        output_path = os.path.join(OUTPUT_DIR, base_name)
        source.render(output_path, format=output_format, cleanup=False)
        
        print(f"Successfully rendered {dot_file} to {base_name}.{output_format}")
        return f"{output_path}.{output_format}"
        
    except Exception as e:
        print(f"Error rendering {dot_file}: {e}")
        return None

def main():
    """Main function to generate all DAG visualizations"""
    
    # List of dot files to render
    dot_files = [
        "baseline_static_parallelization_complete.dot",
        "fa_pool_dynamic_8k_tokens.dot",
        "fa_pool_no_attention_pool.dot"
    ]
    
    generated_files = []
    
    print("Generating DAG visualizations...")
    
    for dot_file in dot_files:
        if os.path.exists(os.path.join(OUTPUT_DIR, dot_file)):
            svg_path = render_dot_file(dot_file, 'svg')
            if svg_path:
                generated_files.append(svg_path)
        else:
            print(f"Warning: {dot_file} not found")
    
    print(f"\nGenerated {len(generated_files)} visualizations:")
    for f in generated_files:
        print(f"  - {f}")
    
    return generated_files

if __name__ == "__main__":
    main()