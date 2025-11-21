import os
import subprocess

def generate_svg(dot_file_path, svg_file_path):
    """Generate SVG from DOT file using Graphviz."""
    try:
        cmd = ['dot', '-Tsvg', dot_file_path, '-o', svg_file_path]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully generated: {svg_file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating SVG: {e}")
        return False

def main():
    output_dir = "../outputs/2025-11-21-15-38-12"
    
    # Files to process
    dag_files = [
        ("baseline_tensor_pipeline_dag.dot", "baseline_tensor_pipeline_dag.svg"),
        ("proposed_layerwise_dag.dot", "proposed_layerwise_dag.svg")
    ]
    
    # Generate SVGs
    for dot_file, svg_file in dag_files:
        dot_path = os.path.join(output_dir, dot_file)
        svg_path = os.path.join(output_dir, svg_file)
        
        if os.path.exists(dot_path):
            generate_svg(dot_path, svg_path)
        else:
            print(f"DOT file not found: {dot_path}")

if __name__ == "__main__":
    main()