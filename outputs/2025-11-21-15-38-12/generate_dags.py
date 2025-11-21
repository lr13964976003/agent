import graphviz
import os

def render_dag(dot_file, output_format='svg'):
    """Render a DOT file to SVG format"""
    try:
        with open(dot_file, 'r') as f:
            dot_content = f.read()
        
        # Extract base name without extension
        base_name = os.path.splitext(os.path.basename(dot_file))[0]
        output_path = os.path.join(os.path.dirname(dot_file), base_name)
        
        # Create graphviz source
        source = graphviz.Source(dot_content)
        
        # Render to SVG
        source.render(output_path, format=output_format, cleanup=True)
        print(f"Successfully rendered {dot_file} to {output_path}.{output_format}")
        return f"{output_path}.{output_format}"
        
    except Exception as e:
        print(f"Error rendering {dot_file}: {str(e)}")
        return None

def validate_dag(dot_file):
    """Validate the DAG structure"""
    try:
        with open(dot_file, 'r') as f:
            content = f.read()
        
        # Check for cycles
        if 'cycle' in content.lower():
            print(f"Warning: {dot_file} may contain cycles")
            return False
        
        # Check for required nodes
        if 'Input' not in content or 'Output' not in content:
            print(f"Warning: {dot_file} missing Input/Output nodes")
            return False
            
        # Check for GPU specifications
        if 'GPU' not in content:
            print(f"Warning: {dot_file} missing GPU specifications")
            return False
            
        print(f"{dot_file} validation passed")
        return True
        
    except Exception as e:
        print(f"Error validating {dot_file}: {str(e)}")
        return False

# Generate SVG images
dot_files = [
    '../outputs/2025-11-21-15-38-12/baseline_dag.dot',
    '../outputs/2025-11-21-15-38-12/proposed_dag.dot'
]

results = {}
for dot_file in dot_files:
    if os.path.exists(dot_file):
        print(f"Processing {dot_file}...")
        
        # Validate DAG
        is_valid = validate_dag(dot_file)
        
        # Render to SVG
        svg_path = render_dag(dot_file)
        
        results[dot_file] = {
            'valid': is_valid,
            'svg_path': svg_path
        }
    else:
        print(f"File not found: {dot_file}")
        results[dot_file] = {'valid': False, 'svg_path': None}

# Print results
print("\n=== DAG Processing Results ===")
for file_path, result in results.items():
    print(f"{file_path}: {result}")

# Save summary
with open('../outputs/2025-11-21-15-38-12/dag_summary.txt', 'w') as f:
    f.write("=== DAG Generation Summary ===\n")
    for file_path, result in results.items():
        f.write(f"{file_path}: {result}\n")