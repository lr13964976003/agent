import subprocess
import os

# Define the paths
dot_files = [
    "baseline_dag_corrected.dot",
    "proposed_dag_corrected.dot"
]

# Generate SVG images
for dot_file in dot_files:
    svg_file = dot_file.replace('.dot', '.svg')
    try:
        subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], check=True)
        print(f"Successfully generated {svg_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating {svg_file}: {e}")
    except FileNotFoundError:
        print("Graphviz 'dot' command not found. Please install graphviz.")
        break

# Verify DAGs have no cycles
import re

def check_cycles_in_dot(dot_path):
    """Check if the DOT file contains cycles by analyzing edge relationships"""
    with open(dot_path, 'r') as f:
        content = f.read()
    
    # Extract node names and edges
    node_pattern = r'\b(\w+)\s*\['
    edge_pattern = r'(\w+)\s*->\s*(\w+)'
    
    nodes = set(re.findall(node_pattern, content))
    edges = re.findall(edge_pattern, content)
    
    # Build adjacency list
    adj = {node: [] for node in nodes}
    for src, dst in edges:
        if src in adj and dst in adj:
            adj[src].append(dst)
    
    # Check for cycles using DFS
    visited = set()
    rec_stack = set()
    
    def has_cycle(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    for node in nodes:
        if node not in visited:
            if has_cycle(node):
                return False
    
    return True

# Verify both DAGs
for dot_file in dot_files:
    has_no_cycles = check_cycles_in_dot(dot_file)
    print(f"{dot_file}: {'DAG verified (no cycles)' if has_no_cycles else 'WARNING: Contains cycles'}")

# Generate summary JSON
import json
summary = {
    "baseline_dag": {
        "dot_path": "baseline_dag_corrected.dot",
        "svg_path": "baseline_dag_corrected.svg",
        "description": "4-layer dense model with TP=8, PP=2 on 16 GPUs",
        "layers": 4,
        "gpus": 16,
        "tensor_parallelism": 8,
        "pipeline_parallelism": 2
    },
    "proposed_dag": {
        "dot_path": "proposed_dag_corrected.dot",
        "svg_path": "proposed_dag_corrected.svg", 
        "description": "4-layer dense model with layer-wise distribution (4 GPUs per layer)",
        "layers": 4,
        "gpus": 16,
        "gpus_per_layer": 4
    }
}

with open('dag_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("DAG generation and verification completed!")