#!/usr/bin/env python3
"""
Script to fix all DAG files by adding explicit output nodes
"""

import os
import re

def add_output_node_to_dag(file_path):
    """Add an explicit output node to a DAG file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Skip if already has output node
    if 'Model Output' in content or 'output [label=' in content:
        return False
    
    lines = content.split('\n')
    
    # Find the last computation node
    last_node = None
    for line in reversed(lines):
        if '[' in line and 'label=' in line and '->' not in line:
            # Extract node name
            node_match = re.match(r'\s*(\w+)\s*\[', line)
            if node_match and node_match.group(1).strip() not in ['rankdir', 'node']:
                last_node = node_match.group(1).strip()
                break
    
    if last_node:
        # Determine appropriate output label based on file type
        if 'baseline' in file_path:
            output_label = "Model Output\nInput: [batch_size=1, seq_len=2048, vocab_size=51200]\nGPU: 0"
        elif 'attention_device' in file_path:
            device_match = re.search(r'device_(\d+)', file_path)
            device_num = device_match.group(1) if device_match else '0'
            output_label = f"Attention Output\nInput: [batch_size=1, seq_len=2048, hidden_dim=4096]\nGPU: {device_num}"
        elif 'mlp_device' in file_path:
            device_match = re.search(r'device_(\d+)', file_path)
            device_num = device_match.group(1) if device_match else '0'
            output_label = f"MLP Output\nInput: [batch_size=1, seq_len=2048, hidden_dim=4096]\nGPU: {device_num}"
        else:
            output_label = "Model Output"
        
        output_node = f'''
    output [label="{output_label}", shape=parallelogram, fillcolor=lightblue];
    {last_node} -> output;
}}'''
        
        # Remove the closing brace and add output
        modified_content = content.rstrip().rstrip('}') + output_node
        
        with open(file_path, 'w') as f:
            f.write(modified_content)
        return True
    
    return False

def fix_all_dag_files():
    """Fix all DAG files in the outputs directory"""
    output_dir = "../outputs/2025-10-29-14-21-46"
    
    # Find all .dot files
    dot_files = [f for f in os.listdir(output_dir) if f.endswith('.dot')]
    
    fixed_count = 0
    for filename in dot_files:
        file_path = os.path.join(output_dir, filename)
        try:
            if add_output_node_to_dag(file_path):
                fixed_count += 1
                print(f"Fixed: {filename}")
        except Exception as e:
            print(f"Error fixing {filename}: {e}")
    
    print(f"Total files fixed: {fixed_count} out of {len(dot_files)} total files")
    return fixed_count

if __name__ == "__main__":
    fix_all_dag_files()