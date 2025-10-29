#!/usr/bin/env python3
import os
import glob

def add_output_to_file(file_path, output_type="Model"):
    """Add output node to a DOT file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Skip if already has output
    if 'output [label=' in content:
        return False
    
    # Find the last node name
    lines = content.strip().split('\n')
    last_node = None
    
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('//') or line == '' or line.startswith('}'):
            continue
        if '->' in line or '[' not in line:
            continue
        
        # Extract node name
        parts = line.strip().split('[')
        if len(parts) >= 2:
            node_name = parts[0].strip()
            if node_name and not node_name.startswith('rankdir') and not node_name.startswith('node'):
                last_node = node_name
                break
    
    if last_node:
        # Determine device for device-specific files
        device_num = "0"
        if '_device_' in file_path:
            device_match = file_path.split('_device_')[-1].split('.')[0]
            device_num = device_match
        
        # Create appropriate output
        if 'baseline' in file_path:
            output_line = f'    output [label="Model Output\\nInput: [batch_size=1, seq_len=2048, vocab_size=51200]\\nGPU: 0", shape=parallelogram, fillcolor=lightblue];'
        elif 'attention_device' in file_path:
            output_line = f'    output [label="Attention Output\\nInput: [batch_size=1, seq_len=2048, hidden_dim=4096]\\nGPU: {device_num}", shape=parallelogram, fillcolor=lightblue];'
        elif 'mlp_device' in file_path:
            output_line = f'    output [label="MLP Output\\nInput: [batch_size=1, seq_len=2048, hidden_dim=4096]\\nGPU: {device_num}", shape=parallelogram, fillcolor=lightblue];'
        else:
            output_line = f'    output [label="{output_type} Output", shape=parallelogram, fillcolor=lightblue];'
        
        # Remove closing brace and add output
        if content.endswith('}'):
            content = content.rstrip('}').rstrip()
        
        new_content = content + f'\n{output_line}\n    {last_node} -> output;\n}}'
        
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    
    return False

# Process all .dot files
output_dir = "../outputs/2025-10-29-14-21-46"
dot_files = glob.glob(f"{output_dir}/*.dot")

fixed_count = 0
for file_path in dot_files:
    try:
        if add_output_to_file(file_path):
            fixed_count += 1
            print(f"Added output to: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error with {file_path}: {e}")

print(f"Fixed {fixed_count} out of {len(dot_files)} files")