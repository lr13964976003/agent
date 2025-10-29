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
    
    # Check the type of DAG and add appropriate output
    if 'baseline' in file_path or 'pipeline' in file_path:
        # For full model DAGs
        lines = content.split('\n')
        new_lines = []
        
        # Find the last computation node
        last_node = None
        for line in reversed(lines):
            if '[' in line and 'label=' in line and '->' not in line:
                # Extract node name
                node_match = re.match(r'\s*(\w+)\s*\[', line)
                if node_match:
                    last_node = node_match.group(1).strip()
                    break
        
        if last_node:
            # Add output node and edge
            output_node = '''
    output [label="Model Output\nInput: [batch_size=1, seq_len=2048, vocab_size=51200]\nGPU: 0", shape=parallelogram, fillcolor=lightblue];
    %s -> output;
}''' % last_node
            
            # Remove the closing brace and add output
            modified_content = content.rstrip().rstrip('}') + output_node
            
            with open(file_path, 'w') as f:
                f.write(modified_content)
            return True
    
    elif 'attention_device' in file_path:
        # For attention device-specific DAGs
        lines = content.split('\n')
        new_lines = []
        
        # Find the last computation node
        last_node = None
        for line in reversed(lines):
            if '[' in line and 'label=' in line and '->' not in line:
                node_match = re.match(r'\s*(\w+)\s*\[', line)
                if node_match:
                    last_node = node_match.group(1).strip()
                    break
        
        if last_node:
            # Extract device number
            device_match = re.search(r'device_(\d+)', file_path)
            device_num = device_match.group(1) if device_match else '0'
            
            output_node = '''
    output [label="Attention Output\nInput: [batch_size=1, seq_len=2048, hidden_dim=4096]\nGPU: %s", shape=parallelogram, fillcolor=lightblue];
    %s -> output;
}''' % (device_num, last_node)
            
            modified_content = content.rstrip().rstrip('}') + output_node
            
            with open(file_path, 'w') as f:
                f.write(modified_content)
            return True
    
    elif 'mlp_device' in file_path:
        # For MLP device-specific DAGs
        lines = content.split('\n')
        new_lines = []
        
        # Find the last computation node
        last_node = None
        for line in reversed(lines):
            if '[' in line and 'label=' in line and '->' not in line:
                node_match = re.match(r'\s*(\w+)\s*\[', line)
                if node_match:
                    last_node = node_match.group(1).strip()
                    break
        
        if last_node:
            # Extract device number
            device_match = re.search(r'device_(\d+)', file_path)
            device_num = device_match.group(1) if device_match else '0'
            
            output_node = '''
    output [label="MLP Output\nInput: [batch_size=1, seq_len=2048, hidden_dim=4096]\nGPU: %s", shape=parallelogram, fillcolor=lightblue];
    %s -> output;
}''' % (device_num, last_node)
            
            modified_content = content.rstrip().rstrip('}') + output_node
            
            with open(file_path, 'w') as f:
                f.write(modified_content)
            return True
    
    return False

def fix_all_dag_files():
    """Fix all DAG files in the outputs directory"""
    output_dir = "../outputs/2025-10-29-14-21-46"
    
    # List of all DAG files to fix
    dag_files = []
    
    # Baseline DAGs
    for model in ['megatron_8_3b', 'megatron_530b', 'megatron_1t', 'gopher_280b', 'palm_540b', 'gpt3_175b']:
        dag_files.append(f"{output_dir}/{model}_baseline.dot")
        dag_files.append(f"{output_dir}/{model}_pipeline.dot")
    
    # Attention device DAGs
    for model in ['megatron_8_3b', 'megatron_530b', 'megatron_1t', 'gopher_280b', 'gpt3_175b']:
        for device in range(8):
            dag_files.append(f"{output_dir}/{model}_attention_device_{device}.dot")
    
    for device in range(12):  # PaLM has 12 devices
        dag_files.append(f"{output_dir}/palm_540b_attention_device_{device}.dot")
    
    # MLP device DAGs
    for model in ['megatron_8_3b', 'megatron_530b', 'megatron_1t', 'gopher_280b', 'gpt3_175b']:
        for device in range(8):
            dag_files.append(f"{output_dir}/{model}_mlp_device_{device}.dot")
    
    for device in range(12):  # PaLM has 12 devices
        dag_files.append(f"{output_dir}/palm_540b_mlp_device_{device}.dot")
    
    fixed_count = 0
    for file_path in dag_files:
        if os.path.exists(file_path):
            try:
                if add_output_node_to_dag(file_path):
                    fixed_count += 1
                    print(f"Fixed: {file_path}")
            except Exception as e:
                print(f"Error fixing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    print(f"Total files fixed: {fixed_count}")
    return fixed_count

if __name__ == "__main__":
    fix_all_dag_files()