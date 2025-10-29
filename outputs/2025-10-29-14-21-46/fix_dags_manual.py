#!/usr/bin/env python3
import os
import re

def fix_baseline_dag(file_path):
    """Fix baseline DAG by adding output node after lm_head"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    if 'output [label=' in content:
        return False
    
    # Replace the closing brace with output node and edge
    new_content = content.rstrip('}').rstrip() + '''

    output [label="Model Output\nInput: [batch_size=1, seq_len=2048, vocab_size=51200]\nGPU: 0", shape=parallelogram, fillcolor=lightblue];
    lm_head -> output;
}'''
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    return True

def fix_attention_device_dag(file_path):
    """Fix attention device DAG by adding output node"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    if 'output [label=' in content:
        return False
    
    # Extract device number
    device_match = re.search(r'device_(\d+)', file_path)
    device_num = device_match.group(1) if device_match else '0'
    
    # Replace the closing brace with output node and edge
    new_content = content.rstrip('}').rstrip() + f'''

    output [label="Attention Output\nInput: [batch_size=1, seq_len=2048, hidden_dim=4096]\nGPU: {device_num}", shape=parallelogram, fillcolor=lightblue];
    attention_output -> output;
}}'''
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    return True

def fix_mlp_device_dag(file_path):
    """Fix MLP device DAG by adding output node"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    if 'output [label=' in content:
        return False
    
    # Extract device number
    device_match = re.search(r'device_(\d+)', file_path)
    device_num = device_match.group(1) if device_match else '0'
    
    # Replace the closing brace with output node and edge
    new_content = content.rstrip('}').rstrip() + f'''

    output [label="MLP Output\nInput: [batch_size=1, seq_len=2048, hidden_dim=4096]\nGPU: {device_num}", shape=parallelogram, fillcolor=lightblue];
    mlp_output -> output;
}}'''
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    return True

def process_all_files():
    """Process all DAG files and add output nodes"""
    output_dir = "../outputs/2025-10-29-14-21-46"
    
    fixed_count = 0
    
    # Process baseline files
    baseline_files = [
        'megatron_8_3b_baseline.dot',
        'megatron_530b_baseline.dot',
        'megatron_1t_baseline.dot',
        'gopher_280b_baseline.dot',
        'palm_540b_baseline.dot',
        'gpt3_175b_baseline.dot'
    ]
    
    for filename in baseline_files:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            if fix_baseline_dag(file_path):
                fixed_count += 1
                print(f"Fixed baseline: {filename}")
    
    # Process attention device files
    attention_files = [
        ('megatron_8_3b_attention_device_', 8),
        ('megatron_530b_attention_device_', 8),
        ('megatron_1t_attention_device_', 8),
        ('gopher_280b_attention_device_', 8),
        ('gpt3_175b_attention_device_', 8),
        ('palm_540b_attention_device_', 12)
    ]
    
    for prefix, count in attention_files:
        for i in range(count):
            filename = f"{prefix}{i}.dot"
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path):
                # Need to modify the attention device files to have consistent end node
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Create a proper DAG with output
                lines = content.split('\n')
                modified_lines = []
                for line in lines:
                    if line.strip().endswith('}'):
                        continue
                    modified_lines.append(line)
                
                # Add output
                device_str = str(i)
                modified_lines.extend([
                    f'',
                    f'    output [label="Attention Output\\nInput: [batch_size=1, seq_len=2048, hidden_dim=4096]\\nGPU: {device_str}", shape=parallelogram, fillcolor=lightblue];',
                    f'    out_proj -> output;',
                    f'}}'
                ])
                
                with open(file_path, 'w') as f:
                    f.write('\n'.join(modified_lines))
                fixed_count += 1
                print(f"Fixed attention: {filename}")
    
    # Process MLP device files
    mlp_files = [
        ('megatron_8_3b_mlp_device_', 8),
        ('megatron_530b_mlp_device_', 8),
        ('megatron_1t_mlp_device_', 8),
        ('gopher_280b_mlp_device_', 8),
        ('gpt3_175b_mlp_device_', 8),
        ('palm_540b_mlp_device_', 12)
    ]
    
    for prefix, count in mlp_files:
        for i in range(count):
            filename = f"{prefix}{i}.dot"
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path):
                # Need to modify the MLP device files to have consistent end node
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Create a proper DAG with output
                lines = content.split('\n')
                modified_lines = []
                for line in lines:
                    if line.strip().endswith('}'):
                        continue
                    modified_lines.append(line)
                
                # Add output
                device_str = str(i)
                modified_lines.extend([
                    f'',
                    f'    output [label="MLP Output\\nInput: [batch_size=1, seq_len=2048, hidden_dim=4096]\\nGPU: {device_str}", shape=parallelogram, fillcolor=lightblue];',
                    f'    down_proj -> output;',
                    f'}}'
                ])
                
                with open(file_path, 'w') as f:
                    f.write('\n'.join(modified_lines))
                fixed_count += 1
                print(f"Fixed MLP: {filename}")
    
    print(f"Total files fixed: {fixed_count}")

if __name__ == "__main__":
    process_all_files()