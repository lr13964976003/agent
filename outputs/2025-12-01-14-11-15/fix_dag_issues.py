#!/usr/bin/env python3

import re
import os

def fix_gpu_mapping_dag():
    """Fix the GPU mapping DAG by removing cycles"""
    input_file = "../outputs/2025-12-01-14-11-15/gpu_mapping_dag.dot"
    output_file = "../outputs/2025-12-01-14-11-15/gpu_mapping_dag_fixed.dot"
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Remove bidirectional tensor parallelism edges (keep only one direction)
    # Keep gpu_0 -> gpu_1, remove gpu_1 -> gpu_0, etc.
    lines = content.split('\n')
    fixed_lines = []
    
    # Track which TP edges we've seen
    tp_edges_seen = set()
    
    for line in lines:
        if '->' in line and 'TP All-Reduce' in line:
            # Parse the edge
            parts = line.split('->')
            if len(parts) == 2:
                src = parts[0].strip()
                dst_part = parts[1].strip()
                dst = dst_part.split('[')[0].strip()
                
                # Create a canonical edge representation (smaller node first)
                edge = tuple(sorted([src, dst]))
                
                if edge not in tp_edges_seen:
                    tp_edges_seen.add(edge)
                    fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Write the fixed content
    fixed_content = '\n'.join(fixed_lines)
    
    with open(output_file, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed GPU mapping DAG saved to {output_file}")
    return output_file

def fix_detailed_moe_layer_dag():
    """Fix the detailed MoE layer DAG by adding missing incoming connections"""
    input_file = "../outputs/2025-12-01-14-11-15/detailed_moe_layer.dot"
    output_file = "../outputs/2025-12-01-14-11-15/detailed_moe_layer_fixed.dot"
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Find all expert nodes that need connections
    expert_pattern = r'expert_(\\d+)(?!_ffn|_gelu)\\s*\\['
    experts = re.findall(expert_pattern, content)
    
    # Find existing connections to understand the pattern
    existing_connections = []
    lines = content.split('\n')
    
    # Find which experts are already connected
    connected_experts = set()
    a2a_connections = {}
    
    for line in lines:
        if 'a2a_group_' in line and '->' in line:
            # Extract the expert number from the destination
            parts = line.split('->')
            if len(parts) == 2:
                src = parts[0].strip()
                dst = parts[1].strip().split('[')[0].strip()
                if 'expert_' in dst:
                    expert_num = int(dst.replace('expert_', ''))
                    connected_experts.add(expert_num)
                    if src not in a2a_connections:
                        a2a_connections[src] = []
                    a2a_connections[src].append(dst)
    
    # Find which experts need connections
    all_experts = set(range(64))  # experts 0-63
    disconnected_experts = all_experts - connected_experts
    
    print(f"Connected experts: {len(connected_experts)}")
    print(f"Disconnected experts: {len(disconnected_experts)}")
    print(f"Disconnected expert numbers: {sorted(disconnected_experts)}")
    
    # Group experts by their GPU group (8 experts per group, 8 groups total)
    expert_groups = {}
    for expert_num in range(64):
        group_id = expert_num // 8
        if group_id not in expert_groups:
            expert_groups[group_id] = []
        expert_groups[group_id].append(expert_num)
    
    # Create new connections
    new_connections = []
    
    for group_id, experts_in_group in expert_groups.items():
        a2a_node = f"a2a_group_{group_id}"
        
        # Connect all experts in the group to their a2a node
        for expert_num in experts_in_group:
            if expert_num in disconnected_experts:
                expert_node = f"expert_{expert_num}"
                new_connections.append(f"\t{a2a_node} -> {expert_node}")
    
    # Add the new connections to the DAG
    # Find where to insert the new connections (before the closing brace)
    insert_point = None
    for i, line in enumerate(lines):
        if linestartswith('}'):
            insert_point = i
            break
    
    if insert_point:
        # Insert new connections before the closing brace
        new_lines = lines[:insert_point] + new_connections + lines[insert_point:]
        fixed_content = '\n'.join(new_lines)
    else:
        # If no closing brace found, append to end
        fixed_content = content + '\n'.join(new_connections) + '\n}'
    
    with open(output_file, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed detailed MoE layer DAG saved to {output_file}")
    return output_file

def main():
    print("Fixing DAG issues...")
    
    # Fix GPU mapping DAG
    gpu_mapping_fixed = fix_gpu_mapping_dag()
    
    # Fix detailed MoE layer DAG
    moe_layer_fixed = fix_detailed_moe_layer_dag()
    
    print("All fixes completed!")
    
    # Verify the fixes using the DAG extraction tool
    print("\nVerifying GPU mapping DAG...")
    from dag_extraction import extract_dag_info
    gpu_info = extract_dag_info(gpu_mapping_fixed)
    print(f"GPU mapping DAG has cycles: {gpu_info['has_cycle']}")
    
    print("\nVerifying detailed MoE layer DAG...")
    moe_info = extract_dag_info(moe_layer_fixed)
    nodes_with_only_outgoing = [node for node, info in moe_info['nodes'].items() if info['in_degree'] == 0 and info['out_degree'] > 0 and node != 'input']
    print(f"Detailed MoE layer DAG nodes with only outgoing edges: {len(nodes_with_only_outgoing)}")
    
    return gpu_mapping_fixed, moe_layer_fixed

if __name__ == "__main__":
    main()