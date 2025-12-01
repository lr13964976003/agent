#!/usr/bin/env python3

import json

def check_dag_structure():
    """Check the DAG structure for expert parallelism"""
    
    # Read the expert parallelism DAG
    with open('./expert_parallelism_dag.json', 'r') as f:
        dag_data = json.load(f)
    
    print("=== DAG Structure Analysis ===")
    print(f"Nodes: {len(dag_data['nodes'])}")
    print(f"Edges: {len(dag_data['edges'])}")
    
    # Analyze nodes
    compute_nodes = [n for n in dag_data['nodes'] if n['type'] == 'compute']
    comm_nodes = [n for n in dag_data['nodes'] if n['type'] == 'comm']
    
    print(f"\nCompute nodes: {len(compute_nodes)}")
    print(f"Communication nodes: {len(comm_nodes)}")
    
    # Check GPU assignments
    gpu_ids = []
    for node in dag_data['nodes']:
        if 'gpu' in node and node['gpu'] is not None:
            gpu_ids.append(node['gpu'])
    
    print(f"GPU IDs used: {sorted(set(gpu_ids))}")
    print(f"Total unique GPUs: {len(set(gpu_ids))}")
    
    # Check if this represents the EP64 strategy
    print(f"\n=== Strategy Compatibility Check ===")
    print(f"This DAG uses {len(set(gpu_ids))} GPUs")
    print(f"EP64_TP2 strategy needs 128 GPUs")
    print(f"This DAG appears to be a simplified example, not the full EP64 strategy")
    
    return dag_data

if __name__ == "__main__":
    check_dag_structure()