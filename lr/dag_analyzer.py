#!/usr/bin/env python3

import re
import sys

def parse_dag(file_path):
    """Parse the DAG file and extract nodes and edges"""
    nodes = set()
    edges = []
    node_labels = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract node definitions
    node_pattern = r'(\w+)\s*\[([^\]]+)\]'
    for match in re.finditer(node_pattern, content):
        node_id = match.group(1)
        attrs = match.group(2)
        nodes.add(node_id)
        
        # Extract label if present
        label_match = re.search(r'label="([^"]*)"', attrs, re.DOTALL)
        if label_match:
            node_labels[node_id] = label_match.group(1)
    
    # Extract edges
    edge_pattern = r'(\w+)\s*->\s*(\w+)\s*(?:\[([^\]]+)\])?'
    for match in re.finditer(edge_pattern, content):
        src = match.group(1)
        dst = match.group(2)
        edges.append((src, dst))
    
    return nodes, edges, node_labels

def analyze_dag(nodes, edges, node_labels):
    """Analyze the DAG based on the specified criteria"""
    
    # Build adjacency lists
    in_degree = {node: 0 for node in nodes}
    out_degree = {node: 0 for node in nodes}
    
    for src, dst in edges:
        out_degree[src] += 1
        in_degree[dst] += 1
    
    print("=== DAG Analysis Results ===")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    print()
    
    # Check 1: Nodes with only in-degree (no inputs)
    nodes_only_in = [node for node in nodes if in_degree[node] > 0 and out_degree[node] == 0]
    print(f"Nodes with only inputs (no outputs): {nodes_only_in}")
    
    # Check 2: Nodes with only out-degree (no outputs) 
    nodes_only_out = [node for node in nodes if out_degree[node] > 0 and in_degree[node] == 0]
    print(f"Nodes with only outputs (no inputs): {nodes_only_out}")
    
    # Check 3: Isolated nodes (no connections)
    isolated_nodes = [node for node in nodes if in_degree[node] == 0 and out_degree[node] == 0]
    print(f"Isolated nodes (no connections): {isolated_nodes}")
    
    print()
    
    # Check 4: All nodes except input should have at least one input
    input_nodes = ['input']  # Known input nodes
    nodes_without_inputs = [node for node in nodes if in_degree[node] == 0 and node not in input_nodes]
    print(f"Nodes without inputs (excluding known input nodes): {nodes_without_inputs}")
    
    # Check 5: All nodes except output should have at least one output
    output_nodes = ['output']  # Known output nodes
    nodes_without_outputs = [node for node in nodes if out_degree[node] == 0 and node not in output_nodes]
    print(f"Nodes without outputs (excluding known output nodes): {nodes_without_outputs}")
    
    print()
    
    # Check 6: Attention block breakdown
    attention_nodes = [node for node in nodes if 'attn' in node.lower()]
    print(f"Attention nodes found: {attention_nodes}")
    
    # Check if attention is broken down into submodules
    for attn_node in attention_nodes:
        print(f"  {attn_node}: {node_labels.get(attn_node, 'No label')}")
    
    print()
    
    # Check 7: Communication patterns
    comm_nodes = [node for node in nodes if 'comm' in node or 'allreduce' in node or 'all2all' in node or 'sendrecv' in node]
    print(f"Communication nodes: {comm_nodes}")
    
    print()
    
    # Check 8: Gate routers
    gate_nodes = [node for node in nodes if 'gate' in node.lower()]
    print(f"Gate router nodes: {gate_nodes}")
    
    print()
    
    # List all nodes with their degrees
    print("=== Node Degree Analysis ===")
    for node in sorted(nodes):
        print(f"{node}: in={in_degree[node]}, out={out_degree[node]}")
    
    return {
        'nodes_only_in': nodes_only_in,
        'nodes_only_out': nodes_only_out,
        'isolated_nodes': isolated_nodes,
        'nodes_without_inputs': nodes_without_inputs,
        'nodes_without_outputs': nodes_without_outputs,
        'attention_nodes': attention_nodes,
        'comm_nodes': comm_nodes,
        'gate_nodes': gate_nodes
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python dag_analyzer.py <dag_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    nodes, edges, node_labels = parse_dag(file_path)
    results = analyze_dag(nodes, edges, node_labels)
    
    # Check for cycles (simplified - would need proper cycle detection for complete analysis)
    print("\n=== Cycle Detection ===")
    print("Basic cycle check: Need to implement proper cycle detection")
    
    return results

if __name__ == "__main__":
    main()