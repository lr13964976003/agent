#!/usr/bin/env python3
import re

def analyze_dag(dot_content):
    """Analyze the DAG for various issues"""
    
    # Extract all nodes and edges
    nodes = set()
    edges = []
    
    # Find all node definitions
    node_pattern = r'(\w+)\s*\[.*?\]'
    for match in re.finditer(node_pattern, dot_content, re.DOTALL):
        nodes.add(match.group(1))
    
    # Find all edges
    edge_pattern = r'(\w+)\s*->\s*(\w+)(?:\s*\[.*?\])?'
    for match in re.finditer(edge_pattern, dot_content):
        edges.append((match.group(1), match.group(2)))
    
    print(f"Total nodes found: {len(nodes)}")
    print(f"Total edges found: {len(edges)}")
    
    # Check 1: Node connectivity
    print("\n=== NODE CONNECTIVITY ANALYSIS ===")
    node_inputs = {}
    node_outputs = {}
    
    for node in nodes:
        node_inputs[node] = set()
        node_outputs[node] = set()
    
    for src, dst in edges:
        node_inputs[dst].add(src)
        node_outputs[src].add(dst)
    
    # Nodes with no inputs (except input node)
    no_input_nodes = [node for node, inputs in node_inputs.items() if len(inputs) == 0]
    print(f"Nodes with no inputs: {no_input_nodes}")
    
    # Nodes with no outputs (except output node)
    no_output_nodes = [node for node, outputs in node_outputs.items() if len(outputs) == 0]
    print(f"Nodes with no outputs: {no_output_nodes}")
    
    # Check 2: Attention block decomposition
    print("\n=== ATTENTION BLOCK ANALYSIS ===")
    attention_nodes = [node for node in nodes if 'attn' in node.lower() or 'attention' in node.lower()]
    print(f"Attention-related nodes: {len(attention_nodes)}")
    for node in attention_nodes:
        print(f"  - {node}")
    
    # Check 3: Communication nodes
    print("\n=== COMMUNICATION ANALYSIS ===")
    comm_nodes = [node for node in nodes if any(term in node.lower() for term in ['allreduce', 'alltoall', 'send', 'receive', 'broadcast'])]
    print(f"Communication nodes: {len(comm_nodes)}")
    for node in comm_nodes:
        print(f"  - {node}")
    
    # Check 4: Parallel strategy representation
    print("\n=== PARALLEL STRATEGY ANALYSIS ===")
    tp_nodes = [node for node in nodes if 'tp' in node.lower()]
    ep_nodes = [node for node in nodes if 'ep' in node.lower() or 'expert' in node.lower()]
    pp_nodes = [node for node in nodes if 'pp' in node.lower() or 'pipeline' in node.lower()]
    dp_nodes = [node for node in nodes if 'dp' in node.lower() or 'data' in node.lower()]
    
    print(f"Tensor Parallelism nodes: {len(tp_nodes)}")
    print(f"Expert Parallelism nodes: {len(ep_nodes)}")
    print(f"Pipeline Parallelism nodes: {len(pp_nodes)}")
    print(f"Data Parallelism nodes: {len(dp_nodes)}")
    
    # Check 5: Cycle detection
    print("\n=== CYCLE DETECTION ===")
    visited = set()
    
    def has_cycle_from_node(node, rec_stack):
        if node in rec_stack:
            return True
        if node in visited:
            return False
        
        visited.add(node)
        rec_stack.add(node)
        
        neighbors = [dst for src, dst in edges if src == node]
        for neighbor in neighbors:
            if has_cycle_from_node(neighbor, rec_stack.copy()):
                return True
        
        rec_stack.remove(node)
        return False
    
    cycle_found = False
    for node in nodes:
        if node not in visited:
            if has_cycle_from_node(node, set()):
                cycle_found = True
                break
    
    print(f"Cycle detected: {cycle_found}")
    
    # Check 6: Expert block analysis
    print("\n=== EXPERT BLOCK ANALYSIS ===")
    expert_nodes = [node for node in nodes if 'expert' in node.lower()]
    print(f"Expert nodes: {len(expert_nodes)}")
    
    # Check if experts are properly connected
    for expert in expert_nodes:
        if expert not in node_inputs or len(node_inputs[expert]) == 0:
            print(f"ERROR: Expert node {expert} has no inputs!")
        if expert not in node_outputs or len(node_outputs[expert]) == 0:
            print(f"ERROR: Expert node {expert} has no outputs!")
    
    return {
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'no_input_nodes': no_input_nodes,
        'no_output_nodes': no_output_nodes,
        'attention_nodes': attention_nodes,
        'communication_nodes': comm_nodes,
        'tp_nodes': tp_nodes,
        'ep_nodes': ep_nodes,
        'pp_nodes': pp_nodes,
        'dp_nodes': dp_nodes,
        'cycle_found': cycle_found,
        'expert_nodes': expert_nodes
    }

if __name__ == "__main__":
    with open('./current_dag.dot', 'r') as f:
        content = f.read()
    
    results = analyze_dag(content)
    
    # Save results
    with open('../outputs/2025-12-22-19-13-40/dag_analysis_results.md', 'w') as f:
        f.write("# DAG Analysis Results\n\n")
        f.write("## Summary\n")
        f.write(f"- Total nodes: {results['total_nodes']}\n")
        f.write(f"- Total edges: {results['total_edges']}\n")
        f.write(f"- Cycle detected: {results['cycle_found']}\n")
        f.write(f"- Nodes with no inputs: {len(results['no_input_nodes'])}\n")
        f.write(f"- Nodes with no outputs: {len(results['no_output_nodes'])}\n")
        f.write(f"- Attention nodes: {len(results['attention_nodes'])}\n")
        f.write(f"- Expert nodes: {len(results['expert_nodes'])}\n")
        f.write(f"- Communication nodes: {len(results['communication_nodes'])}\n")