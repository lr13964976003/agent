#!/usr/bin/env python3
"""
DAG Analysis Script
Checks for errors in DAGs according to specified criteria
"""

import re
import networkx as nx
from typing import Dict, List, Set, Tuple

def parse_dot_file(filepath: str) -> nx.DiGraph:
    """Parse a DOT file and return a NetworkX directed graph"""
    G = nx.DiGraph()
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract node names from subgraphs and main graph
    node_pattern = r'\b(\w+)\s*\[.*label="[^"]*"'
    nodes = re.findall(node_pattern, content)
    
    # Add nodes to graph
    for node in nodes:
        if node not in ['digraph', 'subgraph', 'cluster']:
            G.add_node(node)
    
    # Extract edges
    edge_pattern = r'(\w+)\s*->\s*(\w+)'
    edges = re.findall(edge_pattern, content)
    
    # Add edges to graph
    for src, dst in edges:
        if src in G.nodes() and dst in G.nodes():
            G.add_edge(src, dst)
    
    return G

def analyze_parallel_strategy(G: nx.DiGraph, strategy_name: str, content: str) -> List[str]:
    """Check if parallel strategy is fully reflected"""
    issues = []
    
    # Extract strategy parameters from comment
    strategy_match = re.search(r'//\s*(\w+):\s*([^\s]+)', content)
    if strategy_match:
        strategy_params = strategy_match.group(2)
        
        # Check for TP (Tensor Parallelism) components
        tp_match = re.search(r'TP(\d+)', strategy_params)
        if tp_match:
            tp_size = int(tp_match.group(1))
            # Count TP nodes
            tp_nodes = [n for n in G.nodes() if 'TP' in n or re.search(r'TP\d+', content) if n in G.nodes()]
            if len(tp_nodes) < 3:  # Should have at least QKV, Score, Output for attention
                issues.append(f"Insufficient TP decomposition for TP{tp_size}")
        
        # Check for EP (Expert Parallelism) components
        ep_match = re.search(r'EP(\d+)', strategy_params)
        if ep_match:
            ep_size = int(ep_match.group(1))
            # Count expert nodes
            expert_nodes = [n for n in G.nodes() if 'expert' in n.lower() or 'Expert' in n]
            if len(expert_nodes) < ep_size // 4:  # Should have reasonable number of expert groups
                issues.append(f"Insufficient expert decomposition for EP{ep_size}")
    
    return issues

def check_gpu_communications(G: nx.DiGraph, content: str) -> List[str]:
    """Check if all GPU communications are identified"""
    issues = []
    
    # Look for communication patterns
    communication_keywords = ['allreduce', 'all-to-all', 'dispatch', 'combine']
    communication_nodes = []
    
    for node in G.nodes():
        if any(keyword in node.lower() for keyword in communication_keywords):
            communication_nodes.append(node)
    
    # Check if we have expected communication patterns
    expected_patterns = ['allreduce', 'dispatch', 'combine']
    for pattern in expected_patterns:
        found = any(pattern in node.lower() for node in communication_nodes)
        if not found:
            issues.append(f"Missing {pattern} communication pattern")
    
    return issues

def check_attention_decomposition(G: nx.DiGraph, content: str) -> List[str]:
    """Check if attention blocks are broken down into submodules"""
    issues = []
    
    # Look for attention nodes
    attention_nodes = [n for n in G.nodes() if 'attn' in n.lower() or 'attention' in n.lower()]
    
    if attention_nodes:
        # Check for expected attention submodules
        expected_submodules = ['qkv', 'score', 'output', 'allreduce']
        
        for node in attention_nodes:
            found_submodules = []
            for submodule in expected_submodules:
                if submodule in node.lower():
                    found_submodules.append(submodule)
            
            if not found_submodules:
                issues.append(f"Attention node {node} lacks proper submodule decomposition")
    
    # Check if we have the basic attention pipeline
    qkv_nodes = [n for n in attention_nodes if 'qkv' in n.lower()]
    score_nodes = [n for n in attention_nodes if 'score' in n.lower()]
    output_nodes = [n for n in attention_nodes if 'output' in n.lower()]
    
    if not qkv_nodes:
        issues.append("Missing QKV decomposition in attention")
    if not score_nodes:
        issues.append("Missing score computation in attention")
    if not output_nodes:
        issues.append("Missing output projection in attention")
    
    return issues

def check_node_connectivity(G: nx.DiGraph) -> List[str]:
    """Check if all nodes have proper input/output connections"""
    issues = []
    
    # Check nodes with only input (should be output node)
    input_only = [n for n in G.nodes() if G.in_degree(n) == 0]
    output_only = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    # The input_only should only contain 'input' node
    if len(input_only) != 1 or input_only[0] != 'input':
        issues.append(f"Nodes with no inputs should only be 'input' node, found: {input_only}")
    
    # The output_only should only contain 'output' node
    if len(output_only) != 1 or output_only[0] != 'output':
        issues.append(f"Nodes with no outputs should only be 'output' node, found: {output_only}")
    
    # Check if all other nodes have both input and output
    for node in G.nodes():
        if node not in ['input', 'output']:
            if G.in_degree(node) == 0:
                issues.append(f"Node {node} has no inputs")
            if G.out_degree(node) == 0:
                issues.append(f"Node {node} has no outputs")
    
    return issues

def comprehensive_dag_analysis(filepath: str) -> Dict[str, List[str]]:
    """Perform comprehensive analysis of DAG"""
    
    # Read file content
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse graph
    G = parse_dot_file(filepath)
    
    results = {
        'parallel_strategy': [],
        'gpu_communications': [],
        'cycles': [],
        'attention_decomposition': [],
        'node_connectivity': [],
        'overall': []
    }
    
    # Check for cycles
    try:
        cycle = nx.find_cycle(G)
        if cycle:
            results['cycles'].append(f"Cycle detected: {cycle}")
    except nx.NetworkXNoCycle:
        pass  # No cycle found, which is good
    
    # Check parallel strategy
    results['parallel_strategy'] = analyze_parallel_strategy(G, filepath, content)
    
    # Check GPU communications
    results['gpu_communications'] = check_gpu_communications(G, content)
    
    # Check attention decomposition
    results['attention_decomposition'] = check_attention_decomposition(G, content)
    
    # Check node connectivity
    results['node_connectivity'] = check_node_connectivity(G)
    
    return results

def main():
    # Analyze both DAGs
    current_results = comprehensive_dag_analysis("../outputs/2025-12-22-19-13-40/current_strategy.dot")
    optimal_results = comprehensive_dag_analysis("../outputs/2025-12-22-19-13-40/optimal_strategy.dot")
    
    print("=== CURRENT STRATEGY DAG ANALYSIS ===")
    for category, issues in current_results.items():
        print(f"\n{category.upper()}:")
        if issues:
            for issue in issues:
                print(f"  ❌ {issue}")
        else:
            print("  ✅ No issues found")
    
    print("\n\n=== OPTIMAL STRATEGY DAG ANALYSIS ===")
    for category, issues in optimal_results.items():
        print(f"\n{category.upper()}:")
        if issues:
            for issue in issues:
                print(f"  ❌ {issue}")
        else:
            print("  ✅ No issues found")
    
    # Check if there are any errors that need modification
    current_has_errors = any(len(issues) > 0 for issues in current_results.values())
    optimal_has_errors = any(len(issues) > 0 for issues in optimal_results.values())
    
    if current_has_errors or optimal_has_errors:
        print("\n=== ERRORS FOUND ===")
        if current_has_errors:
            print("Current strategy DAG has errors that need modification")
        if optimal_has_errors:
            print("Optimal strategy DAG has errors that need modification")
    else:
        print("\n=== NO ERRORS FOUND ===")
        print("Congratulation!! Both DAGs are correct")

if __name__ == "__mainSecurit":
    main()