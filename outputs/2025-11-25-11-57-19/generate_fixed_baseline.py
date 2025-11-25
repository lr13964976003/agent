#!/usr/bin/env python3
import graphviz
import os

def generate_baseline_fixed():
    # Read the fixed DOT file
    with open('../outputs/2025-11-25-11-57-19/baseline_moe_dag_fixed.dot', 'r') as f:
        dot_content = f.read()
    
    # Create the graph
    graph = graphviz.Source(dot_content)
    
    # Render to SVG
    graph.render('../outputs/2025-11-25-11-57-19/baseline_moe_dag_fixed', format='svg', cleanup=False)
    
    print("Fixed baseline DAG SVG generated successfully")
    return True

if __name__ == "__main__":
    generate_baseline_fixed()