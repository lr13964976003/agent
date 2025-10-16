#!/usr/bin/env python3
"""
Generate SVG visualization for the MA Separation Model DAG
"""

import os
import subprocess
from graphviz import Source

def validate_dag_structure(dot_file_path):
    """Validate the DAG structure for connectivity issues"""
    with open(dot_file_path, 'r') as f:
        content = f.read()
    
    # Check for layer connectivity
    issues = []
    
    # Check for required connections
    layer_outputs = ['layer_0_output', 'layer_1_output', 'layer_2_output']
    layer_distributes = ['layer_1_distribute', 'layer_2_distribute', 'layer_3_distribute']
    
    for output, distribute in zip(layer_outputs, layer_distributes):
        connection = f"{output} -> {distribute}"
        if connection not in content:
            issues.append(f"Missing connection: {output} -> {distribute}")
    
    return issues

def main():
    dot_file = "./outputs/2025-10-16-12-21-24/ma_separation_model_fixed.dot"
    svg_file = "./outputs/2025-10-16-12-21-24/ma_separation_model_fixed.svg"
    
    # Validate the DAG structure
    issues = validate_dag_structure(dot_file)
    if issues:
        print("DAG validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return
    
    # Generate SVG
    try:
        with open(dot_file, 'r') as f:
            dot_content = f.read()
        
        # Create Source object
        src = Source(dot_content)
        
        # Generate SVG
        src.format = 'svg'
        src.render(dot_file.replace('.dot', ''), cleanup=True)
        
        print(f"SVG generated successfully: {svg_file}")
        
        # Copy to the required location
        src.render(dot_file.replace('ma_separation_model_fixed.dot', 'ma_separation_model'), cleanup=True)
        
    except Exception as e:
        print(f"Error generating SVG: {e}")

if __name__ == "__main__":
    main()