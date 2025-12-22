#!/usr/bin/env python3
"""
Save DOT source files for the generated DAGs
"""

import graphviz
from generate_moe_dag_fixed import create_moe_deployment_dag, create_detailed_single_layer_dag

def main():
    # Create the DAGs
    main_dag = create_moe_deployment_dag()
    detailed_dag = create_detailed_single_layer_dag()
    
    # Save DOT source files explicitly
    main_dot_path = "../outputs/2025-12-22-10-42-54/moe_30b_deployment_ep64_tp8_pp2_dp2.dot"
    detailed_dot_path = "../outputs/2025-12-22-10-42-54/moe_30b_single_layer_detailed.dot"
    
    with open(main_dot_path, 'w') as f:
        f.write(main_dag.source)
    print(f"Main DOT source saved: {main_dot_path}")
    
    with open(detailed_dot_path, 'w') as f:
        f.write(detailed_dag.source)
    print(f"Detailed DOT source saved: {detailed_dot_path}")
    
    return main_dot_path, detailed_dot_path

if __name__ == "__main__":
    main_path, detailed_path = main()
    print(f"\nFiles saved:")
    print(f"  Main: {main_path}")
    print(f"  Detailed: {detailed_path}")