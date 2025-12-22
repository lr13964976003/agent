#!/usr/bin/env python3
"""
Generate Deployment DAG for 30B MoE Model Parallel Strategy
===========================================================

This script generates a comprehensive DAG representation of the optimal parallel
deployment strategy with EP64-TP8-PP2-DP2 configuration.
"""

import graphviz
from typing import Dict, List, Tuple
import json

class DeploymentDAGGenerator:
    """Generate DAG representation of parallel deployment strategy"""
    
    def __init__(self):
        self.config = {
            'ep_degree': 64,
            'tp_degree': 8,
            'pp_degree': 2,
            'dp_degree': 2,
            'num_layers': 16,
            'num_experts': 64,
            'hidden_size': 1024,
            'moe_hidden_size': 2048,
            'batch_size': 128
        }
    
    def generate_simplified_dag(self) -> graphviz.Digraph:
        """Generate simplified deployment DAG"""
        
        dot = graphviz.Digraph(comment='30B MoE Model Simplified Deployment DAG')
        dot.attr(rankdir='LR', size='12,8')
        dot.attr('node', shape='box', style='rounded,filled')
        
        # Main components
        dot.node('input', 'Input Batch\\n128 sequences', fillcolor='lightblue')
        
        # Parallel dimensions
        dot.node('dp', 'Data Parallel\\nDP=2', fillcolor='lightblue')
        dot.node('pp', 'Pipeline Parallel\\nPP=2', fillcolor='lightgreen')
        dot.node('ep', 'Expert Parallel\\nEP=64', fillcolor='lightyellow')
        dot.node('tp', 'Tensor Parallel\\nTP=8', fillcolor='lightcoral')
        
        # Processing flow
        dot.node('moe_layers', 'MoE Layers\\n16 total', fillcolor='lightyellow')
        dot.node('attention', 'Attention Layers\\nTP optimized', fillcolor='lightcoral')
        dot.node('output', 'Output\\nGenerated tokens', fillcolor='lightblue')
        
        # Edges
        dot.edge('input', 'dp')
        dot.edge('dp', 'pp')
        dot.edge('pp', 'ep')
        dot.edge('ep', 'tp')
        dot.edge('tp', 'moe_layers')
        dot.edge('moe_layers', 'attention')
        dot.edge('attention', 'output')
        
        return dot
    
    def generate_layer_wise_dag(self) -> graphviz.Digraph:
        """Generate layer-wise deployment DAG"""
        
        dot = graphviz.Digraph(comment='30B MoE Model Layer-wise Deployment DAG')
        dot.attr(rankdir='TB', size='16,20')
        dot.attr('node', shape='box', style='rounded,filled')
        
        # Add layers with parallel strategy details
        for layer_id in range(self.config['num_layers']):
            pp_stage = layer_id // (self.config['num_layers'] // self.config['pp_degree'])
            
            layer_node = f'layer_{layer_id}'
            
            if layer_id % 2 == 0:  # MoE layer
                dot.node(layer_node,
                    f'Layer {layer_id}\\nMoE Layer\\nPP Stage {pp_stage}\\n64 Experts (EP64)\\nTP8 Parallel',
                    fillcolor='lightyellow', shape='tripleoctagon')
            else:  # Dense layer
                dot.node(layer_node,
                    f'Layer {layer_id}\\nDense Layer\\nPP Stage {pp_stage}\\nTP8 Parallel',
                    fillcolor='lightcoral', shape='doubleoctagon')
            
            # Add edges between layers
            if layer_id > 0:
                prev_layer = f'layer_{layer_id-1}'
                if layer_id == self.config['num_layers'] // 2:  # Pipeline boundary
                    dot.edge(prev_layer, layer_node, label='Pipeline\\nCommunication',
                            style='dashed', color='green', penwidth='2')
                else:
                    dot.edge(prev_layer, layer_node, label='Layer\\nOutput')
        
        return dot
    
    def save_all_dags(self):
        """Generate and save all DAG representations"""
        
        print("Generating simplified deployment DAG...")
        simplified_dag = self.generate_simplified_dag()
        simplified_dag.render('llm_deployment_simplified', format='dot')
        simplified_dag.render('llm_deployment_simplified', format='svg')
        
        print("Generating layer-wise deployment DAG...")
        layer_wise_dag = self.generate_layer_wise_dag()
        layer_wise_dag.render('llm_deployment_layer_wise', format='dot')
        layer_wise_dag.render('llm_deployment_layer_wise', format='svg')
        
        print("All DAGs generated successfully!")
    
    def generate_deployment_summary(self) -> Dict:
        """Generate deployment summary statistics"""
        
        return {
            "total_gpus": self.config['ep_degree'] * self.config['tp_degree'] * 
                         self.config['pp_degree'] * self.config['dp_degree'],
            "parallel_dimensions": {
                "expert_parallelism": self.config['ep_degree'],
                "tensor_parallelism": self.config['tp_degree'],
                "pipeline_parallelism": self.config['pp_degree'],
                "data_parallelism": self.config['dp_degree']
            },
            "module_division": {
                "experts_per_gpu": 1,
                "layers_per_gpu": self.config['num_layers'] / 
                                 (self.config['pp_degree'] * self.config['tp_degree'] * 
                                  self.config['ep_degree'] * self.config['dp_degree']),
                "batch_per_gpu": self.config['batch_size'] / self.config['dp_degree']
            },
            "communication_patterns": {
                "all_to_all": "expert_dispatch_combine",
                "all_reduce": "tensor_synchronization",
                "send_recv": "pipeline_communication"
            },
            "optimization_targets": ["latency", "throughput", "memory_efficiency"]
        }


def main():
    """Main function to generate all deployment DAGs"""
    
    print("Starting deployment DAG generation...")
    
    # Create DAG generator
    generator = DeploymentDAGGenerator()
    
    # Generate all DAGs
    generator.save_all_dags()
    
    # Generate summary
    summary = generator.generate_deployment_summary()
    
    # Save summary to JSON
    with open('deployment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDeployment Summary:")
    print(f"Total GPUs: {summary['total_gpus']}")
    print(f"Module Division: {summary['module_division']}")
    print(f"Communication Patterns: {summary['communication_patterns']}")
    print(f"Optimization Targets: {summary['optimization_targets']}")
    
    print("\nDAG generation completed successfully!")


if __name__ == "__main__":
    main()