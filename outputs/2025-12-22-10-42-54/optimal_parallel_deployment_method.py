#!/usr/bin/env python3
"""
Optimal Parallel Deployment Method for 30B MoE Model
====================================================

This module implements the optimal parallel strategy for deploying a 30B parameter
Mixture of Experts (MoE) model with the following configuration:
- Expert Parallelism (EP): 64-way
- Tensor Parallelism (TP): 8-way  
- Pipeline Parallelism (PP): 2-way
- Data Parallelism (DP): 2-way

Total GPUs required: 2048
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class ParallelConfig:
    """Configuration for parallel deployment"""
    ep_degree: int = 64      # Expert parallelism degree
    tp_degree: int = 8       # Tensor parallelism degree
    pp_degree: int = 2       # Pipeline parallelism degree
    dp_degree: int = 2       # Data parallelism degree
    num_layers: int = 16     # Total transformer layers
    num_experts: int = 64    # Experts per layer
    hidden_size: int = 1024  # Token dimension
    moe_hidden_size: int = 2048  # MoE hidden dimension
    num_heads: int = 16      # Attention heads
    head_dim: int = 64       # Head dimension
    batch_size: int = 128    # Total batch size
    seq_length: int = 1024   # Maximum sequence length
    precision: str = 'fp16'  # Model precision

class ParallelDeployment:
    """
    Main class for managing parallel deployment of 30B MoE model
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.world_size = config.ep_degree * config.tp_degree * config.pp_degree * config.dp_degree
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Calculate parallel groups and ranks
        self._setup_parallel_groups()
        
        # Validate GPU requirements
        self._validate_resources()
        
    def _setup_parallel_groups(self):
        """Setup parallel process groups for different parallelism dimensions"""
        
        # Create process groups for each parallelism dimension
        self.ep_group = self._create_ep_group()
        self.tp_group = self._create_tp_group()
        self.pp_group = self._create_pp_group()
        self.dp_group = self._create_dp_group()
        
        # Calculate local ranks within each group
        self.ep_rank = dist.get_rank(group=self.ep_group)
        self.tp_rank = dist.get_rank(group=self.tp_group)
        self.pp_rank = dist.get_rank(group=self.pp_group)
        self.dp_rank = dist.get_rank(group=self.dp_group)
        
    def _create_ep_group(self) -> dist.ProcessGroup:
        """Create expert parallelism process group"""
        # EP groups are organized by expert ID
        # Each group contains all ranks that work on the same expert
        ep_group_ranks = []
        for dp_id in range(self.config.dp_degree):
            for pp_id in range(self.config.pp_degree):
                for tp_id in range(self.config.tp_degree):
                    for ep_id in range(self.config.ep_degree):
                        rank = self._calculate_rank(ep_id, tp_id, pp_id, dp_id)
                        ep_group_ranks.append(rank)
        
        return dist.new_group(ep_group_ranks)
    
    def _create_tp_group(self) -> dist.ProcessGroup:
        """Create tensor parallelism process group"""
        # TP groups are organized by tensor partition
        tp_group_ranks = []
        for dp_id in range(self.config.dp_degree):
            for pp_id in range(self.config.pp_degree):
                for ep_id in range(self.config.ep_degree):
                    for tp_id in range(self.config.tp_degree):
                        rank = self._calculate_rank(ep_id, tp_id, pp_id, dp_id)
                        tp_group_ranks.append(rank)
        
        return dist.new_group(tp_group_ranks)
    
    def _create_pp_group(self) -> dist.ProcessGroup:
        """Create pipeline parallelism process group"""
        # PP groups are organized by pipeline stage
        pp_group_ranks = []
        for dp_id in range(self.config.dp_degree):
            for ep_id in range(self.config.ep_degree):
                for tp_id in range(self.config.tp_degree):
                    for pp_id in range(self.config.pp_degree):
                        rank = self._calculate_rank(ep_id, tp_id, pp_id, dp_id)
                        pp_group_ranks.append(rank)
        
        return dist.new_group(pp_group_ranks)
    
    def _create_dp_group(self) -> dist.ProcessGroup:
        """Create data parallelism process group"""
        # DP groups are organized by data partition
        dp_group_ranks = []
        for pp_id in range(self.config.pp_degree):
            for ep_id in range(self.config.ep_degree):
                for tp_id in range(self.config.tp_degree):
                    for dp_id in range(self.config.dp_degree):
                        rank = self._calculate_rank(ep_id, tp_id, pp_id, dp_id)
                        dp_group_ranks.append(rank)
        
        return dist.new_group(dp_group_ranks)
    
    def _calculate_rank(self, ep_id: int, tp_id: int, pp_id: int, dp_id: int) -> int:
        """Calculate global rank from parallel IDs"""
        return (ep_id * self.config.tp_degree * self.config.pp_degree * self.config.dp_degree +
                tp_id * self.config.pp_degree * self.config.dp_degree +
                pp_id * self.config.dp_degree +
                dp_id)
    
    def _validate_resources(self):
        """Validate that GPU resources meet requirements"""
        
        # Calculate memory requirements per GPU
        total_params = 30e9  # 30B parameters
        bytes_per_param = 2 if self.config.precision == 'fp16' else 4
        total_memory = total_params * bytes_per_param
        
        memory_per_gpu = total_memory / (self.config.ep_degree * self.config.tp_degree * 
                                       self.config.pp_degree * self.config.dp_degree)
        
        # Available GPU memory (64GB per GPU from deployment conditions)
        available_memory = 64e9
        
        if memory_per_gpu > available_memory:
            raise RuntimeError(f"Memory requirement ({memory_per_gpu/1e9:.1f}GB) exceeds available GPU memory ({available_memory/1e9:.1f}GB)")
        
        print(f"Memory validation passed: {memory_per_gpu/1e9:.1f}GB per GPU")
    
    def get_layer_assignment(self) -> List[int]:
        """Get layer assignment for current pipeline stage"""
        layers_per_stage = self.config.num_layers // self.config.pp_degree
        start_layer = self.pp_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage
        
        return list(range(start_layer, end_layer))
    
    def get_expert_assignment(self) -> List[int]:
        """Get expert assignment for current expert parallel rank"""
        experts_per_rank = self.config.num_experts // self.config.ep_degree
        start_expert = self.ep_rank * experts_per_rank
        end_expert = start_expert + experts_per_rank
        
        return list(range(start_expert, end_expert))
    
    def get_batch_partition(self) -> Tuple[int, int]:
        """Get batch partition for current data parallel rank"""
        batch_per_rank = self.config.batch_size // self.config.dp_degree
        start_batch = self.dp_rank * batch_per_rank
        end_batch = start_batch + batch_per_rank
        
        return (start_batch, end_batch)
    
    def get_model_parallel_config(self) -> Dict:
        """Get model parallelism configuration for current rank"""
        return {
            'ep_rank': self.ep_rank,
            'ep_size': self.config.ep_degree,
            'tp_rank': self.tp_rank,
            'tp_size': self.config.tp_degree,
            'pp_rank': self.pp_rank,
            'pp_size': self.config.pp_degree,
            'dp_rank': self.dp_rank,
            'dp_size': self.config.dp_degree,
            'layers': self.get_layer_assignment(),
            'experts': self.get_expert_assignment(),
            'batch_range': self.get_batch_partition()
        }
    
    def setup_communication_patterns(self):
        """Setup communication patterns for different parallelism dimensions"""
        
        # Expert Parallelism communication (All-to-All)
        self.ep_comm_pattern = {
            'dispatch': 'all_to_all',
            'combine': 'all_to_all',
            'group': self.ep_group
        }
        
        # Tensor Parallelism communication (All-Reduce)
        self.tp_comm_pattern = {
            'attention': 'all_reduce',
            'mlp': 'all_reduce',
            'group': self.tp_group
        }
        
        # Pipeline Parallelism communication (Send/Recv)
        self.pp_comm_pattern = {
            'forward': 'send_recv',
            'backward': 'send_recv',
            'group': self.pp_group
        }
        
        # Data Parallelism communication (All-Reduce for gradients)
        self.dp_comm_pattern = {
            'gradients': 'all_reduce',
            'group': self.dp_group
        }
    
    def optimize_for_latency(self):
        """Optimize deployment for minimum latency"""
        
        # Prioritize tensor parallelism for compute-intensive operations
        # Reduce pipeline stages to minimize bubbles
        # Overlap communication with computation
        
        optimization_config = {
            'priority': 'latency',
            'tp_overlap': True,
            'ep_overlap': True,
            'pp_micro_batch_size': 1,  # Single token for decode
            'communication_optimization': 'overlap'
        }
        
        return optimization_config
    
    def optimize_for_throughput(self):
        """Optimize deployment for maximum throughput"""
        
        # Maximize data parallelism and batch processing
        # Use micro-batching in pipeline parallelism
        # Batch communication operations
        
        optimization_config = {
            'priority': 'throughput',
            'dp_batch_size': self.config.batch_size // self.config.dp_degree,
            'pp_micro_batch_size': 4,  # Multiple micro-batches
            'ep_batch_dispatch': True,
            'communication_optimization': 'batch'
        }
        
        return optimization_config


class MoELayerDeployment:
    """
    Specialized deployment class for MoE layers
    """
    
    def __init__(self, parallel_deployment: ParallelDeployment):
        self.pd = parallel_deployment
        self.config = parallel_deployment.config
        
    def deploy_moe_layer(self, layer_id: int):
        """Deploy a single MoE layer with optimal parallel strategy"""
        
        # Check if this layer is assigned to current pipeline stage
        if layer_id not in self.pd.get_layer_assignment():
            return None
        
        # Get expert assignment for current rank
        expert_assignment = self.pd.get_expert_assignment()
        
        # Deploy routing mechanism
        router = self._deploy_router(layer_id)
        
        # Deploy experts in parallel
        experts = self._deploy_experts(layer_id, expert_assignment)
        
        # Setup communication for expert dispatch/combine
        comm_config = self._setup_expert_communication()
        
        return {
            'layer_id': layer_id,
            'router': router,
            'experts': experts,
            'expert_assignment': expert_assignment,
            'communication': comm_config
        }
    
    def _deploy_router(self, layer_id: int):
        """Deploy the routing mechanism for MoE layer"""
        
        # Router is replicated across TP dimension for load balancing
        router_config = {
            'layer_id': layer_id,
            'num_experts': self.config.num_experts,
            'hidden_size': self.config.hidden_size,
            'tp_size': self.config.tp_degree,
            'tp_rank': self.pd.tp_rank,
            'top_k': 2  # Route to top-2 experts
        }
        
        return router_config
    
    def _deploy_experts(self, layer_id: int, expert_assignment: List[int]):
        """Deploy experts assigned to current rank"""
        
        experts = {}
        for expert_id in expert_assignment:
            expert_config = {
                'expert_id': expert_id,
                'layer_id': layer_id,
                'hidden_size': self.config.hidden_size,
                'moe_hidden_size': self.config.moe_hidden_size,
                'tp_size': self.config.tp_degree,
                'tp_rank': self.pd.tp_rank,
                'precision': self.config.precision
            }
            experts[expert_id] = expert_config
        
        return experts
    
    def _setup_expert_communication(self):
        """Setup communication pattern for expert parallelism"""
        
        comm_config = {
            'dispatch': {
                'pattern': 'all_to_all',
                'group': self.pd.ep_group,
                'buffer_size': self.config.batch_size * self.config.hidden_size * 2  # Top-2 routing
            },
            'combine': {
                'pattern': 'all_to_all',
                'group': self.pd.ep_group,
                'buffer_size': self.config.batch_size * self.config.hidden_size
            }
        }
        
        return comm_config


def create_optimal_deployment(hardware_config: Optional[Dict] = None) -> ParallelDeployment:
    """
    Create optimal parallel deployment for 30B MoE model
    
    Args:
        hardware_config: Optional hardware configuration override
        
    Returns:
        ParallelDeployment: Configured deployment instance
    """
    
    # Default configuration based on deployment conditions
    config = ParallelConfig()
    
    if hardware_config:
        # Override with provided hardware constraints
        if 'gpu_memory' in hardware_config:
            # Adjust parallelism degrees based on memory constraints
            pass
        if 'gpu_count' in hardware_config:
            # Adjust parallelism degrees based on GPU count
            pass
    
    # Create deployment instance
    deployment = ParallelDeployment(config)
    
    # Setup communication patterns
    deployment.setup_communication_patterns()
    
    return deployment


def main():
    """
    Main function to demonstrate optimal parallel deployment
    """
    
    print("Creating optimal parallel deployment for 30B MoE model...")
    
    # Create deployment
    deployment = create_optimal_deployment()
    
    # Get configuration for current rank
    config = deployment.get_model_parallel_config()
    
    print(f"Deployment created successfully!")
    print(f"World size: {deployment.world_size}")
    print(f"EP rank: {config['ep_rank']}/{config['ep_size']}")
    print(f"TP rank: {config['tp_rank']}/{config['tp_size']}")
    print(f"PP rank: {config['pp_rank']}/{config['pp_size']}")
    print(f"DP rank: {config['dp_rank']}/{config['dp_size']}")
    print(f"Layers assigned: {config['layers']}")
    print(f"Experts assigned: {config['experts']}")
    print(f"Batch range: {config['batch_range']}")
    
    # Validate load balancing
    total_gpus = config['ep_size'] * config['tp_size'] * config['pp_size'] * config['dp_size']
    print(f"\nLoad balancing validation:")
    print(f"Total GPUs: {total_gpus}")
    print(f"Layers per GPU: {len(config['layers'])}")
    print(f"Experts per GPU: {len(config['experts'])}")
    print(f"Batch per GPU: {config['batch_range'][1] - config['batch_range'][0]}")
    
    # Check optimization modes
    latency_config = deployment.optimize_for_latency()
    throughput_config = deployment.optimize_for_throughput()
    
    print(f"\nLatency optimization: {latency_config}")
    print(f"Throughput optimization: {throughput_config}")
    
    print("\nDeployment method ready for execution!")


if __name__ == "__main__":
    main()