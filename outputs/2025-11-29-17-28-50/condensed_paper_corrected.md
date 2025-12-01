# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization.

Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks and limiting true expert parallelism. Our cross-node expert parallelism method distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design shifts optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## 2. Methods

### of Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the bottleneck from inter-expert contention to network communication, which can be mitigated through careful scheduling, routing, and overlapping of communication and computation.

### 2.1 Expert Placement Strategy

**Single-Expert-Per-GPU Deployment**: Deploy at most one expert per GPU to maximize expert-level parallelism. For E experts and G GPUs, ensure each expert is assigned to a distinct GPU if E ≤ G. If E > G, replicate experts across GPUs to maximize concurrency while balancing memory usage.

**Cross-Node Distribution**: Use topology-aware placement considering node-to-node bandwidth, latency, GPU memory capacity, and expected token routing patterns. Minimize the maximum number of tokens sent across any single link while maintaining the one-expert-per-GPU principle.

### 2.2 Routing and Load Balancing

**Gating Mechanism**: Use standard top-K gating scores to determine which experts are activated for each input token.

**Token Sharding Across Nodes**: 
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlapping expert computation  
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### 2.3 Communication Overlap and Scheduling

**Overlapping Compute and Communication**: Interleave expert computation and communication using CUDA streams or asynchronous libraries (NCCL/MPI) to prevent data transfer blocking GPU computation.

**Pipeline Scheduling**: In multi-layer MoE networks, ensure token outputs from previous layers are immediately routed to next layer's experts, with subsequent layers starting processing as soon as partial batches arrive.

### 2.4 Large EP Regime (EP ≥ 16)

Network bandwidth becomes the primary limiting factor in large EP setups. Mitigate through topology-aware routing and token batching. The one-expert-per-GPU policy ensures all GPUs are fully utilized for compute while communication costs are amortized across many tokens.

### 2.5 Model Architecture Clarification

**Note on Model Configuration**: The original paper contains conflicting specifications regarding the number of layers. The experiments section describes a 16-layer MoE model, while the conclusion references a 4-layer model. For consistency with the detailed experimental setup and results, this refined paper adopts the **16-layer configuration** as the primary model architecture, with 16 experts per layer, as this provides the complete technical specification necessary for deployment.

## 3. Experiments

### 3.1 Experimental Setup

**Model Configuration**:
- Architecture: 16-layer MoE with 16 experts per layer
- Expert Type: Multi-Layer Perceptron (MLP)
- Precision: BF16
- Batch Size: 128 sequences
- Sequence Length: 10,000 tokens
- Token Dimension: 4096
- MHA: 32 heads, 128 dimensions per head
- MLP Hidden Size: 16384

**Environment**: Inference-only setting using adequate H100 GPUs

### 3.2 Deployment Configurations

**Baseline (TP=8, PP=2)**:
- Tensor Parallelism = 8, Pipeline Parallelism = 2
- Each GPU holds tensor-parallel shard for all layers
- Multiple experts colocated on same GPU
- Tokens flow sequentially through pipeline stages

**Proposed Cross-Node Expert Parallelism**:
- Expert Parallelism = 16 (large EP regime)
- Each GPU hosts exactly one expert per layer
- Input tokens dynamically routed to GPU holding corresponding expert
- Token batches asynchronously sent to minimize idle time

### 3.3 Results

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | adequate | TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | adequate | 1 expert per layer per GPU | 450,000 | 2.2 |

**Performance Improvements**:
- Throughput: 3.75× higher (450,000 vs 120,000 TPS)
- Latency: 3.8× lower (2.2ms vs 8.3ms TPOT)

### 3.4 Key Findings

1. **Expert Isolation**: One expert per GPU eliminates intra-GPU contention
2. **Parallel Efficiency**: All 16 experts compute simultaneously  
3. **Communication Overlap**: Asynchronous routing prevents waiting
4. **Scalability**: Near-linear scaling in large EP regime (EP ≥ 16)

## 4. Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. By shifting the computational bottleneck from intra-GPU contention to communication (effectively mitigated through asynchronous token routing and topology-aware placement), we achieve 3.75× higher throughput and 3.8× lower latency compared to baseline configurations.

The approach provides a scalable blueprint for high-performance MoE inference in environments with abundant GPU resources, demonstrating that distributing experts across GPUs and overlapping communication with computation can dramatically improve performance for large-scale MoE deployments.

## 5. Technical Specifications for Deployment

**Critical Parameters**:
- Expert count must equal or exceed GPU count for optimal performance
- Network infrastructure must support high-bandwidth cross-node communication
- Batch size: 128 sequences, Sequence length: 10,000 tokens
- Model dimensions: Token dimension = 4096, MLP hidden size = 16384

**Parallel Strategy Requirements**:
- Expert Parallelism ≥ 16 for large EP regime
- Topology-aware expert placement
- Asynchronous token routing with batching
- CUDA streams for compute-communication overlap

**Hardware Requirements**:
- H100-class GPUs with sufficient memory per expert
- High-performance interconnects (NVLink, InfiniBand)
- Topology-aware scheduling for optimal expert placement

**Complete Deployment Configuration JSON:**

```json
{
  "deployment_configurations": {
    "baseline": {
      "name": "Baseline TP=8 PP=2 Deployment",
      "parallel_strategy": {
        "tensor_parallelism": 8,
        "pipeline_parallelism": 2,
        "expert_parallelism": 1,
        "data_parallelism": 1
      },
      "model_specifications": {
        "layers": 16,
        "experts_per_layer": 16,
        "token_dimension": 4096,
        "mha_heads": 32,
        "mha_head_dimension": 128,
        "mlp_hidden_size": 16384,
        "precision": "BF16",
        "batch_size": 128,
        "sequence_length": 10000
      },
      "device_mapping": {
        "GPU_0": {
          "pipeline_stage": 0,
          "tensor_parallel_rank": 0,
          "layers": "0-7",
          "experts": "Expert_0, Expert_1, Expert_2, Expert_3, Expert_4, Expert_5, Expert_6, Expert_7",
          "shard_responsibility": "Tensor parallel shard 0 for all layers in stage 0"
        },
        "GPU_1": {
          "pipeline_stage": 0,
          "tensor_parallel_rank": 1,
          "layers": "0-7",
          "experts": "Expert_0, Expert_1, Expert_2, Expert_3, Expert_4, Expert_5, Expert_6, Expert_7",
          "shard_responsibility": "Tensor parallel shard 1 for all layers in stage 0"
        },
        "GPU_2": {
          "pipeline_stage": 0,
          "tensor_parallel_rank": 2,
          "layers": "0-7",
          "experts": "Expert_0, Expert_1, Expert_2, Expert_3, Expert_4, Expert_5, Expert_6, Expert_7",
          "shard_responsibility": "Tensor parallel shard 2 for all layers in stage 0"
        },
        "GPU_3": {
          "pipeline_stage": 0,
          "tensor_parallel_rank": 3,
          "layers": "0-7",
          "experts": "Expert_0, Expert_1, Expert_2, Expert_3, Expert_4, Expert_5, Expert_6, Expert_7",
          "shard_responsibility": "Tensor parallel shard 3 for all layers in stage 0"
        },
        "GPU_4": {
          "pipeline_stage": 0,
          "tensor_parallel_rank": 4,
          "layers": "0-7",
          "experts": "Expert_0, Expert_1, Expert_2, Expert_3, Expert_4, Expert_5, Expert_6, Expert_7",
          "shard_responsibility": "Tensor parallel shard 4 for all layers in stage 0"
        },
        "GPU_5": {
          "pipeline_stage": 0,
          "tensor_parallel_rank": 5,
          "layers": "0-7",
          "experts": "Expert_0, Expert_1, Expert_2, Expert_3, Expert_4, Expert_5, Expert_6, Expert_7",
          "shard_responsibility": "Tensor parallel shard 5 for all layers in stage 0"
        },
        "GPU_6": {
          "pipeline_stage": 0,
          "tensor_parallel_rank": 6,
          "layers": "0-7",
          "experts": "Expert_0, Expert_1, Expert_2, Expert_3, Expert_4, Expert_5, Expert_6, Expert_7",
          "shard_responsibility": "Tensor parallel shard 6 for all layers in stage 0"
        },
        "GPU_7": {
          "pipeline_stage": 0,
          "tensor_parallel_rank": 7,
          "layers": "0-7",
          "experts": "Expert_0, Expert_1, Expert_2, Expert_3, Expert_4, Expert_5, Expert_6, Expert_7",
          "shard_responsibility": "Tensor parallel shard 7 for all layers in stage 0"
        },
        "GPU_8": {
          "pipeline_stage": 1,
          "tensor_parallel_rank": 0,
          "layers": "8-15",
          "experts": "Expert_8, Expert_9, Expert_10, Expert_11, Expert_12, Expert_13, Expert_14, Expert_15",
          "shard_responsibility": "Tensor parallel shard 0 for all layers in stage 1"
        },
        "GPU_9": {
          "pipeline_stage": 1,
          "tensor_parallel_rank": 1,
          "layers": "8-15",
          "experts": "Expert_8, Expert_9, Expert_10, Expert_11, Expert_12, Expert_13, Expert_14, Expert_15",
          "shard_responsibility": "Tensor parallel shard 1 for all layers in stage 1"
        },
        "GPU_10": {
          "pipeline_stage": 1,
          "tensor_parallel_rank": 2,
          "layers": "8-15",
          "experts": "Expert_8, Expert_9, Expert_10, Expert_11, Expert_12, Expert_13, Expert_14, Expert_15",
          "shard_responsibility": "Tensor parallel shard 2 for all layers in stage 1"
        },
        "GPU_11": {
          "pipeline_stage": 1,
          "tensor_parallel_rank": 3,
          "layers": "8-15",
          "experts": "Expert_8, Expert_9, Expert_10, Expert_11, Expert_12, Expert_13, Expert_14, Expert_15",
          "shard_responsibility": "Tensor parallel shard 3 for all layers in stage 1"
        },
        "GPU_12": {
          "pipeline_stage": 1,
          "tensor_parallel_rank": 4,
          "layers": "8-15",
          "experts": "Expert_8, Expert_9, Expert_10, Expert_11, Expert_12, Expert_13, Expert_14, Expert_15",
          "shard_responsibility": "Tensor parallel shard 4 for all layers in stage 1"
        },
        "GPU_13": {
          "pipeline_stage": 1,
          "tensor_parallel_rank": 5,
          "layers": "8-15",
          "experts": "Expert_8, Expert_9, Expert_10, Expert_11, Expert_12, Expert_13, Expert_14, Expert_15",
          "shard_responsibility": "Tensor parallel shard 5 for all layers in stage 1"
        },
        "GPU_14": {
          "pipeline_stage": 1,
          "tensor_parallel_rank": 6,
          "layers": "8-15",
          "experts": "Expert_8, Expert_9, Expert_10, Expert_11, Expert_12, Expert_13, Expert_14, Expert_15",
          "shard_responsibility": "Tensor parallel shard 6 for all layers in stage 1"
        },
        "GPU_15": {
          "pipeline_stage": 1,
          "tensor_parallel_rank": 7,
          "layers": "8-15",
          "experts": "Expert_8, Expert_9, Expert_10, Expert_11, Expert_12, Expert_13, Expert_14, Expert_15",
          "shard_responsibility": "Tensor parallel shard 7 for all layers in stage 1"
        }
      },
      "performance_metrics": {
        "throughput": {
          "value": 120000,
          "unit": "tokens_per_second"
        },
        "latency": {
          "value": 8.3,
          "unit": "ms_per_token"
        }
      }
    },
    "proposed": {
      "name": "Proposed Cross-Node Expert Parallelism",
      "parallel_strategy": {
        "expert_parallelism": 16,
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "data_parallelism": 1
      },
      "model_specifications": {
        "layers": 16,
        "experts_per_layer": 16,
        "token_dimension": 4096,
        "mha_heads": 32,
        "mha_head_dimension": 128,
        "mlp_hidden_size": 16384,
        "precision": "BF16",
        "batch_size": 128,
        "sequence_length": 10000
      },
      "device_mapping": {
        "layer_0": {
          "GPU_0": {
            "expert_id": 0,
            "responsibility": "Expert 0 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 0, sends processed tokens to next layer"
          },
          "GPU_1": {
            "expert_id": 1,
            "responsibility": "Expert 1 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 1, sends processed tokens to next layer"
          },
          "GPU_2": {
            "expert_id": 2,
            "responsibility": "Expert 2 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 2, sends processed tokens to next layer"
          },
          "GPU_3": {
            "expert_id": 3,
            "responsibility": "Expert 3 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 3, sends processed tokens to next layer"
          },
          "GPU_4": {
            "expert_id": 4,
            "responsibility": "Expert 4 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 4, sends processed tokens to next layer"
          },
          "GPU_5": {
            "expert_id": 5,
            "responsibility": "Expert 5 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 5, sends processed tokens to next layer"
          },
          "GPU_6": {
            "expert_id": 6,
            "responsibility": "Expert 6 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 6, sends processed tokens to next layer"
          },
          "GPU_7": {
            "expert_id": 7,
            "responsibility": "Expert 7 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 7, sends processed tokens to next layer"
          },
          "GPU_8": {
            "expert_id": 8,
            "responsibility": "Expert 8 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 8, sends processed tokens to next layer"
          },
          "GPU_9": {
            "expert_id": 9,
            "responsibility": "Expert 9 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 9, sends processed tokens to next layer"
          },
          "GPU_10": {
            "expert_id": 10,
            "responsibility": "Expert 10 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 10, sends processed tokens to next layer"
          },
          "GPU_11": {
            "expert_id": 11,
            "responsibility": "Expert 11 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 11, sends processed tokens to next layer"
          },
          "GPU_12": {
            "expert_id": 12,
            "responsibility": "Expert 12 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 12, sends processed tokens to next layer"
          },
          "GPU_13": {
            "expert_id": 13,
            "responsibility": "Expert 13 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 13, sends processed tokens to next layer"
          },
          "GPU_14": {
            "expert_id": 14,
            "responsibility": "Expert 14 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 14, sends processed tokens to next layer"
          },
          "GPU_15": {
            "expert_id": 15,
            "responsibility": "Expert 15 for Layer 0 - Complete MLP (4096->16384->4096)",
            "memory_allocation": "Complete expert MLP parameters and activations",
            "communication_role": "Receives tokens for Expert 15, sends processed tokens to next layer"
          }
        }
      },
      "communication_optimization": {
        "token_batching": {
          "enabled": true,
          "batch_size": "dynamic_based_on_destination"
        },
        "asynchronous_routing": {
          "enabled": true,
          "implementation": "CUDA_streams_or_NCCL_MPI"
        },
        "topology_aware_placement": {
          "enabled": true,
          "considerations": ["node_to_node_bandwidth", "latency", "gpu_memory_capacity"]
        }
      },
      "performance_metrics": {
        "throughput": {
          "value": 450000,
          "unit": "tokens_per_second"
        },
        "latency": {
          "value": 2.2,
          "unit": "ms_per_token"
        },
        "improvement_factor": {
          "throughput": 3.75,
          "latency": 3.8
        }
      }
    }
  },
  "hardware_requirements": {
    "gpu_specifications": {
      "type": "H100_class",
      "memory_per_gpu": "sufficient_for_single_expert_MLP",
      "minimum_gpus": 16
    },
    "network_requirements": {
      "interconnect": ["NVLink", "InfiniBand"],
      "bandwidth": "high_bandwidth_cross_node",
      "topology": "topology_aware_scheduling_supported"
    }
  }
}
```

**Note**: For the complete 16-layer deployment configuration with all expert-to-GPU mappings, please refer to the separate deployment_configuration_complete.json file, which contains the full device mapping for all layers.