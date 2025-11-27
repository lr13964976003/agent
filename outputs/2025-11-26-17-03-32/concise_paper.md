# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models - Concise Version

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization.

Traditional MoE parallelization assigns multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks and limiting expert-level parallelism. As model and cluster sizes grow, this trade-off becomes suboptimal.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to large numbers (≥16). This shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## **Methods**

### **1. Expert Placement Strategy**

#### **Single-Expert-Per-GPU Deployment**
- Deploy at most one expert per GPU
- For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
- If E > G: replicate experts to maximize independent concurrency while balancing memory
- Each expert processes tokens without contention from other experts on the same device

#### **Cross-Node Distribution**
- Topology-aware placement considering node-to-node bandwidth, latency, GPU memory capacity
- Minimize maximum tokens sent across any single link
- Prevent hotspotting on single nodes

### **2. Routing and Load Balancing**

#### **Token Sharding Across Nodes**
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

#### **Gating Mechanism**
- Top-K selection for expert activation
- Standard MoE compatibility maintained

### **3. Communication Overlap and Scheduling**

#### **Compute-Communication Overlap**
- Interleave expert computation with token transfers
- CUDA streams or NCCL/MPI for asynchronous communication
- Non-blocking data transfers

#### **Pipeline Scheduling**
- Immediate routing between MoE layers
- Start processing partial batches without waiting for full completion
- Fine-grained pipeline to increase throughput

### **4. Scalability Considerations**

#### **Large EP Regime (EP ≥ 16)**
- Network bandwidth becomes primary limiting factor
- Mitigated through topology-aware routing and token batching
- One-expert-per-GPU ensures full GPU utilization

#### **Integration with Other Parallelisms**
- Tensor Parallelism (TP): Partition experts within GPU when needed
- Data Parallelism (DP): Applied across model replicas
- Compatible with models exceeding single-GPU memory

## **Experiments**

### **Experimental Setup**

#### **Model Configuration**
- **Architecture**: 61-layer MoE (first 3 layers dense, remainder MoE)
- **Expert Type**: MLP-based experts
- **Precision**: BF16
- **Token Dimension**: 7168
- **Multi-Head Attention**: 128 heads × 128 dimensions per head
- **MLP Hidden Size**: 2048
- **Variable batch size and sequence length**

#### **Hardware Environment**
- **GPU**: Adequate H100 resources
- **Computing Power**: 400TFlops per GPU
- **Memory**: 64GB per GPU, 1.8TBps bandwidth (80% utilization)
- **Utilization**: 60% MFU (Model FLOPs Utilization)

### **Parallel Deployment**

#### **Proposed Cross-Node Expert Parallelism**
- **GPU Allocation**: One GPU per expert per layer
- **Per-GPU Assignment**: Each GPU hosts exactly one expert per layer
- **Routing**: Dynamic token routing with asynchronous batch sending
- **Execution**: All experts per layer compute in parallel

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. By shifting the bottleneck from intra-GPU contention to communication (effectively mitigated through asynchronous routing and topology-aware placement), we provide a scalable blueprint for high-performance MoE inference in resource-rich environments like H100 clusters.

## **Technical Specifications Summary**

| Parameter | Value |
|-----------|--------|
| Layers | 61 total (3 dense + 58 MoE) |
| Token Dimension | 7168 |
| MHA Heads | 128 |
| Head Dimension | 128 |
| MLP Hidden Size | 2048 |
| Precision | BF16 |
| GPU Type | H100 |
| GPU Memory | 64GB |
| Computing Power | 400TFlops |
| Bandwidth | 1.8TBps |
| MFU Target | 60% |
| EP Degree | ≥16 |
| Experts per GPU | ≤1 |