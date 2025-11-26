## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures have emerged as a powerful approach for scaling large language models (LLMs) while maintaining computational efficiency. By activating only a subset of experts per input token, MoE models can achieve higher parameter counts without proportionally increasing the inference or training cost. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization.

Traditional MoE parallelization strategies often assign multiple experts to the same GPU to reduce inter-node communication. While this minimizes network traffic, it also creates computational bottlenecks and limits the degree of true expert parallelism. As model and cluster sizes grow, this trade-off becomes increasingly suboptimal.

In this work, we present a cross-node expert parallelism method that prioritizes distributing experts across nodes such that each GPU hosts at most one expert. By pushing Expert Parallelism (EP) to large number (defined as EP ≥ 16, qualifying as "large EP"), we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## **Background**

### **Mixture-of-Experts in Large-Scale Models**

MoE models are a variant of transformer architectures where the feed-forward network (FFN) layers are replaced by multiple "experts," each trained to specialize in different input patterns. A gating mechanism determines which subset of experts is activated for each token, leading to sparse computation and improved efficiency in large-scale training.

### **Parallelism Strategies for MoE**

Scaling MoE models typically involves a combination of data parallelism (DP), tensor model parallelism (TP), pipeline parallelism (PP), and expert parallelism (EP). Expert parallelism partitions experts across devices so that different GPUs handle different experts. Standard implementations often aim for a moderate EP degree, placing multiple experts per GPU to limit communication.

However, as network interconnects such as NVLink, InfiniBand, and H100-class NVSwitch fabrics advance, the communication cost becomes less dominant compared to the gains achievable from maximizing compute concurrency. This shifts the bottleneck from communication to synchronization and load balancing.

### **Large Expert Parallelism (Large EP)**

In such regimes, distributing experts across as many devices as possible—ideally one per GPU—minimizes resource contention and maximizes expert-level parallel execution. The challenge lies in efficiently coordinating cross-node communication and ensuring balanced routing without overloading the network. Our proposed method addresses this challenge by exploiting large EP setups, carefully aligning expert placement with the cluster topology, and overlapping communication with computation to achieve near-linear scaling in large MoE deployments.

## **Methods**

### **1. Overview**

Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the bottleneck from inter-expert contention to network communication, which can be mitigated through careful scheduling, routing, and overlapping of communication and computation.

The method consists of three key components:

1. **Expert Placement Strategy** – Assigning experts across GPUs and nodes with mathematical treatment for E ≤ G vs E > G scenarios
2. **Routing and Load Balancing** – Ensuring balanced input distribution to experts with token batching details
3. **Communication Overlap and Scheduling** – Minimizing the impact of cross-node data transfers using CUDA streams

### **2. Expert Placement Strategy**

#### **2.1 Single-Expert-Per-GPU Deployment**

In conventional MoE implementations, multiple experts are colocated on a single GPU to reduce cross-node communication. However, this limits the parallelism achievable at the expert level. In contrast, our method deploys at most one expert per GPU:

**Mathematical Formulation:**
- For a MoE layer with E experts and a cluster of G GPUs:
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs to maximize independent expert concurrency

This approach ensures that each expert can process tokens without contention from other experts on the same device, fully utilizing GPU compute units.

#### **2.2 Cross-Node Distribution**

Experts are distributed across nodes to minimize hotspotting on any single node. We use a topology-aware placement strategy that takes into account:
- Node-to-node bandwidth and latency matrices
- GPU memory capacity per node
- Expected token routing patterns

### **3. Routing and Load Balancing**

#### **3.1 Gating Mechanism**
The routing of tokens to experts is governed by a gating network. For each input token, the top-K gating scores determine which experts are activated.

#### **3.2 Token Sharding Across Nodes**
Given cross-node expert placement, tokens destined for experts on different nodes must be transferred efficiently:

1. **Token Batching**: Group tokens by destination expert to reduce the number of network messages
2. **Asynchronous Routing**: Send token batches asynchronously to overlapping expert computation
3. **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to avoid overloading specific experts

### **4. Communication Overlap and Scheduling**

#### **4.1 Overlapping Compute and Communication**
To mitigate the latency of cross-node token transfers, we interleave expert computation and communication:
- While one batch of tokens is being processed on a GPU, the next batch is simultaneously transferred from other nodes
- CUDA streams or asynchronous communication libraries (e.g., NCCL or MPI) are leveraged to ensure that data transfer does not block GPU computation

#### **4.2 Pipeline Scheduling**
In multi-layer MoE networks, the scheduling ensures that:
- Token outputs from the previous MoE layer are immediately routed to the next layer
- Experts in subsequent layers start processing as soon as a partial batch arrives, rather than waiting for the full batch

### **5. Scalability Considerations**

#### **5.1 Large EP Regime**
Our method is optimized for large EP setups, defined as having 16 or more experts per parallel group. In this regime:
- Network bandwidth becomes the primary limiting factor
- The one-expert-per-GPU policy ensures that all GPUs are fully utilized for compute
- Communication costs are masked by the calculation process

#### **5.2 Memory and Model Parallelism Integration**
To handle very large models that cannot fit on a single GPU:
- Each expert can be further partitioned using tensor model parallelism (TP) within its GPU
- Data parallelism (DP) is applied across replicas of the MoE network

## **Experiments**

### **1. Experimental Setup**

We evaluate the proposed large-scale cross-node expert parallelism method in an **inference-only** setting using adequate H100 GPUs. The model and configuration are:

- **Model**: 61-layer Mixture-of-Experts (MoE), each expert is a MLP. The first three layers are dense, followed by MoE
- **Precision**: BF16
- **Batch size**: Variable batch size
- **Sequence Length**: Variable sequence length
- **Token Dimension**: 7,168
- **Dimension of MHA**: 128 heads × 128 dimensions per head = 16,384 total
- **Hidden size of MLP**: 2,048

**Hardware Environment:**
- **GPUs**: Adequate H100 GPU resources
- **Single-card computing power**: 400 TFLOPS
- **MFU utilization**: 60%
- **VRAM Bandwidth**: 1.8 TB/s
- **Bandwidth utilization**: 80%
- **Single-card video memory capacity**: 64GB

### **2. Parallel Deployment Details**

#### **2.1 Proposed Cross-Node Expert Parallelism**
- **GPUs Used**: Adequate GPUs (one GPU per expert per layer)
- **Per-GPU Allocation**: Each GPU hosts exactly one expert per layer
- **Routing**: Input tokens dynamically routed to GPUs holding corresponding experts, token batches asynchronously sent ensuring minimal idle time

#### **2.2 Traditional Baseline Method**
- **Expert Placement**: Multiple experts per GPU (typical 2-4 experts/GPU)
- **EP Degree**: Moderate (4-8 typical range)
- **Strategy**: Minimize cross-node communication through expert colocation
- **Trade-offs**: Reduced communication overhead but increased intra-GPU contention
- **GPU Requirements**: Fewer GPUs needed compared to large EP approach

### **3. Performance Comparison Framework**

#### **3.1 Metrics Evaluated**
- **Throughput**: Tokens processed per second
- **Latency**: Time per token processing
- **Scalability**: Performance vs. number of experts
- **Resource Utilization**: GPU and network efficiency metrics

#### **3.2 Hardware Utilization**
- **Proposed Method**: 60% MFU, 80% bandwidth utilization
- **Traditional Baseline**: Lower MFU due to expert contention, reduced bandwidth requirements

## **Conclusion**

In this work, we proposed a **large-scale cross-node expert parallelism** method for Mixture-of-Experts (MoE) models, designed to **maximize expert-level parallelism** by deploying at most one expert per GPU. Our approach shifts the computational bottleneck from intra-GPU contention to communication, which is effectively mitigated through **asynchronous token routing**, topology-aware expert placement, and overlap of computation with communication.

Our method provides a **scalable blueprint** for future high-performance MoE inference, particularly in environments with abundant GPU resources such as H100 clusters. Future work may explore extending this approach to **training scenarios**, integrating **dynamic expert routing** for adaptive load balancing, and optimizing communication strategies for **even larger models with thousands of experts**.