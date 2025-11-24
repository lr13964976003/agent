# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

---

## **1. Introduction**

Mixture-of-Experts (MoE) architectures scale language models efficiently by activating only a subset of experts per token. However, traditional strategies colocate multiple experts per GPU, creating computational bottlenecks that worsen with scale. We present a cross-node expert parallelism method that distributes at most one expert per GPU, pushing EP ≥ 16 to maximize concurrent computation and leverage modern HPC networking capabilities.

## **2. Background**

### **2.1 MoE Architecture**
- **Expert Structure**: MLP layers replacing standard FFN in transformers
- **Gating Mechanism**: Top-K selection per token for sparse computation
- **Parallelism Types**: Data (DP), Tensor (TP), Pipeline (PP), Expert (EP)

### **2.2 Large Expert Parallelism**
- **Definition**: EP ≥ 16 qualifies as "large EP"
- **Key Insight**: Shift bottleneck from intra-GPU contention to network communication
- **Requirement**: High-bandwidth, low-latency interconnects (NVLink, InfiniBand)

## **3. Methodology**

### **3.1 Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Ensures maximal expert-level parallelism
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, memory, routing patterns
- **Memory Balancing**: Even distribution across all GPUs

### **3.2 Routing and Load Balancing**
- **Token Sharding**: Group tokens by destination expert
- **Asynchronous Routing**: Non-blocking token transfer using CUDA streams
- **Dynamic Load Balancing**: Real-time monitoring and gating probability adjustment

### **3.3 Communication Overlap**
- **Compute-Communication Overlap**: Double buffering with concurrent streams
- **Pipeline Scheduling**: Partial batch processing across layers
- **Network Mitigation**: Batching and topology-aware routing

## **4. Experimental Setup**

### **4.1 Model Configuration**
- **Architecture**: 16-layer MoE, 16 experts per layer
- **Dimensions**: Token=4096, MLP hidden=16384, MHA=32×128
- **Precision**: BF16
- **Batch**: 128 sequences × 10,000 tokens

### **4.2 Hardware**
- **GPUs**: 16 H100 with NVLink/InfiniBand
- **Network**: High-bandwidth, low-latency interconnects

### **4.3 Deployment Configurations**

#### **Baseline: TP=8, PP=2**
- **Parallelism**: 8-way tensor, 2-stage pipeline
- **Deployment**: 8 experts per GPU + tensor shards
- **Contention**: Shared compute resources among experts

#### **Proposed: EP=16**
- **Parallelism**: 16-way expert parallelism
- **Deployment**: 1 expert per GPU per layer
- **Isolation**: Dedicated GPU compute per expert

## **5. Results**

| Method | GPUs | Deployment | TPS | TPOT | Improvement |
|--------|------|------------|-----|------|-------------|
| Baseline | 16 | 8 experts/GPU + TP | 120k | 8.3ms | - |
| Proposed | 16 | 1 expert/GPU | 450k | 2.2ms | 3.75× |

**Key Benefits**:
- **3.75× higher throughput** through maximal expert parallelism
- **3.8× lower latency** via dedicated GPU resources
- **Near-linear scaling** for EP ≥ 16
- **Effective communication overlap** mitigating network costs

## **6. Conclusion**

Our cross-node expert parallelism method maximizes compute concurrency by deploying at most one expert per GPU, achieving significant performance improvements in large-scale MoE deployments. This approach provides a scalable blueprint for future HPC MoE systems, particularly effective when EP ≥ 16 and high-bandwidth interconnects are available.

---

## **Technical Specifications Summary**

### **Model Dimensions**
- Layers: 16
- Experts/layer: 16
- Token dimension: 4096
- MLP hidden: 16384
- MHA: 32 heads × 128 dim
- Precision: BF16

### **Deployment Parameters**
- **Baseline**: TP=8, PP=2, 16 GPUs, 8 experts/GPU
- **Proposed**: EP=16, 16 GPUs, 1 expert/GPU
- **Batch**: 128 × 10k tokens
- **Performance**: 450k TPS vs 120k TPS baseline

### **Memory Requirements**
- **Per expert**: ~134M parameters
- **Per GPU (proposed)**: 16 × 134M = ~2.14B parameters
- **Total memory/GPU**: ~8.1GB (including buffers)

### **Communication**
- **Type**: All-to-all with async overlap
- **Bandwidth**: NVLink/InfiniBand
- **Protocol**: NCCL/MPI
- **Optimization**: Token batching, topology-aware routing