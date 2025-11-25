## **Keypoints Extraction - Phase 1**

### **Core Problem**
Traditional MoE implementations colocate multiple experts per GPU, creating computational bottlenecks and limiting expert-level parallelism as cluster sizes grow.

### **Proposed Solution**
Large-scale cross-node expert parallelism strategy that deploys at most one expert per GPU, pushing Expert Parallelism (EP) to ≥16, maximizing computational parallelism by fully exploiting distributed resources.

### **Key Innovation**
Shifts optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

### **Technical Components**
1. **Expert Placement Strategy**: Single-expert-per-GPU deployment with cross-node distribution
2. **Routing and Load Balancing**: Dynamic token routing with asynchronous batching
3. **Communication Overlap**: Interleaving expert computation with cross-node communication

### **Performance Claims**
Achieves ~3.75× higher throughput and ~3.8× lower latency compared to baseline (TP=8, PP=2) configuration.

### **Experimental Validation**
Tested on 16-layer MoE model with 16 experts per layer, BF16 precision, 128 sequences per batch, 10000 tokens per sequence, 4096 token dimension.

### **Scalability Target**
Optimized for large EP regime (EP ≥ 16) in HPC environments with abundant GPU resources like H100 clusters.