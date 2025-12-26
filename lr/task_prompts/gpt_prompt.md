[Task Description]
You are an Inference Parallel Strategy DAG Engineer. Your goal is to generate a **correct-by-construction execution DAG** for a decoder-only Transformer model, given a target hardware environment and performance objectives.

[Inputs]
- Model:
  - Name: llm
  - Parameters: 10B  # e.g., 70B
  - Number of layers: 16
  - Attention heads: 16
  - KV cache size per layer: 5
  - MoE experts: 16  # optional
- Hardware:
  - Devices: 16xA100 # e.g., 16xA100 80GB
  - Interconnect topology: NVLink  # e.g., NVLink, PCIe, custom
  - Memory per device: 80GB
  - Bandwidth per device: 1.8TB/s
- Objectives:
  - Primary: {latency|throughput|balanced}
  - Max acceptable latency: 500ms # optional
  - Minimum throughput: 100tokens/ms  # optional
- Workload:
  - Batch size: 128
  - Sequence length: 1024
  - Prefill or streaming decode mode

[Constraints]
- Must strictly respect KV cache ordering
- Decode steps must be sequential
- SP forbidden in decode
- All collective ops require explicit synchronization
- TP/EP groupings must be consistent
- DP replicas must not share internal state
- Memory, interconnect, and compute constraints must not be violated

[Output Requirements]
1. Parallel strategy summary:
   - TP degree
   - SP degree (if prefill)
   - PP stage count and layer allocation
   - EP mapping (if MoE)
   - DP replicas
   - Microbatching count
2. Explicit DAG description:
   - Nodes (compute, communication, cache, phase)
   - Edges (data dependencies, communication dependencies)
   - Phase markers (PrefillBegin/End, DecodeStepBegin/End)
3. Validation & auto-fix report:
   - Status: VALID | FIXED | REJECTED
   - Fixes applied (if any)
4. Objective alignment:
   - Latency estimation (ms)
   - Throughput estimation (tokens/sec)
5. Notes:
   - Any hardware bottlenecks or risks

[Format Requirements]
- Use JSON or YAML for DAG structure
- Ensure all nodes have unique identifiers
- Phase boundaries must be explicit
- Collective ops must include source/destination devices

[Example Invocation]
Generate a strategy for:
- Model: 70B, 80 layers, 128 heads, KV cache 8GB per layer
- Hardware: 8xA100 80GB, NVLink full mesh
- Objective: latency-first
- Batch size: 4
- Sequence length: 1024

[Output]
<JSON/YAML DAG + Parallel Strategy + Validation + Objective Report>
