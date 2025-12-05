# DAG Validation Report – moe_ep16_operator_dag.dot

## 1. Parallel strategy (EP16) fully and accurately reflected
**FAIL**  
- Only 4 experts (E00…E03) are ever dispatched or aggregated inside the graph.  
- No expert instances beyond E03 are reachable from the input; E150…E153 are dangling.  
- Dispatcher outputs split into just 2 branches, not 16.  
- Aggregator node (L0_AR) lists inputs from 8 experts, yet 4 of them are unreachable.  
→ The graph does **not** represent 16-way expert parallelism.

## 2. All GPU-to-GPU communications identified
**FAIL**  
- No send/recv, all-reduce, all-to-all, or broadcast nodes/edges are present.  
- Cross-GPU traffic implied by EP16 is absent; the topology is purely local.

## 3. Cycle detection
**PASS**  
- Extractor reports `has_cycle: False`; no back-edges found.

## 4. Attention block decomposed into sub-modules
**PASS**  
- Q, K, V projections and the attention core (L0_ATT) are explicit nodes.

## 5. Every non-input node has ≥1 incoming edge
**FAIL**  
- Nodes `L0_MLP_RES` and `Output` have **only** incoming edges (no predecessors shown).

## 6. Every non-output node has ≥1 outgoing edge
**FAIL**  
- Nodes `Input`, `L0_E150`, `L0_E151`, `L0_E152`, `L0_E153`, and `L15_MLP_RES` have **only** outgoing edges (no successors shown).

---

## Summary
The DAG is **incorrect**. Required corrections:
1. Add 12 more dispatcher branches so that 16 experts are routed.  
2. Connect the 16 expert nodes (E00…E15) to the aggregator; remove unreachable placeholders.  
3. Insert explicit communication nodes (e.g., `all_to_all_dispatch`, `all_to_all_combine`, `all_reduce`) on every GPU-crossing edge.  
4. Provide predecessors for `L0_MLP_RES` and `Output`.  
5. Provide successors for `Input`, `L0_E150`–`L0_E153`, and `L15_MLP_RES` (or delete unused nodes).