# Nodes Requiring Modification

## Baseline Tensor Pipeline DAG Issues

### Critical Missing Connections:
The following AllGather nodes are defined but NOT connected in the execution flow:

1. **layer_0_allgather_qkv** - Missing input/output connections
2. **layer_0_allgather_mlp** - Missing input/output connections  
3. **layer_1_allgather_qkv** - Missing input/output connections
4. **layer_1_allgather_mlp** - Missing input/output connections
5. **layer_2_allgather_qkv** - Missing input/output connections
6. **layer_2_allgather_mlp** - Missing input/output connections
7. **layer_3_allgather_qkv** - Missing input/output connections
8. **layer_3_allgather_mlp** - Missing input/output connections
9. **layer_4_allgather_qkv** - Missing input/output connections
10. **layer_4_allgather_mlp** - Missing input/output connections
11. **layer_5_allgather_qkv** - Missing input/output connections
12. **layer_5_allgather_mlp** - Missing input/output connections
13. **layer_6_allgather_qkv** - Missing input/output connections
14. **layer_6_allgather_mlp** - Missing input/output connections
15. **layer_7_allgather_qkv** - Missing input/output connections
16. **layer_7_allgather_mlp** - Missing input/output connections
17. **layer_8_allgather_qkv** - Missing input/output connections
18. **layer_8_allgather_mlp** - Missing input/output connections
19. **layer_9_allgather_qkv** - Missing input/output connections
20. **layer_9_allgather_mlp** - Missing input/output connections
21. **layer_10_allgather_qkv** - Missing input/output connections
22. **layer_10_allgather_mlp** - Missing input/output connections
23. **layer_11_allgather_qkv** - Missing input/output connections
24. **layer_11_allgather_mlp** - Missing input/output connections
25. **layer_12_allgather_qkv** - Missing input/output connections
26. **layer_12_allgather_mlp** - Missing input/output connections
27. **layer_13_allgather_qkv** - Missing input/output connections
28. **layer_13_allgather_mlp** - Missing input/output connections
29. **layer_14_allgather_qkv** - Missing input/output connections
30. **layer_14_allgather_mlp** - Missing input/output connections
31. **layer_15_allgather_qkv** - Missing input/output connections
32. **layer_15_allgather_mlp** - Missing input/output connections

### Required Modifications:
1. **Connect AllGather QKV nodes** between QKV projection and attention operations
2. **Connect AllGather MLP nodes** between MLP FC1 and GELU operations  
3. **Establish proper tensor parallelism flow** for all 16 layers
4. **Remove redundancy** - consider simplifying the massive scale

## Proposed Layer-wise DAG Issues

### Missing Communication Primitives:
The entire DAG lacks tensor parallelism communication:

1. **Missing AllReduce operations** - No attention output reduction
2. **Missing AllGather operations** - No tensor gathering for parallelism
3. **Missing collective communication nodes** - Only point-to-point GPU communication

### Required Modifications:
1. **Add AllReduce nodes** for each layer's attention output (layers 0-15)
2. **Add AllGather nodes** for QKV and MLP operations (layers 0-15)
3. **Implement proper tensor parallelism** alongside pipeline parallelism
4. **Add communication edges** connecting collective operations

## Summary

Both DAGs require significant modifications:
- **Baseline**: Fix disconnected AllGather nodes (32 nodes need connection)
- **Proposed**: Add entire tensor parallelism infrastructure (32+ nodes needed)