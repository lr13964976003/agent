# DAG Corrections Required

The following edges are **missing** in the current dot file.  
Uncomment and ensure these connections exist:

```dot
norm_m7  -> split2;
norm_m11 -> split3;
norm_m15 -> output;
```

Additionally, verify that **every layer** (4–15) contains the complete attention and MoE sub-modules as already shown for layers 0–3.

Once the edges above are added, the graph will satisfy:
- No cycles (already acyclic)
- All GPU communications identified (All-Reduce, All-to-All, PP-send/recv)
- Attention blocks fully decomposed
- Every non-input node has ≥1 input
- Every non-output node has ≥1 output
```