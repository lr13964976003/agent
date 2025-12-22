# DAG Analysis Report

## Current Deployment (EP64-TP8-PP2-DP2)

### Issues Found:

1. **Attention Block Structure Issue**: 
   - The attention score calculation has incorrect input connections
   - `layer0_attn_v` connects directly to `layer0_attn_out` but should connect to the attention score mechanism
   - Missing proper attention mechanism flow where Q, K should be used for score calculation and V should be used for output computation

2. **Hidden Size Inconsistency**:
   - Embedding layer outputs [batch_size=128, seq_len=1024, hidden_size=512]
   - But output node expects [batch_size=128, seq_len=1024, hidden_size=4096] as input
   - Missing proper upscaling from 512 to 4096 before output

3. **Pipeline Stage Coverage**:
   - PP Stage 0 shows "Layers 0-7" but only Layer 0 is detailed
   - PP Stage 1 shows "Layers 8-15" but only Layer 15 is shown
   - Missing intermediate layers (1-6, 8-14)

4. **Expert Parallelism Issue**:
   - EP64 strategy but only showing experts 0 and 1 on EP0 and EP1
   - Should show more expert distribution across 64 EP groups

## New Deployment (EP32-TP16-PP4-DP4)

### Issues Found:

1. **Same Attention Block Structure Issue**:
   - Identical incorrect attention flow as current deployment
   - `layer0_attn_v` connects directly to `layer0_attn_out` instead of proper attention mechanism

2. **Hidden Size Inconsistency**:
   - Embedding outputs [batch_size=128, seq_len=1024, hidden_size=256]
   - Output expects [batch_size=128, seq_len=1024, hidden_size=4096] as input
   - Missing upscaling layer

3. **Pipeline Stage Coverage**:
   - PP Stage 0: "Layers 0-3" (only Layer 0 shown)
   - PP Stage 1: "Layers 4-7" (no layers shown)
   - PP Stage 2: "Layers 8-11" (no layers shown)
   - PP Stage 3: "Layers 12-15" (only Layer 15 shown)
   - Missing most intermediate layers

4. **Expert Parallelism Issue**:
   - EP32 strategy but showing experts 0-3 only on EP0 and EP1
   - Should show expert distribution across 32 EP groups

5. **Incomplete Pipeline Flow**:
   - The pipeline stages PP1 and PP2 have no actual computation nodes
   - Only pipeline transfers are shown between stages

## Corrective Actions Needed:

1. Fix attention mechanism flow in both DAGs
2. Add proper hidden size transformation before output
3. Complete pipeline stage representations with all layers
4. Show proper expert distribution across all EP groups
5. Add missing intermediate pipeline stage computations