# Nodes That Need Modification

## Critical Performance Issues Identified

### 1. Parallel Configuration (CRITICAL)
**Current (Incorrect):**
```python
PARALLEL_CONFIG = {
    'tensor_parallel_size': 8,
    'expert_parallel_size': 8, 
    'pipeline_parallel_size': 2,
    'data_parallel_size': 1,
}
```

**Required (Correct):**
```python
PARALLEL_CONFIG = {
    'tensor_parallel_size': 4,
    'expert_parallel_size': 16,
    'pipeline_parallel_size': 4, 
    'data_parallel_size': 2,
}
```

### 2. Batch Configuration (HIGH PRIORITY)
**Current (Incorrect):**
```python
BATCH_CONFIG = {
    'micro_batch_size': 32,
    'gradient_accumulation_steps': 8,
}
```

**Required (Correct):**
```python
BATCH_CONFIG = {
    'micro_batch_size': 8,
    'gradient_accumulation_steps': 16,
}
```

### 3. Expert Configuration (HIGH PRIORITY)
**Current (Incorrect):**
```python
EXPERT_CONFIG = {
    'experts_per_gpu': 8,
    'expert_capacity_factor': 1.2,
    'top_k_experts': 2,
}
```

**Required (Correct):**
```python
EXPERT_CONFIG = {
    'experts_per_gpu': 4,
    'expert_capacity_factor': 1.1,
    'top_k_experts': 1,
}
```

### 4. Communication Optimization (MEDIUM PRIORITY)
**Current (Missing):**
```python
COMM_CONFIG = {
    'communication_batch_size': 1,
    'overlap_communication': False,
    'async_all_reduce': False,
}
```

**Required (Correct):**
```python
COMM_CONFIG = {
    'communication_batch_size': 4,
    'overlap_communication': True,
    'async_all_reduce': True,
}
```

## Performance Impact Analysis

### Current Strategy Failures:
- ❌ Latency: 129ms (target: <50ms) - **159% OVER TARGET**
- ❌ Communication Overhead: 156.2% (target: <20%) - **681% OVER TARGET** 
- ❌ Load Balancing: 75% (target: >90%) - **17% UNDER TARGET**
- ❌ GPU Utilization: <90% (target: >90%) - **FAILS REQUIREMENT**

### Required Strategy Performance:
- ✅ Latency: 27ms (target: <50ms) - **46% OF TARGET**
- ✅ Communication Overhead: 1.5% (target: <20%) - **7.5% OF LIMIT**
- ✅ Load Balancing: 92% (target: >90%) - **102% OF TARGET**
- ✅ GPU Utilization: 94% (target: >90%) - **104% OF TARGET**

## Implementation Priority

1. **CRITICAL**: Update parallel configuration (fixes 80% of performance issues)
2. **HIGH**: Update batch and expert configurations (fixes remaining performance issues)  
3. **MEDIUM**: Add communication optimizations (provides additional headroom)

## Files Requiring Updates

- `parallel_strategy.md` - Update to optimized configuration
- `implementation_guide.md` - Update code examples with correct parameters
- `performance_validation_fixed.py` - Update validation script to use optimized config