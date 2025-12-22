#!/usr/bin/env python3

# Verify the parallel strategy calculations

# Given parameters
TOTAL_GPUS = 1024
TOTAL_EXPERTS = 64
TOTAL_LAYERS = 16
TOTAL_PARAMS = 30_000_000_000  # 30B
DENSE_PARAMS = 4_000_000_000   # 4B
MOE_PARAMS = 26_000_000_000    # 26B
VRAM_PER_GPU = 64_000_000_000  # 64GB in bytes
FP16_BYTES = 2
BATCH_SIZE = 128

# Proposed strategy
EP = 16  # Expert Parallelism
TP = 4   # Tensor Parallelism  
PP = 4   # Pipeline Parallelism
DP = 4   # Data Parallelism

print("=== Parallel Strategy Verification ===")
print()

# Check 1: Total GPU count
print("1. GPU Count Verification:")
total_gpus_used = EP * TP * PP * DP
print(f"   EP{EP} × TP{TP} × PP{PP} × DP{DP} = {total_gpus_used} GPUs")
print(f"   Required: {TOTAL_GPUS} GPUs")
print(f"   Result: {'✓ PASS' if total_gpus_used == TOTAL_GPUS else '✗ FAIL'}")
print()

# Check 2: Expert distribution
print("2. Expert Distribution:")
expert_per_gpu = TOTAL_EXPERTS / EP
print(f"   Total experts: {TOTAL_EXPERTS}")
print(f"   EP{EP}: {expert_per_gpu} experts per GPU")
print(f"   Result: {'✓ PASS' if expert_per_gpu.is_integer() else '✗ FAIL'}")
print()

# Check 3: Layer distribution
print("3. Layer Distribution:")
layers_per_stage = TOTAL_LAYERS / PP
print(f"   Total layers: {TOTAL_LAYERS}")
print(f"   PP{PP}: {layers_per_stage} layers per stage")
print(f"   Result: {'✓ PASS' if layers_per_stage.is_integer() else '✗ FAIL'}")
print()

# Check 4: Memory usage
print("4. Memory Usage Analysis:")
total_param_memory = TOTAL_PARAMS * FP16_BYTES
print(f"   Total parameter memory: {total_param_memory:,} bytes ({total_param_memory/1**9:.1f} GB)")

params_per_gpu = TOTAL_PARAMS / (EP * TP * PP * DP)
memory_per_gpu = params_per_gpu * FP16_BYTES
print(f"   Parameters per GPU: {params_per_gpu:,.0f}")
print(f"   Memory per GPU: {memory_per_gpu:,} bytes ({memory_per_gpu/10**6:.1f} MB)")
print(f"   VRAM per GPU: {VRAM_PER_GPU:,} bytes ({VRAM_PER_GPU/10**9:.1f} GB)")
print(f"   Memory utilization: {memory_per_gpu/VRAM_PER_GPU*100:.3f}%")
print(f"   Result: {'✓ PASS - Sufficient memory' if memory_per_gpu < VRAM_PER_GPU else '✗ FAIL - Insufficient memory'}")
print()

# Check 5: Batch distribution
print("5. Batch Distribution:")
batch_per_dp = BATCH_SIZE / DP
print(f"   Total batch size: {BATCH_SIZE}")
print(f"   DP{DP}: {batch_per_dp} sequences per DP replica")
print(f"   Result: {'✓ PASS' if batch_per_dp.is_integer() else '✗ FAIL'}")
print()

# Overall assessment
print("=== Overall Assessment ===")
all_checks = [
    total_gpus_used == TOTAL_GPUS,
    expert_per_gpu.is_integer(),
    layers_per_stage.is_integer(),
    memory_per_gpu < VRAM_PER_GPU,
    batch_per_dp.is_integer()
]

if all(all_checks):
    print("✓ ALL CHECKS PASSED - Strategy is valid!")
else:
    print("✗ SOME CHECKS FAILED - Strategy needs revision!")
    failed_checks = [i+1 for i, check in enumerate(all_checks) if not check]
    print(f"Failed checks: {failed_checks}")