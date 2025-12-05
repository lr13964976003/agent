#!/usr/bin/env python3

import math

# Verify the calculations in the deployment method
print("=== Verification of Deployment Method Calculations ===\n")

# Model parameters
total_parameters = 30e9  # 30B parameters
layers = 16
experts_per_layer = 64
bytes_per_param = 2  # FP16
vram_per_gpu = 64e9  # 64GB

# Parallelism degrees
ep_degree = 64
tp_degree = 2
pp_degree = 8

print(f"Model Configuration:")
print(f"- Total parameters: {total_parameters/1e9:.1f}B")
print(f"- Layers: {layers}")
print(f"- Experts per layer: {experts_per_layer}")
print(f"- Total experts: {layers * experts_per_layer}")

print(f"\nParallel Strategy:")
print(f"- EP degree: {ep_degree}")
print(f"- TP degree: {tp_degree}")
print(f"- PP degree: {pp_degree}")
print(f"- Total GPUs: {ep_degree * tp_degree * pp_degree}")

# Memory calculations
params_per_expert = total_parameters / (layers * experts_per_layer)
memory_per_expert = params_per_expert * bytes_per_param
memory_per_expert_with_optimizer = memory_per_expert * 4  # optimizer states

print(f"\nMemory Analysis:")
print(f"- Parameters per expert: {params_per_expert/1e6:.1f}M")
print(f"- Memory per expert: {memory_per_expert/1e6:.1f}MB")
print(f"- Memory per expert with optimizer: {memory_per_expert_with_optimizer/1e6:.1f}MB")

# Total memory per GPU
total_memory_per_gpu = memory_per_expert_with_optimizer * layers
print(f"- Total memory per GPU: {total_memory_per_gpu/1e9:.2f}GB")
print(f"- VRAM utilization: {(total_memory_per_gpu/vram_per_gpu)*100:.2f}%")

# Performance metrics
expected_latency = 50e-3  # 50ms
expected_throughput = 2.5e6  # 2.5M tokens/second
gpu_utilization = 0.85
mfu_achievement = 0.55

print(f"\nPerformance Metrics:")
print(f"- Expected latency: {expected_latency*1000:.0f}ms")
print(f"- Expected throughput: {expected_throughput/1e6:.1f}M tokens/second")
print(f"- GPU utilization: {gpu_utilization*100:.0f}%")
print(f"- MFU achievement: {mfu_achievement*100:.0f}%")

# Check for potential issues
print(f"\n=== Potential Issues Check ===")

# Check memory efficiency
memory_efficiency = (total_memory_per_gpu/vram_per_gpu)*100
if memory_efficiency < 10:
    print(f"⚠️  WARNING: Very low memory utilization ({memory_efficiency:.2f}%) - consider increasing batch size or model parallelism")
elif memory_efficiency > 90:
    print(f"⚠️  WARNING: High memory utilization ({memory_efficiency:.2f}%) - risk of OOM")
else:
    print(f"✅ Memory utilization is optimal ({memory_efficiency:.2f}%)")

# Check GPU count alignment
total_gpus = ep_degree * tp_degree * pp_degree
total_expert_instances = layers * experts_per_layer
experts_per_gpu = total_expert_instances / (total_gpus / tp_degree)  # TP pairs share experts

print(f"✅ Total GPUs: {total_gpus}")
print(f"✅ Expert instances per GPU: {experts_per_gpu:.1f}")

# Check if calculations match the deployment method
deployment_params_per_expert = 29.3e6
deployment_memory_per_gpu = 3.75e9

print(f"\n=== Comparison with Deployment Method ===")
print(f"Parameters per expert:")
print(f"  Calculated: {params_per_expert/1e6:.1f}M")
print(f"  Deployment: {deployment_params_per_expert/1e6:.1f}M")
print(f"  Difference: {abs(params_per_expert - deployment_params_per_expert)/1e6:.1f}M")

print(f"\nMemory per GPU:")
print(f"  Calculated: {total_memory_per_gpu/1e9:.2f}GB")
print(f"  Deployment: {deployment_memory_per_gpu/1e9:.2f}GB")
print(f"  Difference: {abs(total_memory_per_gpu - deployment_memory_per_gpu)/1e9:.2f}GB")

# Final verdict
if abs(params_per_expert - deployment_params_per_expert) < 0.5e6 and abs(total_memory_per_gpu - deployment_memory_per_gpu) < 0.1e9:
    print(f"\n✅ Mathematical calculations are accurate!")
else:
    print(f"\n❌ Mathematical calculations have significant discrepancies!")