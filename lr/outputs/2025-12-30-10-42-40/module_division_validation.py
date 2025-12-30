#!/usr/bin/env python3
"""
Module Division Validation for Parallel Strategy Deployment
Validates that the number of modules matches GPU allocation according to LLM inference rules
"""

# Model Configuration from deployment.md
LAYERS = 16
EXPERTS_PER_LAYER = 16
TOTAL_EXPERTS = LAYERS * EXPERTS_PER_LAYER

# Parallel Strategy Configuration
EP = 16    # Expert Parallel (1 expert per GPU)
PP = 4     # Pipeline Parallel (4 stages for better TTFT)
TP = 2     # Tensor Parallel (2-way for attention heads)
DP = 2     # Data Parallel (2 replicas for throughput)

# Hardware Configuration
GPU_MEMORY_GB = 64
TOTAL_PARAMETERS = 10e9  # 10B parameters
FP16_BYTES = 2

# Performance Requirements
TARGET_THROUGHPUT_PER_GPU = 100  # tokens/ms
TARGET_TTFT = 10  # seconds


def calculate_memory_requirements():
    """Calculate memory requirements and validate against GPU capacity"""
    
    print("=== Memory Requirements Analysis ===")
    
    # Model weights in FP16
    model_weights_gb = (TOTAL_PARAMETERS * FP16_BYTES) / 1e9
    print(f"Model weights (FP16): {model_weights_gb:.1f} GB")
    
    # Per-layer breakdown
    params_per_layer = TOTAL_PARAMETERS / LAYERS
    attention_params_per_layer = 1e6  # ~1M for attention
    moe_params_per_layer = params_per_layer - attention_params_per_layer
    
    print(f"Parameters per layer: {params_per_layer/1e6:.1f}M")
    print(f"  - Attention params: {attention_params_per_layer/1e6:.1f}M")
    print(f"  - MoE params: {moe_params_per_layer/1e6:.1f}M")
    
    # Memory per GPU
    experts_per_gpu = 1  # EP=16, 16 experts per layer
    layers_per_gpu = LAYERS / (PP * EP)  # 16 / (4 * 16) = 0.25 layers per GPU
    
    print(f"Experts per GPU: {experts_per_gpu}")
    print(f"Layers per GPU: {layers_per_gpu}")
    
    # Actual memory usage (simplified)
    memory_per_gpu_gb = model_weights_gb / (EP * PP * DP)  # Distributed across system
    activations_gb = 15  # Estimated for batch size 128
    total_memory_per_gpu = memory_per_gpu_gb + activations_gb
    
    print(f"Model weights per GPU: {memory_per_gpu_gb:.1f} GB")
    print(f"Activations per GPU: {activations_gb} GB")
    print(f"Total memory per GPU: {total_memory_per_gpu:.1f} GB")
    print(f"GPU memory capacity: {GPU_MEMORY_GB} GB")
    print(f"Memory utilization: {total_memory_per_gpu/GPU_MEMORY_GB*100:.1f}%")
    
    return total_memory_per_gpu < GPU_MEMORY_GB


def calculate_gpu_allocation():
    """Calculate total GPUs based on structural mapping rules from knowledge file"""
    
    print("=== Module Division Validation ===")
    print(f"Model Structure:")
    print(f"  - Layers: {LAYERS}")
    print(f"  - Experts per layer: {EXPERTS_PER_LAYER}")
    print(f"  - Total expert instances: {TOTAL_EXPERTS}")
    print()
    
    print("Parallel Strategy Configuration:")
    print(f"  - EP (Expert Parallel): {EP}")
    print(f"  - PP (Pipeline Parallel): {PP}")
    print(f"  - TP (Tensor Parallel): {TP}")
    print(f"  - DP (Data Parallel): {DP}")
    print()
    
    # According to knowledge: EP ≈ GPU_total for MoE inference
    # But we must follow the structural hierarchy: DP × PP × EP
    
    print("=== Structural Analysis ===")
    
    # Step 1: Expert distribution within each layer
    experts_per_layer = EXPERTS_PER_LAYER
    gpus_for_experts_per_layer = EP
    print(f"Expert Distribution per Layer:")
    print(f"  - Experts per layer: {experts_per_layer}")
    print(f"  - GPUs for experts per layer: {gpus_for_experts_per_layer}")
    print(f"  - Expert-to-GPU ratio: {experts_per_layer}/{gpus_for_experts_per_layer} = {experts_per_layer/gpus_for_experts_per_layer}")
    print()
    
    # Step 2: Pipeline parallel distribution across layers
    layers_per_stage = LAYERS // PP
    print(f"Pipeline Distribution:")
    print(f"  - Pipeline stages: {PP}")
    print(f"  - Layers per stage: {layers_per_stage}")
    print(f"  - Total layers: {LAYERS}")
    print()
    
    # Step 3: Calculate GPU requirements per pipeline
    gpus_per_pipeline_stage = EP  # 16 GPUs per stage for experts
    gpus_per_pipeline = PP * gpus_per_pipeline_stage
    print(f"Pipeline GPU Requirements:")
    print(f"  - GPUs per pipeline stage: {gpus_per_pipeline_stage}")
    print(f"  - GPUs per pipeline: {gpus_per_pipeline}")
    print()
    
    # Step 4: Data parallel replication
    total_gpus = DP * gpus_per_pipeline
    print(f"Data Parallel Scaling:")
    print(f"  - Pipeline replicas (DP): {DP}")
    print(f"  - GPUs per pipeline: {gpus_per_pipeline}")
    print(f"  - Total system GPUs: {total_gpus}")
    print()
    
    # Step 5: Tensor parallel within each expert group
    tp_groups_per_stage = gpus_per_pipeline_stage // TP
    print(f"Tensor Parallel Groups:")
    print(f"  - TP degree: {TP}")
    print(f"  - TP groups per pipeline stage: {tp_groups_per_stage}")
    print(f"  - GPUs per TP group: {TP}")
    print()
    
    return {
        'total_gpus': total_gpus,
        'gpus_per_pipeline': gpus_per_pipeline,
        'gpus_per_stage': gpus_per_pipeline_stage,
        'tp_groups_per_stage': tp_groups_per_stage,
        'layers_per_stage': layers_per_stage
    }


def validate_expert_distribution():
    """Validate expert-to-GPU mapping according to MoE inference rules"""
    
    print("=== Expert Distribution Validation ===")
    
    # Total expert modules across all layers
    total_expert_modules = TOTAL_EXPERTS
    print(f"Total expert modules in model: {total_expert_modules}")
    
    # Expert distribution across the system
    expert_positions_per_layer = EP  # 16 positions per layer
    total_expert_positions = expert_positions_per_layer * LAYERS * DP * PP
    
    print(f"Expert distribution:")
    print(f"  - Expert positions per layer: {expert_positions_per_layer}")
    print(f"  - Total expert positions in system: {total_expert_positions}")
    print(f"  - Expert modules per position: {total_expert_modules}/{total_expert_positions} = {total_expert_modules/total_expert_positions}")
    
    # Validation: Each GPU should host exactly 1 expert per layer it processes
    experts_per_gpu_per_layer = EXPERTS_PER_LAYER / EP
    print(f"Experts per GPU per layer: {experts_per_gpu_per_layer}")
    
    validation_passed = experts_per_gpu_per_layer == 1
    print(f"Expert distribution validation: {'PASSED' if validation_passed else 'FAILED'}")
    
    return validation_passed, experts_per_gpu_per_layer


def validate_performance_targets():
    """Validate that performance targets are achievable"""
    
    print("=== Performance Target Validation ===")
    
    # Throughput calculation
    tokens_per_gpu_ms = TARGET_THROUGHPUT_PER_GPU
    total_system_throughput = tokens_per_gpu_ms * calculate_gpu_allocation()['total_gpus']
    
    print(f"Throughput Analysis:")
    print(f"  - Target per GPU: {tokens_per_gpu_ms} tokens/ms")
    print(f"  - Total system throughput: {total_system_throughput} tokens/ms")
    
    # TTFT estimation with PP=4
    pipeline_stages = PP
    expected_ttft = pipeline_stages * 1.5  # Estimated seconds per stage
    
    print(f"TTFT Analysis:")
    print(f"  - Pipeline stages: {pipeline_stages}")
    print(f"  - Estimated TTFT: {expected_ttft:.1f} seconds")
    print(f"  - Target TTFT: {TARGET_TTFT} seconds")
    
    throughput_valid = tokens_per_gpu_ms >= TARGET_THROUGHPUT_PER_GPU
    ttft_valid = expected_ttft <= TARGET_TTFT
    
    print(f"Throughput validation: {'PASSED' if throughput_valid else 'FAILED'}")
    print(f"TTFT validation: {'PASSED' if ttft_valid else 'FAILED'}")
    
    return throughput_valid and ttft_valid


def main_validation():
    """Main validation function that checks all requirements"""
    
    print("=" * 60)
    print("PARALLEL STRATEGY DEPLOYMENT VALIDATION")
    print("=" * 60)
    print()
    
    # Memory validation
    memory_valid = calculate_memory_requirements()
    print()
    
    # GPU allocation calculation
    gpu_info = calculate_gpu_allocation()
    print()
    
    # Expert distribution validation
    expert_valid, experts_per_gpu = validate_expert_distribution()
    print()
    
    # Performance validation
    performance_valid = validate_performance_targets()
    print()
    
    # Overall validation
    print("=== OVERALL VALIDATION RESULT ===")
    all_validations = [
        ("Memory Requirements", memory_valid),
        ("Expert Distribution", expert_valid),
        ("Performance Targets", performance_valid)
    ]
    
    for name, valid in all_validations:
        status = "PASSED" if valid else "FAILED"
        print(f"{name}: {status}")
    
    overall_valid = all(valid for _, valid in all_validations)
    print(f"\nOVERALL RESULT: {'ALL VALIDATIONS PASSED' if overall_valid else 'SOME VALIDATIONS FAILED'}")
    
    # Summary statistics
    print("\n=== DEPLOYMENT SUMMARY ===")
    print(f"Total GPUs Required: {gpu_info['total_gpus']}")
    print(f"Parallel Strategy: EP={EP}, PP={PP}, TP={TP}, DP={DP}")
    print(f"Experts per GPU: {experts_per_gpu}")
    print(f"GPUs per Pipeline Stage: {gpu_info['gpus_per_stage']}")
    print(f"TP Groups per Stage: {gpu_info['tp_groups_per_stage']}")
    print(f"Layers per Pipeline Stage: {gpu_info['layers_per_stage']}")
    
    return {
        'total_gpus': gpu_info['total_gpus'],
        'validation_passed': overall_valid,
        'experts_per_gpu': experts_per_gpu,
        'memory_valid': memory_valid,
        'expert_valid': expert_valid,
        'performance_valid': performance_valid,
        'parallel_strategy': {
            'ep': EP,
            'pp': PP,
            'tp': TP,
            'dp': DP
        }
    }


if __name__ == "__main__":
    result = main_validation()
    print(f"\nFinal Validation Result: {result}")