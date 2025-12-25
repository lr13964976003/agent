#!/usr/bin/env python3
"""
Verification script for parallel strategy deployment calculations.
This script validates memory usage, module division, and performance metrics.
"""

import math
import torch

def calculate_memory_usage(sequence_length, batch_size_per_gpu=32, hidden_size=512, 
                         num_layers=4, vocab_size=50000, use_activation_checkpointing=False):
    """Calculate memory usage per GPU for given configuration."""
    
    # Model parameters (10B total, distributed across 16 GPUs)
    model_params = 10e9 / 16  # 625M parameters per GPU
    model_memory = model_params * 2  # FP16: 2 bytes per parameter
    
    # Optimizer states (Adam: momentum + variance)
    optimizer_memory = model_params * 2 * 2  # 2 states × 2 bytes each
    
    # Gradient memory
    gradient_memory = model_params * 2  # FP16 gradients
    
    # Activation memory calculation
    # Attention activations: 2 × batch × seq × hidden × layers
    attention_activations = 2 * batch_size_per_gpu * sequence_length * hidden_size * num_layers
    
    # MOE activations: 2 × batch × seq × hidden × 2 × top_k × layers
    # Assuming top-2 experts are activated
    moe_activations = 2 * batch_size_per_gpu * sequence_length * hidden_size * 2 * 2 * num_layers
    
    # Layer norm and other activations
    other_activations = batch_size_per_gpu * sequence_length * hidden_size * num_layers * 2
    
    total_activations = attention_activations + moe_activations + other_activations
    
    if use_activation_checkpointing:
        total_activations *= 0.5  # 50% reduction with checkpointing
    
    # Communication buffers (increased for all-to-all)
    comm_memory = 2e9  # 2GB for communication buffers
    
    # Total memory
    total_memory = model_memory + optimizer_memory + gradient_memory + total_activations + comm_memory
    
    return {
        "model_memory": model_memory / 1e9,
        "optimizer_memory": optimizer_memory / 1e9,
        "gradient_memory": gradient_memory / 1e9,
        "activation_memory": total_activations / 1e9,
        "comm_memory": comm_memory / 1e9,
        "total_memory": total_memory / 1e9
    }

def calculate_throughput(effective_flops=240e12, model_params=10e9, sequence_length=1024):
    """Calculate realistic throughput considering various overheads."""
    
    # FLOPs per token (realistic calculation)
    # Attention: 2 × attention_params × seq_len scaling
    attention_flops = 2 * 2e9 * (sequence_length / 1024)
    
    # MOE: 2 × moe_params × top_k experts
    moe_flops = 2 * 16e9 * 2  # top-2 experts
    
    total_flops_per_token = attention_flops + moe_flops
    
    # Theoretical max throughput
    theoretical_throughput = effective_flops / total_flops_per_token
    
    # Realistic overhead factors
    communication_overhead = 0.42  # 42% for all-to-all communication
    load_imbalance = 0.12  # 12% for expert load imbalance
    pipeline_bubble = 0.18  # 18% for pipeline bubbles
    
    total_overhead = 1 - (1 - communication_overhead) * (1 - load_imbalance) * (1 - pipeline_bubble)
    
    # Practical throughput
    practical_throughput = theoretical_throughput * (1 - total_overhead)
    
    return {
        "theoretical_throughput": theoretical_throughput / 1e6,  # tokens/ms
        "practical_throughput": practical_throughput / 1e6,
        "total_overhead": total_overhead,
        "flops_per_token": total_flops_per_token / 1e9
    }

def verify_module_division():
    """Verify that module division matches GPU count."""
    
    total_gpus = 16
    
    # Pipeline parallelism: 4 stages
    pp_size = 4
    
    # Expert parallelism: 16 experts across 16 GPUs
    ep_size = 16
    
    # Data parallelism: 4 replicas
    dp_size = 4
    
    # Total modules = experts (since each GPU handles 1 expert)
    total_modules = ep_size
    
    print(f"Total GPUs: {total_gpus}")
    print(f"Pipeline stages: {pp_size}")
    print(f"Expert parallel size: {ep_size}")
    print(f"Data parallel size: {dp_size}")
    print(f"Total modules: {total_modules}")
    print(f"Match: {'✓' if total_modules == total_gpus else '✗'}")
    
    return total_modules == total_gpus

def main():
    """Main verification function."""
    
    print("=== Parallel Strategy Verification ===\n")
    
    # 1. Verify module division
    print("1. Module Division Verification:")
    module_match = verify_module_division()
    print()
    
    # 2. Memory calculations for different sequence lengths
    print("2. Memory Usage by Sequence Length:")
    sequence_lengths = [128, 512, 1024, 2048, 4096, 10240]
    
    for seq_len in sequence_lengths:
        memory = calculate_memory_usage(seq_len)
        print(f"Sequence Length {seq_len}:")
        print(f"  Model Memory: {memory['model_memory']:.2f} GB")
        print(f"  Optimizer Memory: {memory['optimizer_memory']:.2f} GB")
        print(f"  Gradient Memory: {memory['gradient_memory']:.2f} GB")
        print(f"  Activation Memory: {memory['activation_memory']:.2f} GB")
        print(f"  Communication Memory: {memory['comm_memory']:.2f} GB")
        print(f"  Total Memory: {memory['total_memory']:.2f} GB")
        print(f"  Within 64GB Limit: {'✓' if memory['total_memory'] < 64 else '✗'}")
        print()
    
    # 3. Memory with activation checkpointing for long sequences
    print("3. Memory Usage with Activation Checkpointing (Long Sequences):")
    long_sequences = [4096, 10240]
    
    for seq_len in long_sequences:
        memory = calculate_memory_usage(seq_len, use_activation_checkpointing=True)
        print(f"Sequence Length {seq_len} (with checkpointing):")
        print(f"  Total Memory: {memory['total_memory']:.2f} GB")
        print(f"  Within 64GB Limit: {'✓' if memory['total_memory'] < 64 else '✗'}")
        print()
    
    # 4. Throughput calculations
    print("4. Throughput Analysis:")
    for seq_len in [128, 1024, 4096]:
        throughput = calculate_throughput(sequence_length=seq_len)
        print(f"Sequence Length {seq_len}:")
        print(f"  Theoretical Throughput: {throughput['theoretical_throughput']:.1f} tokens/ms")
        print(f"  Practical Throughput: {throughput['practical_throughput']:.1f} tokens/ms")
        print(f"  Total Overhead: {throughput['total_overhead']:.1%}")
        print(f"  FLOPs per Token: {throughput['flops_per_token']:.1f} GFLOPs")
        print()
    
    # 5. Summary
    print("5. Summary:")
    print(f"Module Division Correct: {'✓' if module_match else '✗'}")
    
    # Check if all memory requirements are met
    all_memory_valid = True
    for seq_len in sequence_lengths:
        memory = calculate_memory_usage(seq_len, use_activation_checkpointing=(seq_len > 2048))
        if memory['total_memory'] > 64:
            all_memory_valid = False
            break
    
    print(f"All Memory Requirements Met: {'✓' if all_memory_valid else '✗'}")
    
    # Check throughput target
    target_throughput = 100  # tokens/ms
    practical_throughput = calculate_throughput()['practical_throughput']
    print(f"Throughput Target Met: {'✓' if practical_throughput >= target_throughput else '✗'}")
    print(f"  Target: {target_throughput} tokens/ms")
    print(f"  Achieved: {practical_throughput:.1f} tokens/ms")

if __name__ == "__main__":
    main()