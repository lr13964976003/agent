#!/usr/bin/env python3

# Simple debug version
class DebugDeployment:
    def __init__(self):
        self.num_layers = 16
        self.num_experts_per_layer = 16
        self.token_dim = 512
        self.mha_heads = 16
        self.head_dim = 32
        self.moe_hidden = 1024
        self.bytes_per_param = 2
        self.single_gpu_memory = 64
        
    def calculate_memory_requirements(self):
        # Attention weights per layer
        attention_weights = 4 * self.token_dim * self.token_dim
        
        # MoE weights per layer (16 experts)
        expert_weights = self.num_experts_per_layer * 2 * self.token_dim * self.moe_hidden
        
        # Total weights per layer
        layer_weights = attention_weights + expert_weights
        
        # Total model weights
        total_weights = self.num_layers * layer_weights
        
        # Memory in GB
        model_memory_gb = (total_weights * self.bytes_per_param) / (1024**3)
        
        print(f"Attention weights per layer: {attention_weights}")
        print(f"Expert weights per layer: {expert_weights}")
        print(f"Layer weights: {layer_weights}")
        print(f"Total weights: {total_weights}")
        print(f"Model memory: {model_memory_gb:.2f} GB")
        
        return {
            "model_memory_gb": model_memory_gb,
            "layer_weights": layer_weights
        }
    
    def determine_parallel_strategy(self):
        memory_req = self.calculate_memory_requirements()
        
        # Step 1: Expert Parallel (EP)
        ep_degree = self.num_experts_per_layer
        print(f"EP degree: {ep_degree}")
        
        # Step 2: Pipeline Parallel (PP)
        layer_memory_gb = (memory_req["layer_weights"] * self.bytes_per_param) / (1024**3)
        print(f"Layer memory: {layer_memory_gb:.2f} GB")
        
        layers_per_gpu = max(1, int(self.single_gpu_memory * 0.8 / layer_memory_gb))
        print(f"Layers per GPU: {layers_per_gpu}")
        
        pp_degree = max(1, self.num_layers // layers_per_gpu)
        print(f"PP degree: {pp_degree}")
        
        # Step 3: Tensor Parallel (TP)
        tp_degree = 4
        
        # Step 4: Data Parallel (DP)
        dp_degree = 1
        
        total_gpus = ep_degree * pp_degree * tp_degree * dp_degree
        print(f"Total GPUs: {total_gpus}")
        
        return {
            "ep_degree": ep_degree,
            "pp_degree": pp_degree,
            "tp_degree": tp_degree,
            "dp_degree": dp_degree,
            "total_gpus": total_gpus,
            "layers_per_gpu": layers_per_gpu
        }

if __name__ == "__main__":
    deployment = DebugDeployment()
    strategy = deployment.determine_parallel_strategy()
    print(f"Strategy: {strategy}")