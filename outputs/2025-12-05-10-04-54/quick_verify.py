import json

# Quick verification of the CORRECT strategy
with open('/home/wzc/app/agent/paper_to_dag/../outputs/2025-12-05-10-04-54/parallel_strategy_correct.json', 'r') as f:
    strategy = json.load(f)['parallel_strategy']

hw_config = strategy['hardware_configuration']
parallel_config = strategy['parallel_configuration']

total_gpus = hw_config['total_gpus']
ep_degree = parallel_config['expert_parallelism']['ep_degree']
tp_degree = parallel_config['tensor_parallelism']['tp_degree']
pp_degree = parallel_config['pipeline_parallelism']['pp_degree']
dp_degree = parallel_config['data_parallelism']['dp_degree']

required_gpus = ep_degree * tp_degree * pp_degree * dp_degree

print(f"Available GPUs: {total_gpus}")
print(f"EP: {ep_degree} × TP:{tp_degree} × PP:{pp_degree} × DP:{dp_degree} = {required_gpus}")
print(f"Match: {'✅ YES' if required_gpus == total_gpus else '❌ NO'}")