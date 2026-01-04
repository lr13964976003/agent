
RESOURCES

---

Provide input file located at {task_path} containing hardware environment conditions, model configurations, input data and basic performance requirements. <<<提供包含硬件环境和模型参数以及性能需求的输入文件>>>

Provided knowledge file about how to generate parallel strategy deployment method file is located at {knowledge_path}.


TASK

---

You are tested to completed the following tasks:

Requires a deep understanding of the hardware environment, models, and performance requirements. <<<要求理解输入文件>>> 

The requirement is to maximize the total throughput as much as possible while meeting the TTFT (Time to First Token) criteria. <<<要求在满足基本性能需求的前提下，提出基于当前硬件环境下模型的并行策略>>>

Different parallel strategies can be employed in the prefill phase and the decode phase, as these two phases are separable. <<<PD分离>>>

Request to use as few GPU resources as possible to improve resource utilization. <<<要求充分利用硬件资源>>>

Document requiring the generation of two parallel strategy deployment method file of prefill and decode. <<<要求生成并行策略的部署方案的文件>>>


NOTE

---

You need to follow the following constraints:

After you complete the deployment method, calculate how many parts the module has been divided into and whether it matches the number of GPUs.  <<<要求反思部署方案是否符合实际>>>

Basic performance requirements must be met. <<<要求必须满足基本的性能需求>>>

The number of GPUs required for the deployment method is not simply calculated as EP * TP * PP * DP * SP; instead, there is a complex mapping relationship, and the rules are specified in the knowledge file. <<<提醒GPU总数>>>

Ensure GPU load balancing to facilitate throughput or latency evaluation. <<<确保GPU负载均衡>>>




SUBMISSION

---

The two parallel strategy deployment method files should be saved in JSON format at {save_path}. 

You don't need to submit the complete content because it is too large. Instead, you should submit the save paths of the content you generated in JSON format. <<<提交路径而不是content>>>

<<<用评分标准约束Agent行为>>>

How we would grade this:

Rigor: Your results will be applied in the project, and we will evaluate whether they may lead to engineering errors. 

Understand: We will check whether you have read and understood ALL the sections of the deployment condition file.

Accuracy: We will verify whether your deployment method meets all the requirements.

Performance: We will rigorously evaluate whether the parallel strategies you generate are the optimal solutions under current deployment conditions.


