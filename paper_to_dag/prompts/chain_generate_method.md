You are tasked with some tasks. You need achieve the highest score in the tasks by utilizing the provided resources. Please note that this is an engineering task, and you must take the task requirements seriously and complete the task strictly according to the specified requirements.



RESOURCES

---

You have been provided with a deployment condition file containing the hardware environment and model parameters. <<<提供包含硬件环境和模型参数的部署条件文件>>>

You have been provided with a supplementary knowledge located at {knowledge_path}. <<<提供知识路径>>>



TASK

---

You are tested to completed the following tasks:

Gain a deep understanding of the deployment condition file . <<<要求理解部署条件>>> 

Request to propose a parallel strategy that optimizes the model's performance under the current hardware environment. <<<要求提出当前硬件环境下该模型最优的并行策略>>>

The performance evaluation metrics of the model are as follows: the smaller the model's latency and the larger its throughput, the better the model's performance. <<<模型的性能评价指标>>>






NOTE

---

You need to follow the following constraints:

After you complete the deployment method, calculate how many parts the module has been divided into and whether it matches the number of GPUs.  <<<要求反思部署方案是否符合实际>>>

Ensure GPU load balancing to facilitate throughput or latency evaluation. <<<确保GPU负载均衡>>>




SUBMISSION

---

The generated method should be saved in {save_path}. 

Submit the save paths of all files you generated in JSON format. <<<要求用json格式提交生成dag的路径>>>

<<<用评分标准约束Agent行为>>>

How we would grade this:

Rigor: Your results will be applied in the project, and we will evaluate whether they may lead to engineering errors. 

Understand: We will check whether you have read and understood ALL the sections of the deployment condition file.

Attitude: We will check whether you have strictly adhered to the restrictions in the Note.

Accuracy: We will verify whether your deployment method meets all the requirements.

Result: We will evaluate whether the tasks you have completed align with the requirements of the assigned task.



