
RESOURCES

---

You have been provided with the parallel strategy deployment method file by previous agent. <<<提供部署方案文件>>>

Provided knowledge file about how to evaluate the performance of model is located at {knowledge_path}.

TASK

---

You are tested to completed the following tasks:

Check whether the parallel strategy deployment method is compatible with the current hardware environment and model parameters. <<<检查并行策略是否符合实际>>>

Check whether the performance is optimized while meeting basic performance requirements. <<<检查是否在满足基本性能需求的情况下，性能达到最优>>>

Ensure that the refined version of deployment method file retains sufficient information to generate the directed acyclic graph for the deployment of the experimental model in the paper.<<<提醒要保留足够的信息来生成dag>>>

Is this parallel strategy deployment method incorrect. If incorrect, how to modify.  

Has this parallel strategy deployment plan met the basic performance requirements. If not, how to modify.

Has this parallel strategy deployment plan achieved the optimal outcome under the current environment. If not, how to modify.


NOTE

---

You need to follow the following constraints:

Do not make any changes to the original file.<<<禁止修改源文件>>>

The number of GPUs required for the deployment plan is not simply calculated as EP * TP * PP * DP * SP; instead, there is a complex mapping relationship, and the rules are specified in the knowledge file. <<<提醒GPU总数>>>


SUBMISSION

---

You only need to save the final parallel strategy deployment plan in markdown format at the {save_path} and delete other intermediate result files.

If there are no issues, please say "Congratulation!!" at first and provide the path for submitting the deployment method in markdown format.

---

How we would grade this:

Understand: We will check whether you have read and understood the the deployment method file.

Result: We will check whether your conclusion meets the expected standards.

Performance: We will rigorously evaluate whether the parallel strategy is the optimal solutions under current deployment conditions.


