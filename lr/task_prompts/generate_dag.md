RESOURCES

---

You have been provided with a parallelism strategy deployment method file of LLM located at {method_path}.  

You have been provided with a supplementary knowledge located at {knowledge_path}.<<<提供知识路径>>>


TASK

---

You are tested to completed the following tasks:

Gain a deep understanding of the methods proposed in the parallelism strategy deployment method file. <<<要求理解部署方案>>> 

Generate one complete model deployment DAG(directed acyclic graph) according to parallelism strategy deployment method by calling tools to generate graphviz code, meet the following conditions: <<<要求生成DAG，并遵守以下要求>>>

It is required that the parallel strategy be fully, completely and correctly reflected. <<<要求充分完整正确地体现并行策略>>>

Divide boundaries according to different GPUs, and label each node on the DAG graph with the corresponding GPU. <<<按不同GPU划分边界>>>

Each layer in DAG needs to be detailed down to the operator level.<<<要求dag详细到算子级别>>>

Use ellipses to represent communication, rectangles for computation, and parallelograms for routing/aggregation.<<<指定节点形状>>>

Each nodes in DAG must have the attributions: INPUT DIMENSION and OUTPUT DIMENSION. Sample: Input: \[batch\_size=?, seq\_len=?, heads=?, d\_k=?],Output:\[batch\_size=?, seq\_len=?, heads=?, d\_k=?]<<<每个计算节点必须注明输入维度和输出维度>>>

All communication behaviors between GPUs must be fully and completely reflected. <<<所有GPU间的通信行为要求充分完整体现>>

The aggregation and split of data need to be represented by nodes. <<<显示数据聚合与分割>>>

The gate will select which token needs to be sent among all the experts. This process should be represented with a dashed line.<<<提醒门控是在所有专家中进行选择，要求用虚线表示这个过程>>>


NOTE

---

You need to follow the following constraints:

Do not make any changes to the original file. <<<禁止修改原始文件>>>

Avoid vague expressions like "ALL GPUs"; instead, specify the exact GPUs number. <<<禁止出现ALL GPUs这种模糊的表述，写成具体的GPU>>>

The generated DAG must not contain any cycles.<<<禁止有环>>>

Except for the input node, each node must have a preceding input node; except for the output node, each node must output to another node. <<<禁止有空悬的节点>>>

You can first generate a Python file and then execute the Python file to create the DAG file. <<<提醒可以生成python文件来生成dag>>>

By executing Python, you need to generate images and .dot files.<<<图像和dot文件都需要生成>>>


SUBMISSION

---

The generated DAG should be saved in {save_path}. 

All graphviz codes need to be saved in DOT format.<<<需要提交graphviz代码>>>

All images need to be saved in SVG format. <<<指定图像保存格式为svg>>>

Submit the save paths of all DAGs you generated in JSON format. <<<要求用json格式提交生成dag的路径>>>


<<<用评分标准约束Agent行为>>>

How we would grade this:

Understand: We will check whether you have read and understood ALL the sections of the deployment method file.

Accuracy: We will verify whether your DAG deployment meets the requirements.






