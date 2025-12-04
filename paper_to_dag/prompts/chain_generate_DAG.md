You are tasked with some tasks. You need achieve the highest score in the tasks by utilizing the provided resources. Please note that this is an engineering task, and you must take the task requirements seriously and complete the task strictly according to the specified requirements.



RESOURCES

---

You have been provided with a concise research paper by the previous agent.  

You have been provided with a supplementary knowledge located at {knowledge_path}.<<<提供知识路径>>>



TASK

---

You are tested to completed the following tasks:

Gain a deep understanding of the methods proposed in the deployment method. <<<要求理解部署方案>>> 

Generate one complete model deployment DAG(directed acyclic graph) according to deployment method by calling tools to generate graphviz code, meet the following conditions: <<<要求生成一个DAG，并遵守以下要求>>>

Card Boundary Division (specify which GPU each node is on) <<<按不同GPU划分边界>>>

Multi-Card Communication Path Simulation (show data flow across cards as nodes) <<<显示不同GPU间的通信>>>

Each layer in DAG needs to be detailed down to the operator level.<<<要求dag详细到算子级别>>>

Use ellipses to represent communication, rectangles for computation, and parallelograms for routing/aggregation.<<<指定节点形状>>>

The aggregation and split of data need to be represented by nodes. <<<显示数据聚合与分割>>>

Each nodes in DAG must have the attributions: INPUT DIMENSION and OUTPUT DIMENSION. Sample: Input: \[batch\_size=?, seq\_len=?, heads=?, d\_k=?],Output:\[batch\_size=?, seq\_len=?, heads=?, d\_k=?]<<<每个计算节点必须注明输入维度和输出维度>>>

Please analyze how the dimensions of the module will change. Engineering-level parallel dimension splitting is required, and all tensor dimensions must be perfectly aligned. In the event of any engineering errors, you will bear all consequences. <<<要求分析维度变化是否正确>>>

Information from different dimensions must be separated by commas.<<<不同维度信息用,隔开>>>

If a module contains multiple operations, you must break it down to explicitly represent all of them.<<<包含多个operator的模块要拆>>>

Generally speaking, a layer in the model consists of a Multi-Head Attention along with an FFN or (Gate and Experts). <<<提供模型一层的组成信息>>>

A complete DAG must include a total input and output.<<<提醒要包含完整输入输出>>>

The residual add has at least two inputs. Please ensure not to omit its input connections..<<<注意dag中不能漏残差的边>>>

Prohibit simplification of modules (such as Expert, etc.) <<<禁止简化模块>>>

Ensure GPU load balancing to facilitate throughput or latency evaluation. <<<确保GPU负载均衡>>>



NOTE

---

You need to follow the following constraints:

Do not make any changes to the original file. <<<禁止修改原始文件>>>

The generated DAG must not contain any cycles.<<<禁止有环>>>

Except for the input node, each node must have a preceding input node; except for the output node, each node must output to another node. <<<禁止有空悬的节点>>>

The gate will select which token needs to be sent among all the experts. This process should be represented with a dashed line.<<<提醒门控是在所有专家中进行选择，要求用虚线表示这个过程>>>

Ensure that each node you create is connected to at least one other node. <<<不允许生成无用节点>>>

Ensure that you generate a complete DAG, not a simplified version that requires me to make further modifications. <<<不允许在dag中省略内容>>>

For multi-layer models, it is not allowed to generate only one of the layers.<<<不允许省略层>>>

Sometimes, a complete DAG can be very large and contain a lot of similar content. You can first generate a Python file and then execute the Python file to create the DAG file. <<<提醒可以生成python文件来生成dag>>>

By executing Python, you need to generate images and .dot files.<<<图像和dot文件都需要生成>>>

This will be a task with many steps. Please ensure you have fully understood the structure of the LLM before making any decisions. <<<这是一个多步任务，不要急于求成>>>



SUBMISSION

---

The generated DAG should be saved in {save_path}. 

All graphviz codes need to be saved in DOT format.<<<需要提交graphviz代码>>>

All images need to be saved in SVG format. <<<指定图像保存格式为svg>>>

Submit the save paths of all DAGs you generated in JSON format. <<<要求用json格式提交生成dag的路径>>>

<<<用评分标准约束Agent行为>>>

How we would grade this:

Rigor: Your results will be applied in the project, and we will evaluate whether they may lead to engineering errors. 

Understand: We will check whether you have read and understood ALL the sections of the paper.

Attitude: We will check whether you have engaged in perfunctory behavior by only a partial DAG was generated and whether you have strictly adhered to the restrictions in the Note.

Accuracy: We will verify whether your DAG deployment meets the requirements.

Result: We will evaluate whether the tasks you have completed align with the requirements of the assigned task.





