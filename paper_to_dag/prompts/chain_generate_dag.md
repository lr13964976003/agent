You are tasked with some tasks. You need achieve the highest score in the tasks by utilizing the provided resources. Please note that this is an engineering task, and you must take the task requirements seriously and complete the task strictly according to the specified requirements.


RESOURCES

---

You have been provided with a concise research paper by the previous agent.  

You have been provided with a supplementary knowledge located at {knowledge_path}.<<<提供知识路径>>>


TASK

---

You are tested to completed the following tasks:

Gain a deep understanding of the methods proposed in the paper. <<<要求理解论文>>> 

Propose a corresponding deployment plan based on the parallel strategy of the paper and the hardware environment. <<<要求按论文的并行策略和硬件环境，提出对应的部署方案>>>

Generate complete model deployment DAGs(directed acyclic graph) according to your deployment plan by calling tools to generate graphviz code, meet the following conditions: <<<要求生成DAG，并遵守以下要求>>>

Request to generate only one DAG graph, consolidating the content together. <<<要求只生成一张DAG图，将内容聚合到一起>>>

Divide the boundary according to GPUs, where each node in the DAG graph represents a GPU. <<<按照GPU划分边界，每个DAG图的节点都是一个GPU>>>

Each node in DAG needs to be detailed down to the operator level.<<<要求dag图的节点详细到算子级别>>>

Use ellipses to represent communication, rectangles for computation, and parallelograms for routing/aggregation.<<<指定DAG图节点形状>>>

Each nodes in DAG must have the attributions: INPUT DIMENSION and OUTPUT DIMENSION. Sample: Input: \[batch\_size=?, seq\_len=?, heads=?, d\_k=?],Output:\[batch\_size=?, seq\_len=?, heads=?, d\_k=?]<<<每个DAG图节点必须注明输入维度和输出维度>>>

Information from different dimensions must be separated by commas.<<<不同维度信息用,隔开>>>

Communication between nodes in DAG needs to be demonstrated. <<<DAG图上的节点间的通信要体现出来>>>

In the DAG diagram, the GPU numbers must be clearly specified; it is not allowed to use abbreviations such as GPU: ALL or GPU: Shared. <<<DAG图中GPU的序号要明确写出来，不能使用 GPU：ALL或者GPU: Shared 省略表示>>>

Nodes in DAG containing multiple operators must be split. <<<包含多个算子的DAG图节点必须拆分>>>

One layer in the model consists of a Multi-Head Attention along with an FFN(Gate and Experts). <<<模型中的一层由多头注意力机制以及前馈神经网络（包括门控和专家模块）组成>>>

The residual add has at least two inputs. Please ensure not to omit its input connections..<<<注意DAG中不能漏残差的边>>>

The aggregation and split of tensor need to be represented by nodes. <<<显示张量的聚合与分割>>>

This will be a task with many steps. Please ensure you have fully understood the structure of the LLM before making any decisions. <<<这是一个多步任务，不要急于求成>>>


NOTE

---

You need to follow the following constraints:

The generated DAG must not contain any cycles.<<<禁止有环>>>

Except for the input node, each node in DAG must have a preceding input node; except for the output node, each node must output to another node. <<<禁止有空悬的节点>>>

Do not make any changes to the original file. <<<禁止修改原始文件>>>

In a batch, there are a total of batch size independent data points.<<<batch中数据是独立的>>>


The gate will select which token needs to be sent among all the experts. This process should be represented with a dashed line.<<<提醒门控是在所有专家中进行选择，要求用虚线表示这个过程>>>

Sometimes, a complete DAG can be very large and contain a lot of similar content. You can first generate a Python file and then execute the Python file to create the DAG file. <<<提醒可以生成python文件来生成dag>>>

By executing Python, you need to generate images and .dot files.<<<图像和dot文件都需要生成>>>



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

Accuracy: We will verify whether your DAG deployment meets all the requirements above.

Result: We will evaluate whether your deployment plan is practical.
































