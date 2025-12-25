You are tasked with some tasks. You need achieve the highest score in the tasks by utilizing the provided resources. Please note that this is an engineering task, and you must take the task requirements seriously and complete the task strictly according to the specified requirements.



RESOURCES

---

You have been provided with a research paper located at {paper_path}. <<<提供论文路径>>>

You have been provided with a supplementary knowledge located at {knowledge_path}.<<<提供知识路径>>>

TASK

---

You are tested to completed the following tasks:

Gain a deep understanding of the methods proposed in the paper. <<<要求理解论文>>>

Condense the paper into a shorter version while retaining all key sections.<<<要求精炼论文>>>

Retain the original Abstract<<<保留原始摘要>>>

Refine the paper in stages. Phase one: Extract the keypoint of the paper. Phase two: Extract the methodology. Phase three: Extract the experiments. <<<要求按三个阶段提炼论文>>>

Describe how to deploy the model according to the paper's methodology using JSON format. Each model mentioned in the paper, including the baseline, should be reflected in the JSON. The JSON should include the parallel strategy used and its corresponding parameters; how all modules are divided under the parallel strategy and the parameters of the modules; how to map modules to devices.<<<要求用json格式说明如何部署模型>>>

At each stage of refining the paper, save the refined version of the paper.<<<要求保存阶段成果>>>



NOTE

---

You need to follow the following constraints:

Do not make any changes to the original file.<<<禁止修改源文件>>>

Complete one stage at a time per step.<<<一次最多完成一个stage>>>

Ensure that the refined version of your paper retains sufficient information to generate the directed acyclic graph for the deployment of the experimental model in the paper.<<<提醒要保留足够的信息来生成dag>>>

Dimension information is very important and needs to be retained.<<<提醒保留维度信息>>>

For model deployment, it is essential to firmly set each parameter, and no ambiguous consideration of multiple possible parameters is allowed.<<<参数设置不允许含糊>>>

In the deployment configuration, it must be specified what is mapped on each device.<<<每个设备的map都要显示表示>>>

Deployment configuration must be complete, any omission is forbidden.<<<禁止在部署方案中缺省>>>

This will be a task with many steps. Please ensure you have fully understood the paper.



SUBMISSION

---

All submission should be saved in {save_path}.

You don't need to submit the complete content because it is too large. Instead, you should submit the save paths of the content you generated in JSON format. <<<提交路径而不是content>>>

How we would grade this:

Understand: We will check whether you have read and understood ALL the sections of the paper.

Simplify: We will review whether you have simplified the unnecessary parts of your paper.

Keypoints: We will check whether you have retained all the key points in the paper.



