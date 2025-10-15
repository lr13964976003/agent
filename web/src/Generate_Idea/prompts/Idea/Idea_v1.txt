You are an expert researcher. After carefully reading the research paper, please finish the following tasks and strictly adhere to the constraints.



RESOURCES

---

You have been provided with a research paper located at {paper\_path}. <<<提供论文路径>>>

TASK

---

You are tested to completed the following tasks:

Gain a deep understanding of the methods proposed in the paper. <<<要求理解论文>>>

Retain the original Abstract<<<保留原始摘要>>>

Refine the paper in stages. Phase one: Extract the keypoint of the paper. Phase two: Extract the methodology. Phase three: Extract the experiments. <<<要求按三个阶段提炼论文>>>

Identify gaps or limitations in the paper, suggest feasible improvements or research extensions for each.<<<要求找不足并提出改进建议>>>

Intuitively, the specific changes in the improved time must be represented. You can use matrix multiplication with \[m, k, n] to denote the computation time, which is the time required to multiply matrices of size \[m, k] and \[k, n]. Please use this method to represent the required runtime for the baseline method in the paper, the proposed method in the paper, and the method after your improvements.<<<用m,k,n形式表示运行时间>>>

If there is a communication time, it also needs to be appended.<<<不能遗漏通信时间>>>

Describe all the innovative points of the paper as well as the improvements made, with each point accompanied by a corresponding explanation using JSON format. If there is a change in runtime, it should also be represented. The models used in the article and the experimental parameters should also be included.<<<要求用json格式说明创新点与改善点>>>

The title of the paper needs to be specified in the json.<<<json中需要保留title>>>

At each stage of refining the paper, save the refined version of the paper.<<<要求保存阶段成果>>>



NOTE

---

You need to follow the following constraints:

Do not make any changes to the original file.<<<禁止修改源文件>>>

Complete one stage at a time per step.<<<一次最多完成一个stage>>>

JSON file must be complete, any omission is forbidden.<<<禁止在json中缺省>>>

When refining your paper, you need to add content step by step rather than completing it all at once.<<<精炼论文时要按步骤来>>>

This will be a task with many steps. Please ensure you have fully understood the paper.



SUBMISSION

---

All submission should be saved in {save\_path}.

Three markdown files about refining paper named as keypoints.md, methodology.md, experiments.md respectively.<<<规定命名>>>

One json file about gaps and improvements names as improvements.json.

You don't need to submit the complete content because it is too large. Instead, you should submit the save paths of the content you generated in JSON format. <<<提交路径而不是content>>>

How we would grade this:

Understand: We will check whether you have read and understood ALL the sections of the paper.

Simplify: We will review whether you have simplified the unnecessary parts of your paper.

Keypoints: We will check whether you have retained all the key points in the paper.

Innovation: We will verify whether the innovative points you proposed are feasible.

