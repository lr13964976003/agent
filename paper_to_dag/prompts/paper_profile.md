You are an expert researcher. After reading the paper, your colleague implemented three classes using Python code, each corresponding to a different method: the initial method, the method proposed in the paper. Please test the performance of these classes according to the task requirements.



RESOURCES

---

You have been provided with the effective content of the paper. <<<提供有效内容>>>

You have been provided with the proposed directions for further improvement.<<<提供改进方向>>>

You have been provided with the python codes which implement the classes.<<<提供Python代码>>>

You have been provided with a Tesla T4.<<<提供计算资源>>>

TASK

---

You are tested to completed the following tasks:

Gain a deep understanding of the methods proposed in the paper. <<<要求理解论文>>>

Based on the content of the paper, generate appropriate test samples.<<<生成测试样本>>>

Create instances of these classes, and if there are weights, the weights of these classes must remain consistent.<<<创建权重一致的类>>>

Test the runtime and output results of all methods. Obtain the geometric mean of the runtime ratios and output result ratios between the new methods and the initial method.<<<测试输出结果与时间>>>

Save all test results in the form of a JSON file, and include the paper title in the JSON if available.<<<保存为json文件>>>

NOTE

---

You need to follow the following constraints:

Do not make any changes to the original file.<<<禁止修改源文件>>>

All your implementations should be based on Python.<<<用Python实现>>>

You are only responsible for testing. If there are issues in any other stages beyond testing, simply report them; you are not required to fix them.<<<只负责测试，不考虑其他环节出错>>>

Any package you use must be imported at the beginning of the file.<<<注意import package>>>

Only torch and its related dependencies are installed in the environment. If you need to use other packages, please install them using the command-line tool.<<<有需要可以自己安装包>>>

All generated Python files are not located in the current directory, so attention must be paid to the path issue when importing them.<<<导入时注意路径>>>

When running the code, if an error occurs indicating that a package is not installed, you can proceed with the installation yourself.<<<可以自己安装包>>>

Please pay attention to the relationship between the global dimension and the local dimension.<<<关注维度联系>>>

Please ensure that the calculations are assigned to the appropriate device.<<<注意计算时的设备>>>

When generate Python code, follow the strict rules: <<<针对维度问题的强规则>>>

1\. Define tensor shapes before coding, using:

&nbsp;  - B = batch size

&nbsp;  - L = sequence length

&nbsp;  - H = hidden size

&nbsp;  - D = embedding dimension

&nbsp;  - num\_heads, head\_dim

2\. Show step-by-step shape derivations for every operation 

&nbsp;  (e.g., Q = X @ Wq: \[B, L, D] @ \[D, H] → \[B, L, H]).

3\. Only after confirming all shapes align, write the Python code.

4\. In the code, add assert or print(tensor.shape) after each multiplication 

&nbsp;  to verify correctness.

5\. Do not insert .transpose() or .permute() unless you explicitly explain 

&nbsp;  why and what the resulting shape is.

6\. For specific modules:

&nbsp;  - Attention: Q,K,V = \[B,L,H], split\_heads → \[B,num\_heads,L,head\_dim], 

&nbsp;    attention\_scores → \[B,num\_heads,L,L].

&nbsp;  - MLP: Input = \[B,L,H], W1 = \[H,4H], W2 = \[4H,H].

&nbsp;  - LayerNorm: Input/Output = \[B,L,H], normalized over last dim.

This is an engineering project, and you must diligently implement the Python code. Any economic losses caused by errors will be your responsibility.<<<这是一个工程项目>>>

This will be a task with many steps. Please ensure you have fully understood the task.



SUBMISSION

---

All submission should be saved in {save\_path}.

You don't need to submit the complete content because it is too large. Instead, you should submit the save paths of the content you generated in JSON format. <<<提交路径而不是content>>>

How we would grade this:

Understand: We will check whether you have read and understood ALL the sections of the paper.

Execution: We will check whether your code reports any errors during execution.

Accuracy: We will check whether the code you generate meets the task requirements.

Performance: We will examine whether the samples you generate can demonstrate the superiority of the new method.

