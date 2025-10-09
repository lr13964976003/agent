You are an expert researcher. Your colleague has read a paper, extracted the effective content, and proposed directions for further improvement. Please implement the initial method, the method proposed in the paper using Python code.



RESOURCES

---

You have been provided with the effective content of the paper. <<<提供有效内容>>>

You have been provided with the proposed directions for further improvement.<<<提供改进方向>>>

You have been provided with a Tesla T4.<<<提供计算资源>>>

TASK

---

You are tested to completed the following tasks:

Gain a deep understanding of the methods proposed in the paper. <<<要求理解论文>>>

Create at least two classes, each corresponding to the implementation of a method, namely the initialization method, the method proposed in the paper.<<<创建三个对应不同方法的类>>>

Each class must be capable of loading existing weights if necessary.<<<类必须能加载权重>>>

NOTE

---

You need to follow the following constraints:

Do not make any changes to the original file.<<<禁止修改源文件>>>

Any package you use must be imported at the beginning of the file.<<<注意import package>>>

Only torch and its related dependencies are installed in the environment. If you need to use other packages, please install them using the command-line tool.<<<有需要可以自己安装包>>>

You can implement different classes in multiple steps, with each class placed in a separate Python file for easier testing.<<<不同类放不同文件下>>>

After completing the code, you can run a small demo on your own to verify whether the code will produce any errors.<<<用demo验证是否会报错>>>

When running the code, if an error occurs indicating that a package is not installed, you can proceed with the installation yourself.<<<可以自己安装包>>>

Please ensure all operations are performed on CUDA.<<<on cuda>>>

Please pay attention to the relationship between the global dimension and the local dimension.<<<关注维度联系>>>

All generated Python files are not located in the current directory, so attention must be paid to the path issue when importing them.<<<导入时注意路径>>>

Please ensure that the calculations are assigned to the appropriate device.<<<注意计算时的设备>>>

You don't need to test the performance of this method yourself; that part of the work will be handled by your colleague.<<<不需要自己测试>>>

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

This will be a task with many steps. Please ensure you have fully understood the paper.



SUBMISSION

---

All submission should be saved in {save_path}.

You don't need to submit the complete content because it is too large. Instead, you should submit the save paths of the content you generated in JSON format. <<<提交路径而不是content>>>

How we would grade this:

Understand: We will check whether you have read and understood ALL the sections of the paper.

Execution: We will check whether your code reports any errors during execution.

Accuracy: We will verify whether your code successfully replicates the methods described in the paper.

Performance: We will evaluate the efficiency.

