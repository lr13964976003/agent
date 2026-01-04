
RESOURCES

---

You have been provided with a research paper located at {paper_path}. <<<提供论文路径>>>


TASK

---

You are tested to completed the following tasks:

Gain a deep understanding of the methods proposed in the paper. <<<要求理解论文>>>

Condense the content of the paper according to the following requirements in different parts:

Part1: Extract the content related to the hardware deployment environment in the experimental section of the paper, such as the number of GPUs, GPU memory capacity, single-GPU computing power, network bandwidth, etc.

Part2: Extract the parameter information of the models used in the experimental section of the paper, such as the number of model parameters, model architecture (e.g., MLA + MOE, GQA + MLP, etc.), number of attention heads, hidden layer dimensions, number of experts, etc.

Part3: Refine the format of input data for the experimental section of the paper, such as batch size, sequence length, etc.

Part4: Extracting the parallel strategy combinations in the experimental section of the paper may include PP, TP, DP, EP, SP, etc.



NOTE

---

You need to follow the following constraints:

Do not make any changes to the original file.<<<禁止修改源文件>>>

Each section of the refined content of the paper should include specific configuration parameters; vague expressions are not allowed.



SUBMISSION

---

The refined content of the paper should be saved in {save_path}.

You don't need to submit the complete content because it is too large. Instead, you should submit the save paths of the content you generated in Markdown format. <<<提交路径而不是content>>>

How we would grade this:

Understand: We will check whether you have read and understood ALL the contents of the paper.

Keypoints: We will check whether you have retained all the parts metioned before.



