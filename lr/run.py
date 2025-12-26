# main.py
import os
import time
import agenta as ag
from agenta.sdk.types import PromptTemplate
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, FileReadTool, FileWriterTool
from langchain_openai import ChatOpenAI
import re
from tools import *
from build_agents import *
from datetime import datetime 
from opentelemetry import trace

# Save results
now = datetime.now()
submission_dir = now.strftime("%Y-%m-%d-%H-%M-%S")
if os.path.exists(f"./output/{submission_dir}") is False:
    os.mkdir(f"./outputs/{submission_dir}")
	
# fetch prompts
def fetch_prompt_local(slug:str, inputs:dict) -> str:
    with open(f"./task_prompts/{slug}.md","r") as f:
        prompt = f.read()
    prompt = re.sub(r'<<<.*?>>>', '', prompt, flags=re.DOTALL)
    prompt = prompt.format(**inputs)
    return prompt


#@ag.instrument(spankind="workflow")
def main():
    MAX_ITER = 2

    variant = {
		    "generate_method": {
                "slug": "generate_method",
                "version": 1,
                "inputs": {
					"task_path": "./inputs/task.md",
					"knowledge_path": "./knowledge/moe_parallelism.md",
                    "save_path": f"./outputs/{submission_dir}"
                    },
                "tools": [
                     FileReadTool(),
					 PythonTool(),
                     CommandTool(),
                     FileWriterTool()
                     ]
                 },
		     "check_method": {
                 "slug": "performance_evaluation",
                 "version": 1,
                 "inputs": {
					 "task_path": "./inputs/task.md",
					 "knowledge_path": "./knowledge/moe_parallelism.md",
                     "save_path": f"./outputs/{submission_dir}"
                     },
                 "tools": [
                     FileReadTool(),
					 PythonTool(),
                     CommandTool(),
                     FileWriterTool()
                     ]
                 },
            "generate_dag": {
                "slug": "generate_dag",
                "version": 1,
                "inputs": {
                    "save_path": f"./outputs/{submission_dir}"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    CommandTool(),
		            PythonTool(),
                    ExtractEdgeFromDAGTool()
                    ]
                },
             "check_dag": {
                 "slug": "check_dag",
                 "version": 1,
                 "inputs": {
                     "save_path": f"./outputs/{submission_dir}"
                     },
                 "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    CommandTool(),
		            PythonTool(),
                    ExtractEdgeFromDAGTool()
                     ]
                 }
            }
    prompts = []
    tools = []
    expected_outputs = ["The path of parallelism strategy deployment method file", "Performance Evaluation and Modify", "The path of graphviz code describing the DAG", "DAG Modify Method"]
    for k in variant.keys():
        prompts.append(fetch_prompt_local(variant[k]["slug"], variant[k]["inputs"]))
        tools.append(variant[k]["tools"])
	# Generate_Method_Agent
    GMA = Researcher("openai/Kimi-K2",tools[0])
    GMT = build_task(prompts[0], expected_outputs[0], GMA)

	# Performance_Evaluation_Agent
    PEA = Engineer("openai/Kimi-K2",tools[1])
    PET = build_task(prompts[1], expected_outputs[1], PEA)

	# Generate_DAG_Agent
    GDA = Engineer("openai/Kimi-K2",tools[2])
    GDT = build_task(prompts[2], expected_outputs[2], PEA)

	# Check_DAG_Agent
    CDA = Engineer("openai/Kimi-K2",tools[3])
    CDT = build_task(prompts[3], expected_outputs[3], CDA)

    # Method_Loop
    method_loop = ReviewLoop(worker=GMA, reviewer=PEA, work_task=GMT, review_task=PET)
    method_result = method_loop.run()

    # DAG_Loop
    dag_loop = ReviewLoop(worker=GDA, reviewer=CDA, work_task=GDT, review_task=CDT, inputs=method_result)
    dag_result = dag_loop.run()
    

if __name__ == "__main__":
    main()


