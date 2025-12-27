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
		    "read_paper": {
                "slug": "read_paper",
                "inputs": {
                    "paper_path": "./papers/EP/paper.md",
                    "knowledge_path": "./knowledge/moe_parallelism.md",
                    "save_path": f"./outputs/{submission_dir}"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    AppendFileTool(),
                    CommandTool()
                    ]
                },
            "check_paper": {
                "slug" : "check_paper",
                "inputs": {
                    "origin_paper_path" : "./papers/EP/paper.md",
					"save_path": f"./outputs/{submission_dir}"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    AppendFileTool(),
                    CommandTool(),
                    SearchFileTool()
                    ]
                },
		    "generate_method": {
                "slug": "generate_method",
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
    expected_outputs = ["The path of refined paper", "check result",
						"The path of parallelism strategy deployment method file", "Performance Evaluation and Modify", 
						"The path of graphviz code describing the DAG", "DAG Modify Method"]
	
    for k in variant.keys():
        prompts.append(fetch_prompt_local(variant[k]["slug"], variant[k]["inputs"]))
        tools.append(variant[k]["tools"])
	# Read_Paper_Agent
    RPA = Researcher("openai/Kimi-K2",tools[0])
    RPT = build_task(prompts[0], expected_outputs[0], RPA)
	
	# Check_Paper_Agent
    CPA = Researcher("openai/Kimi-K2",tools[1])
    CPT = build_task(prompts[1], expected_outputs[1], CPA)
	
	# Generate_Method_Agent
    GMA = Researcher("openai/Kimi-K2",tools[2])
    GMT = build_task(prompts[2], expected_outputs[2], GMA)

	# Performance_Evaluation_Agent
    PEA = Engineer("openai/Kimi-K2",tools[3])
    PET = build_task(prompts[3], expected_outputs[3], PEA)

	# Generate_DAG_Agent
    GDA = Engineer("openai/Kimi-K2",tools[4])
    GDT = build_task(prompts[4], expected_outputs[4], PEA)

	# Check_DAG_Agent
    CDA = Engineer("openai/Kimi-K2",tools[5])
    CDT = build_task(prompts[5], expected_outputs[5], CDA)

	# Paper_Loop
    paper_loop = ReviewLoop(worker=RPA, reviewer=CPA, work_task=RPT, review_task=CPT)
    paper_result = paper_loop.run()

    return

    # Method_Loop
    method_loop = ReviewLoop(worker=GMA, reviewer=PEA, work_task=GMT, review_task=PET, inputs=paper_result)
    method_result = method_loop.run()

    # DAG_Loop
    dag_loop = ReviewLoop(worker=GDA, reviewer=CDA, work_task=GDT, review_task=CDT, inputs=method_result)
    dag_result = dag_loop.run()
    

if __name__ == "__main__":
    main()


