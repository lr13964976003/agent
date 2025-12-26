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
from gpt_build_agent import *
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
		    "gpt": {
                "slug": "gpt_prompt",
                "version": 1,
                "inputs": {
					"task_path": "./inputs/task.md",
                    "save_path": f"./outputs/{submission_dir}"
                    },
                "tools": [
                     FileReadTool(),
					 PythonTool(),
                     CommandTool(),
                     FileWriterTool()
                     ]
                 },
            }
    prompts = []
    tools = []
    expected_outputs = "Parallelism strategy deployment method file and Graphviz code describing the DAG"
    for k in variant.keys():
        prompts.append(fetch_prompt_local(variant[k]["slug"], variant[k]["inputs"]))
        tools.append(variant[k]["tools"])
	# GPT
    KNOWLEDGE = [
        "knowledge/04_sequence_parallelism.md",
        "knowledge/04_tensor_parallelism.md",
        "knowledge/04_expert_parallelism.md",
        "knowledge/04_data_parallelism.md",
        "knowledge/04_pipeline_parallelism.md"
    ]
    agent = build_agent("openai/Kimi-K2",tools[0])
    task = build_task(prompts[0], expected_outputs, agent)
    agent.run_task(task)
    return 

    

if __name__ == "__main__":
    main()


