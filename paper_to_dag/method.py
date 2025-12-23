# main.py
import os
import time
import agenta as ag
from agenta.sdk.types import PromptTemplate
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, FileReadTool, FileWriterTool
from langchain_openai import ChatOpenAI
import re
from _my_tools import *
from _build_agent import *
from datetime import datetime 
from opentelemetry import trace


now = datetime.now()
submission_dir = now.strftime("%Y-%m-%d-%H-%M-%S")
if os.path.exists(f"../output/{submission_dir}") is False:
    os.mkdir(f"../outputs/{submission_dir}")

def fetch_prompt_local(slug:str, inputs:dict) -> str:
    with open(f"./prompts/{slug}.md","r") as f:
        prompt = f.read()
    prompt = re.sub(r'<<<.*?>>>', '', prompt, flags=re.DOTALL)
    prompt = prompt.format(**inputs)
    return prompt


#@ag.instrument(spankind="workflow")
def main():
    MAX_ITER = 2

    variant = {
		    "generate_method": {
                 "slug": "chain_generate_method",
                 "version": 1,
                 "inputs": {
					 "environment_path": "../environment/EP/deployment.md",
					 "knowledge_path": "../knowledges/llm.md",
                     "save_path": f"../outputs/{submission_dir}"
                     },
                 "tools": [
                     FileReadTool(),
					 PythonTool(),
                     CommandTool(),
                     FileWriterTool()
                     ]
                 },
		     "check_method": {
                 "slug": "chain_check_method",
                 "version": 1,
                 "inputs": {
                     "save_path": f"../outputs/{submission_dir}"
                     },
                 "tools": [
                     FileReadTool(),
					 PythonTool(),
                     CommandTool(),
                     FileWriterTool()
                     ]
                 },
            "generate_dag": {
                "slug": "chain_generate_DAG",
                "version": 1,
                "inputs": {
					"knowledge_path": "../knowledges/llm.md",
                    "save_path": f"../outputs/{submission_dir}"
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
                 "slug": "chain_check_DAG",
                 "version": 1,
                 "inputs": {
                     "save_path": f"../outputs/{submission_dir}"
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
    agents = []
    tasks = []
    i = 0
    expected_outputs = ["The file path of the deployment method", "Check Result", "The path of graphviz code describing the DAG", "Check Result"]
    for k in variant.keys():
        prompt = fetch_prompt_local(variant[k]["slug"], variant[k]["inputs"])
        tools = variant[k]["tools"]
        agents.append(build_agent("openai/Kimi-K2",tools))
        tasks.append(build_task(prompt, expected_outputs[i], agents[i]))
        i = i + 1
    
    # check_result = run_pipeline([agents[0]], [tasks[0]])
    #if "failed" in check_result.lower():
    #    return "The paper is not relevant to the topic"

    method_loop = ReviewLoop(worker=agents[0], reviewer=agents[1], work_task=tasks[0], review_task=tasks[1])
    method_result = method_loop.run()
    dag_loop = ReviewLoop(worker=agents[2], reviewer=agents[3], work_task=tasks[2], review_task=tasks[3], inputs=method_result)
    dag_result = dag_loop.run()
    
    return

if __name__ == "__main__":
    main()

