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
					 "knowledge_path": "../knowledges/llm_parallel_strategies.md",
                     "save_path": f"../outputs/{submission_dir}"
                     },
                 "tools": [
                     ExtractEdgeFromDAGTool(),
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
                     ExtractEdgeFromDAGTool(),
                     FileReadTool(),
					 PythonTool(),
                     CommandTool(),
                     FileWriterTool()
                     ]
                 },
            "generate_dag": {
                "slug": "chain_generate_dag",
                "version": 15,
                "inputs": {
                    "knowledge_path": "../knowledges/llm_parallel_strategies.md",
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
                 "slug": "chain_check_dag",
                 "version": 3,
                 "inputs": {
                     "save_path": f"../outputs/{submission_dir}"
                     },
                 "tools": [
                     FileWriterTool(),
                     CommandTool(),
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
    dag_loop = ReviewLoop(worker=agents[2], reviewer=agents[3], work_task=tasks[2], review_task=tasks[3], inputs=environment_result)
    dag_result = dag_loop.run()
    
    return

    perf_task = tasks[5]
    perf_task.description = tasks[5].description + \
    f"There are the submissions of previous agents: \n\n{dag_result}"
    
    init_perf = run_pipeline([agents[5]], [perf_task])
    
    #with open("temp.txt","r") as f:
    #    iter_input = f.read()

    for i in range(MAX_ITER):
        
        if i == 0:
            iter_input = f"{dag_result}\n\n{init_perf}"
        else:
            iter_input = f"{iter_result}\n\n{iter_perf}"
        
        iter_loop = ReviewLoop(worker=agents[6], reviewer=agents[4], work_task=tasks[6], review_task=tasks[4], inputs=iter_input)
        iter_result = iter_loop.run()
        perf_task = tasks[5]
        perf_task.description = tasks[5].description +\
                        f"There are the submissions of previous agents: \n\n{iter_result}"
        iter_perf =run_pipeline([agents[5]], [perf_task])

    slug_list = []
    for k,v in variant.items():
        slug_list.append([v["slug"],v["version"]])
    return slug_list





if __name__ == "__main__":
    #tracer = trace.get_tracer(__name__)
    #with tracer.start_as_current_span("workflow-root") as root:
    main()

