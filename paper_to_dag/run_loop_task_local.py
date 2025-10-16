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
if os.path.exists(f"./output/{submission_dir}") is False:
    os.mkdir(f"./outputs/{submission_dir}")

def fetch_prompt_local(slug:str, inputs:dict) -> str:
    with open(f"./prompts/{slug}.md","r") as f:
        prompt = f.read()
    prompt = re.sub(r'<<<.*?>>>', '', prompt, flags=re.DOTALL)
    prompt = prompt.format(**inputs)
    return prompt


#@ag.instrument(spankind="workflow")
def main():
    variant = {
            "read_paper": {
                "slug": "chain_read_paper",
                "version": 10,
                "inputs": {
                    "paper_path": "./papers/MA/paper.md",
                    "knowledge_path": "./knowledges/llm_parallel_strategies.md",
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
                "slug" : "chain_check_paper",
                "version" : 8,
                "inputs": {
                    "origin_paper_path" : "./papers/MA/paper.md",
                    "plan_path": "./papers/MA/deployment_config.json"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    AppendFileTool(),
                    CommandTool(),
                    SearchFileTool()
                    ]
                },
            "generate_dag": {
                "slug": "chain_generate_dag",
                "version": 15,
                "inputs": {
                    "knowledge_path": "./knowledges/llm_parallel_strategies.md",
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
                 "slug": "chain_check_dag",
                 "version": 3,
                 "inputs": {
                     "save_path": f"./outputs/{submission_dir}"
                     },
                 "tools": [
                     FileWriterTool(),
                     CommandTool(),
                     ExtractEdgeFromDAGTool()
                     ]
                 },
             "performance": {
                 "slug": "chain_performance",
                 "version": 17,
                 "inputs": {
                     "save_path": f"./outputs/{submission_dir}"
                     },
                 "tools": [
                     ExtractEdgeFromDAGTool(),
                     FileReadTool(),
                     CommandTool(),
                     FileWriterTool()
                     ]
                 }
            }
    agents = []
    tasks = []
    i = 0
    expected_outputs = ["The file path of concise paper and deployment configuration", "Check Result", "The path of graphviz code describing the DAG", "Check Result", "The performance of DAG"]
    for k in variant.keys():
        prompt = fetch_prompt_local(variant[k]["slug"], variant[k]["inputs"])
        tools = variant[k]["tools"]
        agents.append(build_agent("openai/Kimi-K2",tools))
        tasks.append(build_task(prompt, expected_outputs[i], agents[i]))
        i = i + 1
    paper_loop = ReviewLoop(worker=agents[0], reviewer=agents[1], work_task=tasks[0], review_task=tasks[1])
    paper_result = paper_loop.run()
    dag_loop = ReviewLoop(worker=agents[2], reviewer=agents[3], work_task=tasks[2], review_task=tasks[3], inputs=paper_result)
    dag_result = dag_loop.run()
    
    perf_task = tasks[4]
    perf_task.description = tasks[4].description + \
    f"There are the submissions of previous agents: \n\n{dag_result}"
    
    perf_result = run_pipeline([agents[4]], [perf_task])

    slug_list = []
    for k,v in variant.items():
        slug_list.append([v["slug"],v["version"]])
    return slug_list



if __name__ == "__main__":
    #tracer = trace.get_tracer(__name__)
    #with tracer.start_as_current_span("workflow-root") as root:
    main()
