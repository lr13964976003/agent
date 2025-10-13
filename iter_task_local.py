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
if os.path.exists(f"/home/wzc/data/file-share/logs/{submission_dir}") is False:
    os.mkdir(f"/home/wzc/data/file-share/logs/{submission_dir}")

def fetch_prompt_local(slug:str, inputs:dict) -> str:
    with open(f"./prompts/{slug}.md","r") as f:
        prompt = f.read()
    prompt = re.sub(r'<<<.*?>>>', '', prompt, flags=re.DOTALL)
    prompt = prompt.format(**inputs)
    return prompt


#@ag.instrument(spankind="workflow")
def main():
    TOPIC = "Read Paper, Generate DAG and Performance"
    APP_SLUG = "paper_to_real"          # 换成你在 Agenta 的 app_slug
    ENV_SLUG = "production"      # 或 staging/dev

    MAX_ITER = 5

    variant = {
            "check_topic": {
                "slug": "check_topic",
                "version": 10,
                "inputs": {
                    "paper_path": "./papers/EP/paper.md",
                    "score_path": "./knowledges/llm_parallelism_classification_schema.json"
                },
                "tools": [
                FileReadTool()
                ]
            },
            "read_paper": {
                "slug": "chain_read_paper",
                "version": 10,
                "inputs": {
                    "paper_path": "/home/wzc/data/papers/EP/paper.md",
                    "knowledge_path": "/home/wzc/data/knowledges/llm_parallel_strategies.md",
                    "save_path": f"/home/wzc/data/file-share/logs/{submission_dir}"
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
                    "origin_paper_path" : "./papers/EP/paper.md",
                    "plan_path": "./papers/EP/deployment_config.json"
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
                    "save_path": f"/home/wzc/data/file-share/logs/{submission_dir}"
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
                     "save_path": f"/home/wzc/data/file-share/logs/{submission_dir}"
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
                     "save_path": f"/home/wzc/data/file-share/logs/{submission_dir}"
                     },
                 "tools": [
                     ExtractEdgeFromDAGTool(),
                     FileReadTool(),
                     CommandTool(),
                     FileWriterTool()
                     ]
                 },
              "iter_generate_dag": {
                  "slug": "iter_generate_dag",
                  "version": 17,
                  "inputs": {
                      "knowledge_path": "./knowledges/llm_parallel_strategies.md",
                      "save_path": f"/home/wzc/data/file-share/logs/{submission_dir}"
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
    for k in variant.keys():
        prompt = fetch_prompt_local(variant[k]["slug"], variant[k]["inputs"])
        tools = variant[k]["tools"]
        agents.append(build_agent(prompt, tools))
    tasks = []
    descriptions = ["Check whether the paper is relevant to the topic", "Read Paper and Refine Paper", "Check the refine paper", "Read concise Paper and Generate DAG", "Check the DAG", "Compute the performance of DAG", "Optimize the performance of DAG"]
    expected_outputs = ["Check Result", "The file path of concise paper and deployment configuration", "Check Result", "The path of graphviz code describing the DAG", "Check Result", "The performance of DAG", "The path of graphviz code describing the DAG"]
    for i in range(len(agents)):
        tasks.append(build_task(descriptions[i], expected_outputs[i], agents[i]))
    
    check_result = run_pipeline([agents[0]], [tasks[0]])
    if "failed" in check_result.lower():
        return "The paper is not relevant to the topic"

    paper_loop = ReviewLoop(worker=agents[1], reviewer=agents[2], task_description=descriptions[1], expected_output=expected_outputs[1])
    paper_result = paper_loop.run()
    dag_loop = ReviewLoop(worker=agents[3], reviewer=agents[4], task_description=descriptions[3], expected_output=expected_outputs[3], inputs=paper_result)
    dag_result = dag_loop.run()

    description = f"There are the submissions of previous agents: \n\n{paper_result}\n\n{dag_result}"
    perf_task = Task(
        description = description,
        agent = agents[5],
        expected_output = expected_outputs[5]
    )
    init_perf = run_pipeline([agents[5]], [perf_task])

    for i in range(MAX_ITER):
        if i == 0:
            iter_input = f"{paper_result}\n\n{init_perf}"
            iter_loop = ReviewLoop(worker=agents[6], reviewer=agents[4], task_description=description, expected_output=expected_outputs[3], inputs=iter_input)
            iter_result = iter_loop.run()
            description = f"There are the submissions of previous agents: \n\n{iter_result}"
            perf_task = Task(
                description = description,
                agent = agents[5],
                expected_output = expected_outputs[5]
            )
            iter_perf = run_pipeline([agents[5]], [perf_task])
        else:
            iter_input = f"{iter_result}\n\n{iter_perf}"
            iter_loop = ReviewLoop(worker=agents[6], reviewer=agents[4], task_description=description, expected_output=expected_outputs[3], inputs=iter_input)
            iter_result = iter_loop.run()
            description=f"There are the submissions of previous agents: \n\n{iter_result}"
            perf_task = Task(
                description = description,
                agent = agents[5],
                expected_output = expected_outputs[5]
            )
            iter_perf =run_pipeline([agents[5]], [perf_task])
    slug_list = []
    for k,v in variant.items():
        slug_list.append([v["slug"],v["version"]])
    return slug_list





if __name__ == "__main__":
    #tracer = trace.get_tracer(__name__)
    #with tracer.start_as_current_span("workflow-root") as root:
    main()
