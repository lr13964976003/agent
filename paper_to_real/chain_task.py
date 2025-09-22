# main.py
import os
import time
import agenta as ag
from agenta.sdk.types import PromptTemplate
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, FileReadTool, FileWriterTool
from langchain_openai import ChatOpenAI
import re
from _my_tools import PythonTool, AppendFileTool, ExtractEdgeFromDAGTool, SearchFileTool
from _build_agent import fetch_prompt, build_agent, build_task, log_task_output, log_step_output, run_pipeline
from datetime import datetime 


now = datetime.now()
submission_dir = now.strftime("%Y-%m-%d-%H-%M-%S")
if os.path.exists(f"/home/wzc/data/file-share/{submission_dir}") is False:
    os.mkdir(f"/home/wzc/data/file-share/{submission_dir}")

if __name__ == "__main__":
    TOPIC = "Read Paper, Generate DAG and Performance"
    APP_SLUG = "paper_to_real"          # 换成你在 Agenta 的 app_slug
    ENV_SLUG = "production"      # 或 staging/dev

    variant_list = ["read_paper", "check_paper"]#, "generate_dag", "check_dag"] #, "performance"]
    variant = {
            "read_paper": {
                "slug": "chain_read_paper",
                "version": 7,
                "inputs": {
                    "paper_path": "/home/wzc/data/papers/helix/paper.md",
                    "save_path": f"/home/wzc/data/file-share/{submission_dir}"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    AppendFileTool()
                    ]
                },
            "check_paper": {
                "slug" : "chain_check_paper",
                "version" : 6,
                "inputs": {
                    "origin_paper_path" : "/home/wzc/data/papers/helix/paper.md",
                    "plan_path": "/home/wzc/data/papers/helix/deployment_config.json"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    AppendFileTool(),
                    SearchFileTool()
                    ]
                },
            "generate_dag": {
                "slug": "chain_generate_dag",
                "version": 10,
                "inputs": {
                    "knowledge_path": "/home/wzc/data/knowledges/llm_parallel_strategies.md",
                    "save_path": f"/home/wzc/data/file-share/{submission_dir}"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
		            PythonTool()
                    ]
                },
             "check_dag": {
                 "slug": "chain_check_dag",
                 "version": 2,
                 "inputs": {
                     "save_path": f"/home/wzc/data/file-share/{submission_dir}"
                     },
                 "tools": [
                     FileWriterTool(),
                     ExtractEdgeFromDAGTool()
                     ]
                 },
             "performance": {
                 "slug": "chain_performance",
                 "version": 2,
                 "inputs": {
                     "save_path": f"/home/wzc/data/file-share/{submission_dir}"
                     },
                 "tools": [
                     FileReadTool(),
                     FileWriterTool()
                     ]
                 }
            }
    agents = []
    for k in variant_list:
        prompt = fetch_prompt(APP_SLUG, variant[k]["slug"], variant[k]["version"], variant[k]["inputs"])
        tools = variant[k]["tools"]
        agents.append(build_agent(prompt, tools))
    tasks = []
    descriptions = ["Read Paper and Refine Paper", "Check the refine paper", "Read concise Paper and Generate DAG", "Check the DAG", "Compute the performance of DAG"]
    expected_outputs = ["The concise paper", "Check Result", "One graphviz code describing the DAG", "Check Result", "The performance of DAG"]
    for i in range(len(agents)):
        tasks.append(build_task(descriptions[i], expected_outputs[i], agents[i]))
    result = run_pipeline(TOPIC, ENV_SLUG, agents, tasks)

