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



@ag.instrument(spankind="workflow")
def main():
    TOPIC = "Read Paper, Generate DAG and Performance"
    APP_SLUG = "paper_to_idea"          # 换成你在 Agenta 的 app_slug
    ENV_SLUG = "production"      # 或 staging/dev

    variant_list = ["read_paper", "generate_code", "sample_test"]
    variant = {
            "read_paper": {
                "slug": "paper_idea",
                "version": 1,
                "inputs": {
                    "paper_path": "/home/wzc/data/papers/2505.14708v1/paper.md",
                    "save_path": f"/home/wzc/data/file-share/logs/{submission_dir}"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    AppendFileTool(),
                    CommandTool()
                    ]
                },
            "generate_code": {
                "slug" : "paper_code",
                "version" : 4,
                "inputs": {
                    "save_path": f"/home/wzc/data/file-share/logs/{submission_dir}"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    CommandTool(),
                    PythonTool()
                    ]
                },
            "sample_test": {
                "slug": "paper_profile",
                "version": 4,
                "inputs": {
                    "save_path": f"/home/wzc/data/file-share/logs/{submission_dir}"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    CommandTool(),
		            PythonTool()
                    ]
                },
            }
    agents = []
    for k in variant_list:
        prompt = fetch_prompt(APP_SLUG, variant[k]["slug"], variant[k]["version"], variant[k]["inputs"])
        tools = variant[k]["tools"]
        agents.append(build_agent(prompt, tools))
    tasks = []
    descriptions = ["Read Paper, Refine Paper and Create Idea", "Implement methods in Python", "Testing the performance"]
    expected_outputs = ["The file path of concise paper and new idea", "The path of Python code of implement methods", "The path of testing result"]
    for i in range(len(agents)):
        tasks.append(build_task(descriptions[i], expected_outputs[i], agents[i]))
    
    run_pipeline(agents, tasks)

    slug_list = []
    for k,v in variant.items():
        slug_list.append([v["slug"],v["version"]])
    return slug_list





if __name__ == "__main__":
    #tracer = trace.get_tracer(__name__)
    #with tracer.start_as_current_span("workflow-root") as root:
    main()
