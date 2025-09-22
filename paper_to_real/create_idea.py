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
if os.path.exists(f"/home/wzc/data/file-share/{submission_dir}") is False:
    os.mkdir(f"/home/wzc/data/file-share/{submission_dir}")



@ag.instrument(spankind="workflow")
def main():
    TOPIC = "Create Idea from paper"
    APP_SLUG = "paper_to_real"          # 换成你在 Agenta 的 app_slug
    ENV_SLUG = "production"      # 或 staging/dev

    variant_list = ["create_idea"]
    variant = {
            "create_idea": {
                "slug": "paper_idea",
                "version": 8,
                "inputs": {
                    "paper_path": "/home/wzc/data/papers/2503.16428v1/paper.md",
                    "save_path": f"/home/wzc/data/file-share/{submission_dir}"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    AppendFileTool(),
                    SearchFileTool(),
                    CommandTool()
                    ]
                }
            }
    agents = []
    for k in variant_list:
        prompt = fetch_prompt(APP_SLUG, variant[k]["slug"], variant[k]["version"], variant[k]["inputs"])
        tools = variant[k]["tools"]
        agents.append(build_agent(prompt, tools))
    tasks = []
    descriptions = ["Read Paper, Refine Paper and Create Idea"]
    expected_outputs = ["The file path of concise paper and JSON file"]
    for i in range(len(agents)):
        tasks.append(build_task(descriptions[i], expected_outputs[i], agents[i]))
    
    run_pipeline(agents, tasks)

    #slug_list = []
    #for k,v in variant.items():
    #    slug_list.append({"slug":v["slug"],"version":v["version"]})
    return variant #slug_list



if __name__ == "__main__":
    main()
