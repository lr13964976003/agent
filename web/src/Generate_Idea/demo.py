# main.py
import os
import time
import agenta as ag
from agenta.sdk.types import PromptTemplate
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, FileReadTool, FileWriterTool
from langchain_openai import ChatOpenAI
import re
from .._my_tools import *
from .._build_agent import *
from ..download_utils import *
from datetime import datetime 
from opentelemetry import trace
import argparse



def fetch_prompt_local(slug:str, inputs:dict) -> str:
    with open(f"./prompts/{slug}.md","r") as f:
        prompt = f.read()
    prompt = re.sub(r'<<<.*?>>>', '', prompt, flags=re.DOTALL)
    prompt = prompt.format(**inputs)
    return prompt

#@ag.instrument(spankind="workflow")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv_id", type=str, help="The id od paper in arxiv")
    args = parser.parse_args()

    arxiv_id = args.arxiv_id
    download_paper(arxiv_id, "../papers")

    if os.path.exists(f"../../generated/{arxiv_id}") is False:
        os.mkdir(f"../../generated/{arxiv_id}")


    variant_list = ["read_paper", "generate_code", "sample_test"]
    variant = {
            "read_paper": {
                "slug": "paper_idea",
                "version": 2,
                "inputs": {
                    "paper_path": f"../papers/{arxiv_id}/paper.md",
                    "save_path": f"../../generated/{arxiv_id}"
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
                "version" : 7,
                "inputs": {
                    "save_path": f"../../generated/{arxiv_id}"
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
                "version": 7,
                "inputs": {
                    "save_path": f"../../generated/{arxiv_id}"
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
        prompt = fetch_prompt_local(variant[k]["slug"], variant[k]["inputs"])
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
