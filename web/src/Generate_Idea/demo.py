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
import argparse
import json



def fetch_prompt_local(slug:str, version:str, inputs:dict) -> str:
    with open(f"./prompts/{slug}_{version}.md","r") as f:
        prompt = f.read()
    prompt = re.sub(r'<<<.*?>>>', '', prompt, flags=re.DOTALL)
    prompt = prompt.format(**inputs)
    return prompt

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|] ', "_", name)

def download_paper(arxiv_id: str, save_dir: str):
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(search.results())
    
    title = sanitize_filename(result.title)
    save_dir = os.path.join(save_dir, arxiv_id)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    
    pdf_path = os.path.join(save_dir, "paper.pdf")
    result.download_pdf(filename=pdf_path)
    
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    with open(os.path.join(save_dir, "paper.md"), "w") as file:
        file.write(text)

#@ag.instrument(spankind="workflow")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv_id", type=str, help="The id od paper in arxiv")
    parser.add_argument("--prompts_json", type=str, help="prompt version")
    args = parser.parse_args()

    arxiv_id = args.arxiv_id
    prompts_json = json.loads(args.prompts_json)
    

    if os.path.exists(f"./papers/{arxiv_id}/paper.md") is False:
        print(f'begin download {arxiv_id}')
        download_paper(arxiv_id, "./papers")

    output_dir = os.path.join("./generated_docs", args.arxiv_id)
    os.makedirs(output_dir, exist_ok=True)


    variant = {
            "read_paper": {
                "slug": "Idea",
                "version": prompts_json['Idea/Idea'],
                "inputs": {
                    "paper_path": f"./papers/{arxiv_id}/paper.md",
                    "save_path": output_dir
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    AppendFileTool(),
                    CommandTool()
                    ]
                },
            "generate_code": {
                "slug" : "Code",
                "version" : prompts_json['Code/Code'],
                "inputs": {
                    "save_path": output_dir
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    CommandTool(),
                    PythonTool()
                    ]
                },
            "sample_test": {
                "slug": "Profing",
                "version": prompts_json["Profing/Profing"],
                "inputs": {
                    "save_path": output_dir
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
    tasks = []
    expected_outputs = ["The file path of concise paper and new idea", \
                        "The path of Python code of implement methods", \
                            "The path of testing result"]
    i = 0
    for k in variant.keys():
        prompt = fetch_prompt_local(variant[k]["slug"], variant[k]["version"], variant[k]["inputs"])
        tools = variant[k]["tools"]
        agents.append(build_agent(tools))
        tasks.append(build_task(prompt, expected_outputs[i], agents[i]))
        i = i + 1
    
    run_pipeline(agents, tasks)

    slug_list = []
    for k,v in variant.items():
        slug_list.append([v["slug"],v["version"]])
    return slug_list





if __name__ == "__main__":
    #tracer = trace.get_tracer(__name__)
    #with tracer.start_as_current_span("workflow-root") as root:
    main()
