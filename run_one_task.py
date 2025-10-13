# main.py
import os
import time
import agenta as ag
from agenta.sdk.types import PromptTemplate
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, FileReadTool, FileWriterTool
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from _build_agent import *
import re
from _my_tools import *

def fetch_prompt_local(slug:str, inputs:dict) -> str:
    with open(f"./prompts/{slug}.md","r") as f:
        prompt = f.read()
        prompt = re.sub(r'<<<.*?>>>', '', prompt, flags=re.DOTALL)
        prompt = prompt.format(**inputs)
    return prompt


if __name__ == "__main__":
    TOPIC = "Generate DAG from Paper"
    APP_SLUG = "paper_to_real"          # 换成你在 Agenta 的 app_slug
    ENV_SLUG = "production"      # 或 staging/dev

    variant = {
            "check": {
                "slug": "check_topic",
                "version": 18,
                "inputs": {
                    "paper_path": "./papers/2508.12969v1/paper.md",
                    "score_path": "./knowledges/llm_module_optimization_classification_schema.json"
                    },
                "tools": [
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
    descriptions = ["Check whether the paper is relevant to the topic"]
    expected_outputs = ["The check result"]
    for i in range(len(descriptions)):
        tasks.append(build_task(descriptions[i],expected_outputs[i],agents[i]))

    run_pipeline(agents,tasks)
    #perf_loop = ReviewLoop(worker=agents[0], reviewer=agents[1], task_description=descriptions[0], expected_output=expected_outputs[0])
    #perf_result = perf_loop.run()

