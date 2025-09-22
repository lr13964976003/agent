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


if __name__ == "__main__":
    TOPIC = "Generate DAG from Paper"
    Ahelix_SLUG = "paper_to_real"          # 换成你在 Agenta 的 app_slug
    ENV_SLUG = "production"      # 或 staging/dev

    variant = {
            "performance": {
                "slug": "chain_performance",
                "version": 18,
                "inputs": {
                    "dag_path": "/home/wzc/data/file-share/2025-09-09-09-01-50/proposed_moe.dot",
                    "save_path": "/home/wzc/data/file-share/submission"
                    },
                "tools": [
                    ExtractEdgeFromDAGTool(),
                    FileReadTool(),
                    CommandTool(),
                    FileWriterTool()
                    ]
                },
            "check_performance": {
                "slug": "chain_check_perf",
                "version": 3,
                "inputs": {
                    "save_path": "/home/wzc/data/file-share/submission"
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
        prompt = fetch_prompt(Ahelix_SLUG, variant[k]["slug"], variant[k]["version"], variant[k]["inputs"])
        tools = variant[k]["tools"]
        agents.append(build_agent(prompt, tools))
    tasks = []
    descriptions = ["Estimate performance"]
    expected_outputs = ["The performance of DAG"]

    perf_loop = ReviewLoop(worker=agents[0], reviewer=agents[1], task_description=descriptions[0], expected_output=expected_outputs[0])
    perf_result = perf_loop.run()

