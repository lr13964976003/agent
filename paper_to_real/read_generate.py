# main.py
import os
import time
import agenta as ag
from agenta.sdk.types import PromptTemplate
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, FileReadTool, FileWriterTool
from langchain_openai import ChatOpenAI
import re
from _my_tools import PythonTool, AppendFileTool


ag.init()  # 会读取 AGENTA_HOST / AGENTA_API_KEY

#@ag.instrument()
def fetch_prompt(app_slug: str, variant_slug: str, variant_version: int, inputs: dict) -> PromptTemplate:
    """
    从 Agenta 配置注册表获取 prompt 并格式化
    """
    config = ag.ConfigManager.get_from_registry(
        app_slug=app_slug,
        variant_slug=variant_slug,
        variant_version=variant_version
    )
    prompt = config["prompt"]["messages"][0]["content"]
    prompt = re.sub(r'<<<.*?>>>', '', prompt, flags=re.DOTALL)
    prompt = prompt.format(**inputs)
    #print(prompt)
    return prompt

#@ag.instrument()
def build_agent(prompt: str, tools: list):
    llm = ChatOpenAI(
            model = "anthropic/kimi-k2-0711-preview",
            temperature = 0.7,
            max_tokens = 16384,
            request_timeout = 1800
            )
    agent = Agent(
                role="Assistant",
                goal=prompt,
                backstory="You are a helpful assistant",
                tools=tools,
                allow_delegation=False,
		# allow_code_execution=True,
                verbose=True,
                llm=llm
            )
    return agent

#@ag.instrument()
def build_task(description, expected_output, agent):
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
    return task

@ag.instrument()
def log_task_output(task_output):
    """capture each trace of task completition"""
    return {"task_output": str(task_output)}

@ag.instrument()
def log_step_output(step_output):
    """capture each trace of step"""
    attrs = ["result", "thought", "tool", "tool_result"]
    return {attr: getattr(step_output, attr, None) for attr in attrs}

@ag.instrument()
def run_pipeline(topic: str, environment_slug: str, agents: list, tasks: list):
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        task_callback=log_task_output,
        step_callback=log_step_output
    )
    final_output = crew.kickoff(inputs={"topic": topic})


    return final_output

# ========= 4) 运行入口 =========
if __name__ == "__main__":
    TOPIC = "Read and Generate"
    APP_SLUG = "paper_to_real"          # 换成你在 Agenta 的 app_slug
    ENV_SLUG = "production"      # 或 staging/dev

    variant_list = ["read_paper", "check_paper", "generate_dag"]
    variant = {
            "read_paper": {
                "slug": "chain_read_paper",
                "version": 5,
                "inputs": {
                    "paper_path": "/home/wzc/data/papers/SP/paper.md",
                    "save_path": "/home/wzc/data/file-share/submission"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    AppendFileTool()
                    ]
                },
            "check_paper": {
                "slug" : "chain_check_paper",
                "version" : 4,
                "inputs": {
                    "origin_paper_path" : "/home/wzc/data/papers/SP/paper.md"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
                    AppendFileTool()
                    ]
                },
            "generate_dag": {
                "slug": "chain_generate_dag",
                "version": 7,
                "inputs": {
                    "knowledge_path": "/home/wzc/data/knowledges/llm_parallel_strategies.md",
                    "save_path": "/home/wzc/data/file-share/submission"
                    },
                "tools": [
                    FileReadTool(),
                    FileWriterTool(),
		            PythonTool()
                    ]
                }
            }
    agents = []
    for k in variant_list:
        prompt = fetch_prompt(APP_SLUG, variant[k]["slug"], variant[k]["version"], variant[k]["inputs"])
        tools = variant[k]["tools"]
        agents.append(build_agent(prompt, tools))
    tasks = []
    descriptions = ["Read Paper and Refine Paper", "Check the refine paper", "Read concise Paper and Generate DAG"]
    expected_outputs = ["The concise paper", "Check Result", "One graphviz code describing the DAG"]
    for i in range(len(agents)):
        tasks.append(build_task(descriptions[i], expected_outputs[i], agents[i]))
    result = run_pipeline(TOPIC, ENV_SLUG, agents, tasks)

