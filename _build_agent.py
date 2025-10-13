import os
import time
import agenta as ag
from agenta.sdk.types import PromptTemplate
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from opentelemetry import trace, context
import re



ag.init()  # 会读取 AGENTA_HOST / AGENTA_API_KEY
#tracer = trace.get_tracer(__name__)


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
            #model = "anthropic/GLM-4.5",
            #model = "openai/kimi-k2-250711",
            #model = "openai/doubao-seed-1-6-thinking-250715",
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
                max_execution_time=1800,
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

#@ag.instrument()
def log_task_output(task_output):
    """capture each trace of task completition"""
    return {"task_output": str(task_output)}

#@ag.instrument()
def log_step(step_output):
    """capture each trace of step"""
    attrs = ["result", "thought", "tool", "tool_result"]
    
    #@ag.instrument()
    def log_step_output(step_output, attrs):
        return {attr: getattr(step_output, attr, None) for attr in attrs}

    if getattr(step_output, "thought", None) is not None:
        return log_step_output(step_output, attrs)

#@ag.instrument()
def run_pipeline(agents: list, tasks: list):
    parent_span = trace.get_current_span()
    parent_ctx = trace.set_span_in_context(parent_span)
    def step_cb_wrapped(step_output):
        token = context.attach(parent_ctx)
        try:
            log_step(step_output)
        finally:
            context.detach(token)

    def task_cb_wrapped(task_output):
        token = context.attach(parent_ctx)
        try:
            log_task_output(task_output)
        finally:
            context.detach(token)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        task_callback=task_cb_wrapped,
        step_callback=step_cb_wrapped
    )
    final_output = crew.kickoff()
    return tasks[-1].output.raw #final_output.raw

class ReviewLoop:
    def __init__(self, worker, reviewer, task_description, expected_output, inputs=None, max_rounds=5):
        self.worker = worker
        self.reviewer = reviewer
        self.task_description = task_description
        self.expected_output = expected_output
        self.inputs=inputs
        self.max_rounds = max_rounds
    
    #@ag.instrument()
    def run(self):
        #parent_span = ag.tracing.get_current_span()
        #tracer = trace.get_tracer(__name__)


        round_num = 0
        work_result = None
        review_result = None

        while round_num < self.max_rounds:
            round_num = round_num + 1

            if round_num == 1:
                if self.inputs == None:
                    description = self.task_description
                else:
                    description = self.task_description + \
                                  f"\nThere is the submission of previous agent: {self.inputs}"
                work_task = Task(
                    description = self.task_description,
                    agent = self.worker,
                    expected_output = self.expected_output
                )
            else:
                if self.inputs == None:
                    description = "Your previous version submission was not approved. Please make the necessary changes based on the feedback provided\n" + \
                                  f"The previous submission: {work_result}\n" + \
                                  f"The feedback: {review_result}"
                else:
                    description = "Your previous version submission was not approved. Please make the necessary changes based on the feedback provided\n" + \
                    f"The previous submission: {work_result}\n" + \
                    f"The feedback: {review_result}\n" + \
                    f"There is the submission of previous agent: {self.inputs}"
                work_task = Task(
                    description = description,
                    agent = self.worker,
                    expected_output = self.expected_output
                )

            '''    
            work_crew = Crew(
                agents=[self.worker],
                tasks=[work_task],
                process=Process.sequential,
                verbose=True,
                task_callback=log_task_output,
                step_callback=log_step_output
            )

            work_crew.kickoff()
            work_result = work_task.output.raw
            '''
            review_task = Task(
                description=f"There is the submission of previous agent. You need to check it",
                agent=self.reviewer,
                expected_output = "Check Result"
            )
            '''
            review_crew = Crew(
                agents=[self.reviewer],
                tasks=[review_task],
                process=Process.sequential,
                verbose=True,
                task_callback=log_task_output,
                step_callback=log_step_output
            )
            #with tracer.start_as_current_span("parent_span") as parent_span:
            review_crew.kickoff()
            '''
            run_pipeline([self.worker, self.reviewer], [work_task, review_task])
            work_result = work_task.output.raw
            review_result = review_task.output.raw
            #print(dir(work_result))

            if "congratulation" in review_result.lower():
                return work_result

        raise ValueError("Exceeded the loop limit, the Agent failed to provide a qualified result")


