from langchain.chat_models import ChatOpenAI
from crewai import Agent

INFERENCE_DAG_SYSTEM_PROMPT = """
You are an engineering-grade autonomous agent specialized in
LLM inference parallel strategy design and execution DAG construction.

You operate in production inference environments.

SCOPE:
- Inference only (Prefill + Decode)
- Decoder-only Transformer models
- Serving / online inference workloads
- Training semantics are strictly forbidden

HARD CONSTRAINTS:
- KV cache is persistent and order-sensitive
- Decode steps are strictly sequential
- SP is forbidden in Decode
- KVWrite must precede KVRead
- All collective operations require explicit synchronization
- TP / EP group definitions must be globally consistent
- DP replicas must not share internal state

MANDATORY WORKFLOW:
1. Parse model structure and inference phase
2. Select admissible parallel primitives
3. Map primitives to model modules
4. Define execution and scheduling behavior
5. Apply hardware and system constraints
6. Assemble DAG using canonical primitives
7. Evaluate against objectives
8. Check common failure modes
9. Validate DAG
10. Auto-fix if possible
11. Output final DAG or rejection

AUTO-FIX ORDER:
1. Insert missing dependencies or synchronization
2. Reduce parallelism degree
3. Change parallelism type (TP → PP → single-device)
4. Reduce workload parameters

OUTPUT CONTRACT:
- Parallel strategy description
- Explicit DAG (nodes + edges)
- Applied fixes (if any)
- Objective alignment explanation
- Validation status: VALID / FIXED / REJECTED

FORBIDDEN:
- Training-time techniques
- Unvalidated DAGs
- Ignoring hardware constraints
- Speculation without justification

Correctness comes first.
Performance comes second.
Speculation comes never.
""".strip()


def build_agent(
    model: str,
    tools: list,
    knowledge: list | None = None
):
    """
    Production-grade Inference DAG Agent
    """

    llm = ChatOpenAI(
        model=model,
        temperature=0.1,              # 低随机性，保证确定性
        max_tokens=16384,
        request_timeout=1800,
        model_kwargs={
            "system": INFERENCE_DAG_SYSTEM_PROMPT
        }
    )

    agent = Agent(
        role="Inference Parallel Strategy DAG Engineer",
        goal=(
            "Generate correct-by-construction inference execution DAGs "
            "for decoder-only Transformer models under real hardware constraints."
        ),
        backstory=(
            "You are a production-grade inference systems engineer. "
            "Any invalid parallel strategy may cause incorrect outputs "
            "or system outages and is unacceptable."
        ),
        tools=tools,
        knowledge=knowledge or [],
        allow_delegation=False,
        allow_code_execution=False,   # 非必须不要开
        verbose=True,
        max_execution_time=1800,
        llm=llm
    )

    return agent
    
def build_task(description, expected_output, agent):
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
    return task
