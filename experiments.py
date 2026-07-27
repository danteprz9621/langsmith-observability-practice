"""
Experiments
"""

import asyncio
import os

from agents.rag_agent import GROQ_API_KEY, ask
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory
from openai import AsyncOpenAI
from langsmith import Client, evaluate
from golden_dataset import DATASET_NAME

# Same local-vs-Groq split as agents/rag_agent.py -- CI sets GROQ_API_KEY,
# local dev doesn't, so the judge picks up whichever the agent is already
# using without a separate flag.
if GROQ_API_KEY:
    client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
    judge_llm = llm_factory("llama-3.1-8b-instant", provider="openai", client=client)
else:
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
faithfulness = Faithfulness(llm=judge_llm)

# CI sets EVAL_SAMPLE_SIZE to run the gate against only the first N golden
# examples instead of the full set -- keeps a CI run's request count well
# inside Groq's free-tier rate limits and cuts runtime. Unset locally, so
# local runs still cover the whole dataset.
_SAMPLE_SIZE = os.environ.get("EVAL_SAMPLE_SIZE")


def _dataset_for_run():
    if not _SAMPLE_SIZE:
        return DATASET_NAME
    ls = Client()
    dataset = ls.read_dataset(dataset_name=DATASET_NAME)
    return list(ls.list_examples(dataset_id=dataset.id, limit=int(_SAMPLE_SIZE)))

def target(inputs: dict) -> dict:
    result = ask(inputs["question"])
    return {"answer": result["answer"], "retrieval_context": result["retrieval_context"]}

def has_retrieval(inputs, outputs, reference_outputs) -> dict:
    ctx = outputs.get("retrieval_context") or []
    return {"key": "has_retrieval", "score": bool(ctx)}

def faithfulness_eval(inputs, outputs, reference_outputs) -> dict:
    ctx = outputs.get("retrieval_context") or []
    if not ctx:                                     # skip instead of letting ascore raise
        return {"key": "faithfulness", "score": None}
    result = asyncio.run(faithfulness.ascore(
        user_input=inputs["question"],
        response=outputs["answer"],
        retrieved_contexts=ctx,
    ))
    return {"key": "faithfulness", "score": float(result.value)}

def experiment_evaluate():
   return evaluate(
      target,
      data=_dataset_for_run(),
      evaluators=[has_retrieval, faithfulness_eval],
      experiment_prefix="retrieval_faithfulness",
   )


