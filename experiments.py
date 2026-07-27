"""
Experiments
"""

import asyncio
from agents.rag_agent import ask
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory
from openai import AsyncOpenAI
from langsmith import evaluate
from golden_dataset import DATASET_NAME

client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
faithfulness = Faithfulness(llm=judge_llm)

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
      data=DATASET_NAME,
      evaluators=[has_retrieval, faithfulness_eval],
      experiment_prefix="retrieval_faithfulness",
   )


