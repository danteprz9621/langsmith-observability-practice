"""
Chapter 5 -- Online-eval / production drift monitoring.

Scores simulated "live" traffic (no reference answer, unlike golden_dataset.py's
curated goldens) using the same referenceless evaluators as experiments.py
(has_retrieval, faithfulness), attaching scores directly to each trace via
LangSmith feedback -- see the note above run_online_eval() re: TODO 4.

Docs: docs.smith.langchain.com/evaluation/how_to_guides/online_evaluations,
docs.smith.langchain.com/observability/how_to_guides/monitoring
"""

import asyncio
import uuid

from langsmith import Client
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from agents.rag_agent import ask

judge_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
judge_llm = llm_factory("llama3.1:8b", provider="openai", client=judge_client)
faithfulness = Faithfulness(llm=judge_llm)

ls = Client()

# TODO 1: simulated production traffic -- deliberately NOT golden_dataset.py's
# curated 8. Messier phrasing, typos, multi-part asks, and one question
# that's genuinely off-topic (no answer anywhere in the KB).
PRODUCTION_TRAFFIC = [
    "hey how many bags can i check on an international economy fare and whats the weight limit",
    "need to push my flight back a week, is that even possible and whats it gonna cost me",
    "airline just cancelled my flight!! what happens now",
    "how do points work and do they ever expire",
    "refund taking forever, how long is this supposed to take",
    "whats covered under the travel insurance yall sell",
    "do i need a visa? i'm not sure whos even responsible for sorting that out",
    "can my emotional support dog fly in the cabin with me and what does that cost",
    "whats the wifi password on your flights",  # off-topic, not in the KB at all
    "my bag got lost AND delayed, what do i do, does insurance cover that too",
]


def has_retrieval(outputs: dict) -> bool:
    return bool(outputs.get("retrieval_context"))


def faithfulness_score(question: str, outputs: dict) -> float | None:
    ctx = outputs.get("retrieval_context") or []
    if not ctx:
        return None
    result = asyncio.run(
        faithfulness.ascore(
            user_input=question,
            response=outputs["answer"],
            retrieved_contexts=ctx,
        )
    )
    return float(result.value)


def run_online_eval() -> list[dict]:
    """TODO 2 + 3: call ask() over production traffic, traced + scored referencelessly.

    NOTE on TODO 4: as of langsmith==0.10.10, the Python SDK's Client has no
    create_rule()/automation method -- attaching an evaluator as a live
    "online evaluation rule" that scores future traces automatically is a
    LangSmith UI step (project -> Automations -> Add Rule -> Online
    evaluator), not something this script can do. What this function does
    instead is the manual equivalent: score each trace right after it's
    made and attach the scores as feedback via the API, which IS fully
    supported. Re-run this script periodically (or from a scheduled job)
    until/unless you set up the UI rule to do it continuously.
    """
    results = []
    for question in PRODUCTION_TRAFFIC:
        run_id = str(uuid.uuid4())
        outputs = ask(question, langsmith_extra={"run_id": run_id, "tags": ["online_eval"]})

        retrieval_ok = has_retrieval(outputs)
        ls.create_feedback(run_id=run_id, key="has_retrieval", score=float(retrieval_ok))

        f_score = faithfulness_score(question, outputs)
        if f_score is not None:
            ls.create_feedback(run_id=run_id, key="faithfulness", score=f_score)

        results.append(
            {
                "run_id": run_id,
                "question": question,
                "answer": outputs["answer"],
                "has_retrieval": retrieval_ok,
                "faithfulness": f_score,
            }
        )
    return results


def lowest_scoring(results: list[dict], n: int = 3) -> list[dict]:
    """TODO 5: surface the worst-scoring traces for hand review.

    Hand review + turning confirmed misses into new golden_dataset.py entries is
    a judgment call, not something to automate -- this just does the
    "find the candidates" part.
    """
    scored = [r for r in results if r["faithfulness"] is not None]
    return sorted(scored, key=lambda r: r["faithfulness"])[:n]


if __name__ == "__main__":
    results = run_online_eval()
    print(f"Scored {len(results)} production-style traces.\n")
    for r in results:
        print(f"[faithfulness={r['faithfulness']}] {r['question'][:60]}...  (run {r['run_id']})")

    print("\nLowest-scoring -- review these, consider adding to golden_dataset.py:")
    for r in lowest_scoring(results):
        print(f"\nQ: {r['question']}\nA: {r['answer']}\nfaithfulness={r['faithfulness']}")
