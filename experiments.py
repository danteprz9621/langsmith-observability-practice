"""
Chapter 3b -- Experiments: evaluate() with code evaluators and LLM-as-judge.

Goal: run agents.rag_agent.ask() over the whole datasets.py dataset in one
`evaluate()` call, scored by two different evaluator styles, and land the
results in the LangSmith UI as a comparable experiment run.

Docs: docs.smith.langchain.com/evaluation/how_to_guides/evaluate_llm_application

TODO:
1. Write a target function: takes a dataset example's inputs, calls
   agents.rag_agent.ask(), returns outputs in the shape evaluators expect.

2. Write one **code evaluator** -- pure Python, no LLM call, e.g. checking
   the answer contains an expected keyword/phrase, or that retrieval_context
   is non-empty. Fast and deterministic; use it for anything checkable
   without judgment.

3. Write one **LLM-as-judge evaluator** using LangSmith's evaluator
   helpers (or DeepEval/RAGAS metrics wrapped as a LangSmith evaluator) for
   the criteria-based examples from datasets.py step 3 that a code
   evaluator can't score -- e.g. "did the answer correctly refuse when the
   KB had no relevant info."

4. Call `evaluate()` with the target function, both evaluators, and the
   dataset name from datasets.py. Confirm per-example scores and an
   aggregate summary appear in the LangSmith UI, not just stdout.

5. Run it twice (e.g. after a deliberate prompt tweak in rag_agent.py) and
   use the LangSmith UI's experiment-comparison view to see which examples
   regressed vs. improved -- this comparison is the exact mechanism
   ci_gate.py in Chapter 4 turns into an automated CI check.
"""
