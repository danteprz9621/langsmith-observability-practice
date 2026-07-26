"""
Chapter 5 -- Online-eval / production drift monitoring.

Goal: score a sample of *live* traffic (no reference answer available,
unlike datasets.py's offline goldens) and catch quality drift after
deployment -- the gap every gate in Chapter 4 can't cover, since CI gates
only see the questions someone thought to write down.

Docs: docs.smith.langchain.com/evaluation/how_to_guides/online_evaluations,
docs.smith.langchain.com/observability/how_to_guides/monitoring

TODO:
1. Simulate "production traffic" by generating a batch of questions not in
   datasets.py's dataset (vary phrasing/topic mix vs. the curated goldens
   -- realistic traffic is messier than a hand-written golden set).

2. Run agents.rag_agent.ask() over that batch with tracing.py's tracing
   already wired in, so every call lands in LangSmith as production-style
   traces rather than an experiment run.

3. Write a referenceless evaluator -- no expected answer to compare
   against, so score properties of the answer alone: groundedness against
   its own retrieved context (reuse the Faithfulness approach from
   ../ragas-capstone/tests/test_rag_agent.py), refusal-when-appropriate,
   response length/format sanity.

4. Attach that evaluator as a LangSmith online evaluation rule so it
   scores sampled traces automatically going forward, not just this one
   batch.

5. Close the loop: pull the lowest-scoring traces from this run, hand-review
   them, and turn confirmed failures into new golden examples added back
   into datasets.py -- production failures becoming new regression-gate
   coverage is the point, not just a dashboard number.
"""
