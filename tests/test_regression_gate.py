"""
Chapter 4 -- Prompt-regression gating in CI.

Goal: turn experiments.py's evaluate() run into a pass/fail pytest gate
that fails a PR when a prompt/retrieval change drops scores below a fixed
threshold -- the same shift from "eval as a report you read" to "eval as a
gate you enforce" that Chapter 1 (gates vs. observability) sets up
conceptually.

Docs: docs.smith.langchain.com/evaluation/how_to_guides/evaluate_llm_application
(look for pytest / CI integration section), plus this project's own
../ragas-capstone (noise-aware CI gating) and ../promptfoo-redteam-practice
(`redteam.yaml` CI gate) for the non-determinism-handling pattern already
used twice in this series.

TODO:
1. Import and call experiments.py's evaluate() run from inside a test
   function (not a standalone script run) so pytest can assert on it.

2. Assert the aggregate score for each evaluator from experiments.py is
   >= a fixed threshold. Pick thresholds deliberately below the current
   score (headroom for judge variance), same reasoning as
   ../ragas-capstone/scripts/measure_noise.py -- exact-match-to-current-
   score thresholds will flake on judge non-determinism alone.

3. Add response caching (LangSmith's evaluate() supports a cache, or use
   pytest fixtures/marks) so re-running the suite in CI doesn't re-spend
   API/LLM cost on unchanged examples every run.

4. Wire this test into a CI config (CircleCI, matching the roadmap's
   pick) as a required check on PRs that touch agents/rag_agent.py or its
   prompt. Confirm a deliberately bad prompt change actually fails the
   build, not just the local pytest run.
"""
