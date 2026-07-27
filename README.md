# LangSmith Observability Practice Project

A practice project for the Observability & Eval Platform phase of the
roadmap (`../../ai-testing-roadmap.md`, Phase 4) — tracing, versioned
datasets/experiments, CI regression gating, and production drift
monitoring with [LangSmith](https://docs.smith.langchain.com), layered on
top of the same "Trailhead Travel" RAG agent used since
[`../deepeval-capstone`](../deepeval-capstone), most recently guarded in
[`../promptfoo-redteam-practice`](../promptfoo-redteam-practice).

This is a **skeleton**, not a finished project. Every `.py` file below has
numbered TODO comments describing what to build — no implementation yet.
Same pattern `../promptfoo-redteam-practice` used before it was completed.

## Project structure

```
langsmith-observability-practice/
├── agents/
│   └── rag_agent.py                # unchanged from ../promptfoo-redteam-practice
├── data/
│   └── knowledge_base/             # unchanged, same 7 policy docs
├── tracing.py                      # SKELETON: Ch 2 — tracing + root-cause analysis
├── golden_dataset.py                # SKELETON: Ch 3a — versioned dataset of goldens
├── experiments.py                  # SKELETON: Ch 3b — evaluate() with code + LLM-as-judge evaluators
├── tests/
│   └── test_regression_gate.py     # SKELETON: Ch 4 — pytest CI gate on evaluate() thresholds
├── online_eval.py                  # SKELETON: Ch 5 — referenceless online eval / drift monitoring
├── requirements.txt
├── .env.example
└── .gitignore
```

## The 5 chapters — what to do, in order

**Chapter 1 — Gates vs. observability (conceptual, no code).**
Before writing anything, work out in your own words how a CI gate
(pass/fail, blocks a merge) differs from observability (a trace/dashboard
you look at) and how they complement rather than replace each other. This
framing is what makes Chapter 4 (a gate) and Chapter 5 (observability) feel
like two tools instead of a redundant pair. No file for this one — just
notes, if you want them, in this README or a scratch doc.

**Chapter 2 — Tracing + root-cause analysis.** → [`tracing.py`](tracing.py)
Wire `@traceable` / `wrap_anthropic` into `agents/rag_agent.py`'s call
path so retrieval and the LLM call show up as separate, nested spans in
the LangSmith UI. Ends with a deliberate-break exercise: prove you can
tell "bad retrieval" apart from "bad generation" from the trace alone.

**Chapter 3 — Versioned datasets + experiments.**
→ [`golden_dataset.py`](golden_dataset.py), [`experiments.py`](experiments.py)
Build a golden dataset from the knowledge base, then run `evaluate()`
against it with one code evaluator and one LLM-as-judge evaluator. Ends
with using the LangSmith experiment-comparison view to see score deltas
after a deliberate prompt change.

**Chapter 4 — Prompt-regression gating in CI.**
→ [`tests/test_regression_gate.py`](tests/test_regression_gate.py)
Turn Chapter 3's `evaluate()` run into a pytest assertion with
noise-aware thresholds (same reasoning as `../ragas-capstone`'s noise
calibration), add cost caching, and wire it into CircleCI as a required
check.

**Chapter 5 — Online-eval / production drift monitoring.**
→ [`online_eval.py`](online_eval.py)
Score simulated "live" traffic with a referenceless evaluator (no
expected answer available), attach it as a LangSmith online evaluation
rule, and close the loop by turning bad production traces into new
Chapter 3 goldens.

**Capstone.** Once all five land, write a short `CAPSTONE.md` (or fold
into this README) showing all five capabilities stacked over the same
agent: traced, dataset-evaluated, CI-gated, and production-monitored.

## Setup

```bash
cd langsmith-observability-practice
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `LANGCHAIN_API_KEY` (free
LangSmith account — smith.langchain.com). `ANTHROPIC_API_KEY` is only
needed if Chapter 2, step 3 decides to route the traced LLM call through
Anthropic instead of the existing Ollama call.

The agent itself stays free/local — same Ollama setup as
`../ragas-capstone`:

```bash
ollama pull qwen2.5-coder:7b
```

Try it standalone before wiring in tracing:

```bash
python agents/rag_agent.py
```

## Suggested build order

1. `tracing.py` — get one trace showing up in the LangSmith UI before
   anything else; every later chapter assumes tracing already works.
2. `golden_dataset.py` — write the goldens.
3. `experiments.py` — score the goldens.
4. `tests/test_regression_gate.py` — turn that score into a gate.
5. `online_eval.py` — extend eval past the gate, to live traffic.
