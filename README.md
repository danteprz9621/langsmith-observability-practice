# Trailhead Travel Observability

Tracing, evaluation, and CI regression gating for Trailhead Travel's
customer-support RAG agent, using [LangSmith](https://docs.smith.langchain.com).
The earlier projects in this series —
[`trailhead-travel-agent-eval`](https://github.com/danteprz9621/trailhead-travel-agent-eval),
[`trailhead-travel-rag-eval`](https://github.com/danteprz9621/trailhead-travel-rag-eval),
[`trailhead-travel-red-team`](https://github.com/danteprz9621/trailhead-travel-red-team) —
all answer "is this answer good, and is it safe?" at test time. This one
answers the question that matters once the agent is actually running: when
an answer goes wrong, can you find out why in under a minute, and can you
stop a bad prompt change from ever reaching production in the first place?

[![Prompt-regression gate](https://github.com/danteprz9621/trailhead-travel-observability/actions/workflows/regression-gate.yml/badge.svg)](https://github.com/danteprz9621/trailhead-travel-observability/actions/workflows/regression-gate.yml)

## What's here

- **Tracing + root-cause analysis** — every call through `agents/rag_agent.py`'s
  `retrieve()` and `call_llm()` is `@traceable`-decorated and nested under
  a parent `ask()` trace, so a bad answer can be root-caused to "retrieval
  pulled the wrong doc" vs. "the LLM ignored good context" from a single
  trace in the LangSmith UI — no re-running required.
- **A versioned golden dataset + experiments** ([`golden_dataset.py`](golden_dataset.py),
  [`experiments.py`](experiments.py)) — 8 hand-written question/criteria
  pairs covering the knowledge base's 7 policy topics (plus one
  deliberately unanswerable question), scored on every run by both a code
  evaluator (`has_retrieval`) and an LLM-as-judge evaluator (`faithfulness`,
  via Ragas).
- **A CI prompt-regression gate** ([`tests/test_regression_gate.py`](tests/test_regression_gate.py),
  [`.github/workflows/regression-gate.yml`](.github/workflows/regression-gate.yml)) —
  turns that experiment run into a required GitHub Actions check with
  noise-aware thresholds, so a prompt change that tanks faithfulness or
  drops retrieval fails the PR instead of shipping silently. CI runs
  against Groq's hosted inference rather than local Ollama (a CPU-only
  runner can't run 7-8B models fast enough) on a capped 3-example sample,
  switched purely by the presence of a `GROQ_API_KEY` secret — local dev
  is unaffected and still uses Ollama against the full dataset.
- **Production drift monitoring** ([`online_eval.py`](online_eval.py)) —
  scores simulated "live" traffic (messier, uncurated questions, no
  reference answer available) with the same referenceless evaluators,
  attaching scores to each trace as LangSmith feedback — the scripted
  equivalent of an online-evaluation rule that would run continuously.

## Project structure

```
trailhead-travel-observability/
├── agents/
│   └── rag_agent.py                # unchanged from trailhead-travel-red-team, now @traceable + Ollama/Groq-switchable
├── data/
│   └── knowledge_base/             # unchanged, same 7 policy docs
├── golden_dataset.py                # builds/syncs the "travel-agency-golden" LangSmith dataset
├── experiments.py                  # evaluate() run: has_retrieval (code) + faithfulness (LLM-as-judge)
├── tests/
│   └── test_regression_gate.py     # pytest gate on evaluate() thresholds
├── .github/workflows/
│   └── regression-gate.yml         # GitHub Actions: runs the gate on every PR/push to master
├── online_eval.py                  # referenceless scoring of simulated production traffic
├── requirements.txt
├── pytest.ini
├── .env.example
└── .gitignore
```

## Setup

```bash
cd trailhead-travel-observability
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `LANGCHAIN_API_KEY` (free
LangSmith account — smith.langchain.com).

The agent itself stays free/local for development — same Ollama setup as
`trailhead-travel-rag-eval`:

```bash
ollama pull qwen2.5-coder:7b
```

Try it standalone before wiring in anything else:

```bash
python agents/rag_agent.py
```

## Running it

```bash
python golden_dataset.py                 # (re)syncs the golden dataset in LangSmith
pytest tests/test_regression_gate.py -v   # runs the eval, asserts it against thresholds
python online_eval.py                     # scores simulated production traffic
```

Every call lands a trace, and every eval run lands an experiment, in the
LangSmith UI under whatever `LANGCHAIN_PROJECT` is set to.

`golden_dataset.py` has no import guard, so importing it (as `experiments.py`
does) resyncs the dataset from `qa_criteria` every time — harmless, since
that list is static, just extra API calls per run rather than a one-time
setup step.

## CI

`.github/workflows/regression-gate.yml` runs the gate above on every push
and PR against `master`, using Groq instead of Ollama (see "What's here").
Required secrets/variables, set under the repo's **Settings → Secrets and
variables → Actions**:

- `LANGCHAIN_API_KEY` (secret)
- `GROQ_API_KEY` (secret) — free tier, no card required: console.groq.com
- `LANGCHAIN_PROJECT` (variable, optional)
