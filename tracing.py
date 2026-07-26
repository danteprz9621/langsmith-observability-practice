"""
Chapter 2 -- Tracing + root-cause analysis.

Goal: every call agents/rag_agent.ask() makes (retrieval + LLM call) shows
up as a nested trace in the LangSmith UI, so a bad answer can be root-caused
to "retrieval pulled the wrong doc" vs. "LLM ignored good context" without
re-running anything.

Two tracing mechanisms LangSmith ships, both needed here for different
reasons:
  - `@traceable` -- generic decorator, wraps ANY function (retrieve(), or
    the whole ask() pipeline) as a named span. Use this for retrieve() since
    it's plain Python, not an LLM call.
  - `wrap_anthropic` -- client wrapper specifically for Anthropic's SDK,
    auto-captures prompt/completion/token usage/latency on every call made
    through the wrapped client. rag_agent.py currently calls Ollama directly
    (see agents/rag_agent.py's `ollama.chat(...)`), so step 3 below is what
    decides how that reconciles -- don't silently swap providers without
    deciding that first.

Docs: docs.smith.langchain.com/observability (Tracing quickstart, Annotating
code for tracing)

TODO:
1. `pip install langsmith`, set LANGCHAIN_TRACING_V2=true and
   LANGCHAIN_API_KEY in .env (see .env.example). Confirm a trace shows up
   in the LangSmith UI for a single throwaway @traceable-decorated function
   before touching the real agent.

2. Decorate agents.rag_agent.retrieve() with @traceable(run_type="retriever")
   so retrieved chunks appear as a distinct span, separate from the LLM call.

3. Decide how the LLM call gets traced. Either:
   (a) swap agents/rag_agent.py's `ollama.chat()` for an Anthropic client
       call and wrap it with `wrap_anthropic`, or
   (b) keep Ollama and wrap the whole ask() in @traceable(run_type="llm")
       instead, logging inputs/outputs manually.
   Write down which one and why -- this decision affects every later
   chapter that reads traces back out.

4. Wrap the top-level ask() call itself in @traceable(run_type="chain") so
   the retriever span and the LLM span nest under one parent trace per
   question, instead of appearing as two unrelated traces.

5. Root-cause exercise: deliberately break something (e.g. shrink
   retrieve()'s top_k to 1 so it drops a needed doc) and use the LangSmith
   UI to find which span diverged -- retrieval returning the wrong/thin
   context should be visibly distinguishable from the LLM ignoring good
   context. Write down what you saw and how you told the two apart.
"""
