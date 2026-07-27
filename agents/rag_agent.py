"""A minimal RAG agent: TF-IDF retrieval over local .txt docs + LLM answer.

Retrieval is deliberately low-tech (TF-IDF cosine similarity, no vector DB,
no embedding API calls) so the example runs fast and free. Swap `retrieve()`
for a real embedding-based retriever later if you want to push the example
further.
"""

import os
from glob import glob

import ollama
from dotenv import load_dotenv
from langsmith import traceable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Every LANGCHAIN_* var (tracing on/off, API key, project) lives in .env --
# load it here since this module's @traceable calls are what actually need
# it, and everything else (experiments.py, online_eval.py) imports from
# here before making its own LangSmith calls.
load_dotenv(override=True)

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")

SYSTEM_PROMPT = (
    "You are a support agent for Trailhead Travel. Answer the user's "
    "question using ONLY the provided context. If the context doesn't "
    "contain the answer, say you don't have that information."
)


def _load_documents() -> list[str]:
    docs = []
    for path in sorted(glob(os.path.join(KB_DIR, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append(f.read())
    return docs

@traceable(run_type="retriever")
def retrieve(question: str, top_k: int = 2) -> list[str]:
    """Return the top_k most relevant document chunks for the question."""
    documents = _load_documents()
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, doc_vectors)[0]
    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

@traceable(run_type="chain")
def ask(question: str, model: str = "qwen2.5-coder:7b") -> dict:
    """Retrieve relevant context, then answer the question grounded in it.

    Returns a dict with both "answer" and "retrieval_context" so tests can
    build LLMTestCase objects with retrieval_context populated.
    """
    context_chunks = retrieve(question)
    context_block = "\n\n---\n\n".join(context_chunks)
    response = call_llm(
        model=model,
        context_block=context_block,
        question=question,
        system_prompt = SYSTEM_PROMPT
    )
    return {
        "answer": response,
        "retrieval_context": context_chunks,
    }

@traceable(run_type="llm")
def call_llm(system_prompt: str, context_block: str, question: str, model: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
        ],
    )
    return response["message"]["content"]

if __name__ == "__main__":
    result = ask("What is the refund policy for cancelled flights?")
    print(result["answer"])
