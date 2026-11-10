"""
RAG Evaluation using Ragas metrics.

Metrics computed:
- Recall@k         : are relevant chunks retrieved?
- MRR              : how highly are relevant chunks ranked?
- Faithfulness     : does the answer hallucinate?
- Answer Relevance : does the answer actually address the question?
- Latency          : how long does the system take?

Usage:
    from evaluation import evaluate_rag, save_result, print_summary
"""

import time
import pandas as pd
import os
import pandas as pd

# ── Ragas imports ──────────────────────────────────────────────────────────────
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Retrieval metrics  (no LLM required)
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_docs, relevant_sources, k=None):
    """
    Recall@k – fraction of relevant source files found in the top-k chunks.

    Parameters
    ----------
    retrieved_docs   : list of LangChain Document objects (already top-k)
    relevant_sources : set/list of ground-truth source filenames
    k                : if None, uses all retrieved docs

    Returns float in [0, 1], or None if relevant_sources not provided.
    """
    if not relevant_sources:
        return None
    top = retrieved_docs[:k] if k else retrieved_docs
    retrieved_sources = set(d.metadata.get("source", "") for d in top)
    hits = retrieved_sources & set(relevant_sources)
    return len(hits) / len(set(relevant_sources))


def mean_reciprocal_rank(retrieved_docs, relevant_sources):
    """
    MRR – 1 / rank of the first relevant chunk.

    Returns float in (0, 1], or 0.0 if no relevant chunk found.
    Returns None if relevant_sources not provided.
    """
    if not relevant_sources:
        return None
    relevant_set = set(relevant_sources)
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc.metadata.get("source", "") in relevant_set:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Generation metrics via Ragas  (requires LLM, e.g. OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ragas_metrics(question: str, answer: str, retrieved_docs) -> dict:
    """
    Compute Faithfulness and Answer Relevance via Ragas.

    Returns dict with keys:
        faithfulness        – float in [0, 1]
        answer_relevancy    – float in [0, 1]

    If Ragas / OpenAI is unavailable, returns None values with an error message.

    Note: Ragas needs access to an LLM (default: OpenAI gpt-4o-mini).
    Set the OPENAI_API_KEY environment variable, or configure a custom LLM:
    https://docs.ragas.io/en/stable/howtos/customizations/
    """
    try:
        contexts = [doc.page_content for doc in retrieved_docs]
        data = {
            "question": [question],
            "answer":   [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
        )
        scores = result.to_pandas().iloc[0]
        return {
            "faithfulness":     float(scores.get("faithfulness",     float("nan"))),
            "answer_relevancy": float(scores.get("answer_relevancy", float("nan"))),
        }
    except Exception as e:
        print(f"[Ragas] Could not compute generation metrics: {e}")
        return {"faithfulness": None, "answer_relevancy": None}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rag(
    question: str,
    answer: str,
    retrieved_docs,
    embedding: str,
    llm: str,
    latency_sec: float,
    relevant_sources=None,
    k: int = 5,
    compute_generation_metrics: bool = True,
) -> dict:
    """
    Evaluate a single RAG response and return a dict of all metrics.

    Parameters
    ----------
    question                  : the user's question
    answer                    : the model's generated answer
    retrieved_docs            : list of LangChain Documents returned by retriever
    embedding                 : embedding model name key (e.g. "bge")
    llm                       : LLM name key (e.g. "qwen7b")
    latency_sec               : wall-clock time of the full pipeline (seconds)
    relevant_sources          : list of ground-truth source filenames (for Recall/MRR)
    k                         : top-k cutoff for Recall@k
    compute_generation_metrics: set False to skip Ragas (faster, no API key needed)
    """
    row = {
        "question":    question,
        "embedding":   embedding,
        "llm":         llm,
        "answer":      answer,
        "sources":     str([d.metadata.get("source", "") for d in retrieved_docs]),
        "latency_sec": round(latency_sec, 3),
    }

    # Retrieval metrics
    row["recall_at_k"] = recall_at_k(retrieved_docs, relevant_sources, k=k)
    row["mrr"]         = mean_reciprocal_rank(retrieved_docs, relevant_sources)

    # Generation metrics (Ragas)
    if compute_generation_metrics:
        gen = compute_ragas_metrics(question, answer, retrieved_docs)
        row["faithfulness"]     = gen["faithfulness"]
        row["answer_relevancy"] = gen["answer_relevancy"]
    else:
        row["faithfulness"]     = None
        row["answer_relevancy"] = None

    return row


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_result(row: dict, path: str = "results/results.csv"):
    """Append a single evaluation row to the CSV."""
    df_new = pd.DataFrame([row])
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            existing = pd.read_csv(path)
        else:
            existing = pd.DataFrame()
        df_new = pd.concat([existing, df_new], ignore_index=True)
    except FileNotFoundError:
        pass
    df_new.to_csv(path, index=False)


def print_summary(path: str = "results/results.csv"):
    """Print mean metrics grouped by (embedding, llm) configuration."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("No results file found.")
        return

    numeric_cols = ["latency_sec", "recall_at_k", "mrr",
                    "faithfulness", "answer_relevancy"]
    available = [c for c in numeric_cols if c in df.columns]

    summary = (
        df.groupby(["embedding", "llm"])[available]
        .mean()
        .round(4)
    )
    print("\n=== RAG Evaluation Summary ===")
    print(summary.to_string())
    return summary
