"""Flat RAG baseline: OpenAI embeddings + numpy cosine similarity (no native deps)."""
import json
import pickle
import time
import numpy as np
from openai import OpenAI

from . import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

INDEX_PATH = config.OUTPUTS_DIR / "flat_index.pkl"


def _embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # normalize for cosine = dot product
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs


def index(corpus_path=None):
    corpus_path = corpus_path or config.CORPUS_PATH
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    docs = [d["text"] for d in corpus]
    ids = [d["id"] for d in corpus]
    vecs = _embed(docs)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"ids": ids, "docs": docs, "vecs": vecs}, f)
    print(f"Indexed {len(docs)} documents -> {INDEX_PATH}")


def _load():
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def answer(question: str, k: int = 4) -> dict:
    idx = _load()
    t0 = time.time()
    qv = _embed([question])[0]
    sims = idx["vecs"] @ qv
    top = np.argsort(-sims)[:k]
    docs = [idx["docs"][i] for i in top]
    context = "\n\n".join(f"- {d}" for d in docs)

    prompt = f"""Answer the question using ONLY the context below. If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
    resp = client.chat.completions.create(
        model=config.ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    elapsed = time.time() - t0
    return {
        "answer": resp.choices[0].message.content.strip(),
        "context_docs": docs,
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    index()
    r = answer("Who founded OpenAI?")
    print(r["answer"])
