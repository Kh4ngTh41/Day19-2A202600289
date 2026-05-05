"""Chạy 20 câu benchmark trên Flat RAG và GraphRAG, xuất comparison.csv."""
import json
import pandas as pd
from tqdm import tqdm

from . import config, flat_rag, graph_rag


def auto_score(answer: str, expected: list[str]) -> int:
    a = answer.lower()
    hits = sum(1 for e in expected if e.lower() in a)
    return 1 if hits >= max(1, len(expected) // 2) else 0


def run():
    with open(config.BENCHMARK_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    for q in tqdm(questions, desc="Benchmark"):
        try:
            fr = flat_rag.answer(q["question"])
        except Exception as e:
            fr = {"answer": f"ERROR: {e}", "prompt_tokens": 0, "completion_tokens": 0, "elapsed_sec": 0}
        try:
            gr = graph_rag.answer(q["question"], hops=3)
        except Exception as e:
            gr = {"answer": f"ERROR: {e}", "prompt_tokens": 0, "completion_tokens": 0, "elapsed_sec": 0}

        rows.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "expected": " | ".join(q["expected_entities"]),
            "flat_rag_answer": fr["answer"],
            "flat_rag_score": auto_score(fr["answer"], q["expected_entities"]),
            "flat_rag_tokens": fr["prompt_tokens"] + fr["completion_tokens"],
            "flat_rag_time": round(fr["elapsed_sec"], 2),
            "graph_rag_answer": gr["answer"],
            "graph_rag_score": auto_score(gr["answer"], q["expected_entities"]),
            "graph_rag_tokens": gr["prompt_tokens"] + gr["completion_tokens"],
            "graph_rag_time": round(gr["elapsed_sec"], 2),
            "graph_rag_seeds": " | ".join(gr.get("seeds", [])),
        })

    df = pd.DataFrame(rows)
    df.to_csv(config.COMPARISON_CSV, index=False, encoding="utf-8")
    print(f"\nSaved -> {config.COMPARISON_CSV}")
    print("\n=== Summary ===")
    print(f"Flat RAG  accuracy: {df['flat_rag_score'].mean()*100:.1f}%   "
          f"avg tokens: {df['flat_rag_tokens'].mean():.0f}   "
          f"avg time: {df['flat_rag_time'].mean():.2f}s")
    print(f"GraphRAG  accuracy: {df['graph_rag_score'].mean()*100:.1f}%   "
          f"avg tokens: {df['graph_rag_tokens'].mean():.0f}   "
          f"avg time: {df['graph_rag_time'].mean():.2f}s")
    by_type = df.groupby("type")[["flat_rag_score", "graph_rag_score"]].mean() * 100
    print("\nAccuracy by question type (%):")
    print(by_type.round(1))
    return df


if __name__ == "__main__":
    run()
