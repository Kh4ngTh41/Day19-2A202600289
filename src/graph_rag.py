"""GraphRAG: entity-link → 2-hop BFS → textualize → LLM."""
import json
import time
from collections import deque
from openai import OpenAI

from . import config, graph_builder

client = OpenAI(api_key=config.OPENAI_API_KEY)


ENTITY_PROMPT = """Extract the named entities (people, companies, products) mentioned in the question. Return JSON: {"entities": ["..."]}"""


def extract_question_entities(question: str) -> list[str]:
    resp = client.chat.completions.create(
        model=config.ANSWER_MODEL,
        messages=[
            {"role": "system", "content": ENTITY_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("entities", []), resp.usage


def link_entities(entities: list[str], graph_nodes: list[str]) -> list[str]:
    """Fuzzy match question entities to graph nodes (lowercase substring)."""
    matched = []
    nodes_lower = {n.lower(): n for n in graph_nodes}
    for e in entities:
        el = e.lower().strip()
        if el in nodes_lower:
            matched.append(nodes_lower[el])
            continue
        # substring match
        for nl, n in nodes_lower.items():
            if el in nl or nl in el:
                matched.append(n)
                break
    return list(dict.fromkeys(matched))


def bfs_subgraph(G, seeds: list[str], hops: int = 2):
    """BFS k-hop, treating graph as undirected for traversal but keeping directed edges."""
    visited = set(seeds)
    frontier = deque((s, 0) for s in seeds)
    while frontier:
        node, depth = frontier.popleft()
        if depth >= hops:
            continue
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                frontier.append((nb, depth + 1))
    return G.subgraph(visited).copy()


def textualize(subG) -> str:
    lines = []
    for u, v, data in subG.edges(data=True):
        rel = data["relation"].replace("_", " ").lower()
        lines.append(f"- {u} {rel} {v}.")
    return "\n".join(lines)


def answer(question: str, hops: int = 2) -> dict:
    G = graph_builder.load_graph()
    t0 = time.time()
    entities, ent_usage = extract_question_entities(question)
    seeds = link_entities(entities, list(G.nodes()))

    if not seeds:
        elapsed = time.time() - t0
        return {
            "answer": "I don't know — no matching entity in the knowledge graph.",
            "entities": entities, "seeds": [], "context": "",
            "prompt_tokens": ent_usage.prompt_tokens,
            "completion_tokens": ent_usage.completion_tokens,
            "elapsed_sec": elapsed,
        }

    subG = bfs_subgraph(G, seeds, hops=hops)
    context = textualize(subG)

    prompt = f"""You are answering a question using a knowledge graph. Use ONLY the facts below. If the answer is not present, say "I don't know".

Knowledge graph facts:
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
        "entities": entities,
        "seeds": seeds,
        "subgraph_nodes": subG.number_of_nodes(),
        "subgraph_edges": subG.number_of_edges(),
        "context": context,
        "prompt_tokens": ent_usage.prompt_tokens + resp.usage.prompt_tokens,
        "completion_tokens": ent_usage.completion_tokens + resp.usage.completion_tokens,
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    r = answer("Who founded the company that owns Instagram?")
    print(r["answer"])
    print("---seeds:", r["seeds"])
