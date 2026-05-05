"""Xây dựng đồ thị NetworkX (+ Neo4j optional) từ triples."""
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt

from . import config


def build_networkx(triples: list[dict]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for t in triples:
        s, r, o = t["subject"], t["relation"], t["object"]
        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, relation=r, source_doc=t.get("source_doc"))
    return G


def save_graph(G: nx.MultiDiGraph, path=None):
    path = path or config.GRAPH_PATH
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_graph(path=None) -> nx.MultiDiGraph:
    path = path or config.GRAPH_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


def visualize(G: nx.MultiDiGraph, path=None, max_nodes=60):
    path = path or config.GRAPH_PNG
    if G.number_of_nodes() > max_nodes:
        deg = dict(G.degree())
        top = sorted(deg, key=deg.get, reverse=True)[:max_nodes]
        H = G.subgraph(top).copy()
    else:
        H = G

    plt.figure(figsize=(20, 14))
    pos = nx.spring_layout(H, k=1.2, iterations=80, seed=42)
    nx.draw_networkx_nodes(H, pos, node_size=700, node_color="#90caf9", alpha=0.9)
    nx.draw_networkx_labels(H, pos, font_size=8)
    nx.draw_networkx_edges(H, pos, edge_color="#888", arrows=True, alpha=0.5,
                           connectionstyle="arc3,rad=0.1")
    edge_labels = {(u, v): d["relation"] for u, v, d in H.edges(data=True)}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=6)
    plt.title(f"Tech Company Knowledge Graph ({H.number_of_nodes()} nodes / {H.number_of_edges()} edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph image -> {path}")


def push_to_neo4j(triples: list[dict]):
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("neo4j driver not installed; skipping")
        return
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        for t in triples:
            s.run(
                "MERGE (a:Entity {name:$s}) MERGE (b:Entity {name:$o}) "
                "MERGE (a)-[r:REL {type:$r}]->(b)",
                s=t["subject"], o=t["object"], r=t["relation"],
            )
    driver.close()
    print(f"Pushed {len(triples)} triples to Neo4j @ {config.NEO4J_URI}")


def run(triples_path=None, push_neo4j: bool = False):
    triples_path = triples_path or config.TRIPLES_PATH
    with open(triples_path, encoding="utf-8") as f:
        data = json.load(f)
    triples = data["triples"]

    G = build_networkx(triples)
    save_graph(G)
    visualize(G)
    print(f"Graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    if push_neo4j:
        push_to_neo4j(triples)
    return G


if __name__ == "__main__":
    import sys
    run(push_neo4j="--neo4j" in sys.argv)
