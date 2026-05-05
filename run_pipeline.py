"""End-to-end pipeline: extract -> build graph -> index flat -> evaluate."""
import argparse
from src import extractor, graph_builder, flat_rag, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true", help="Reuse existing triples.json")
    ap.add_argument("--skip-flat-index", action="store_true", help="Reuse existing ChromaDB")
    ap.add_argument("--neo4j", action="store_true", help="Also push to Neo4j")
    ap.add_argument("--only", choices=["extract", "graph", "flat", "eval"])
    args = ap.parse_args()

    if args.only == "extract" or (not args.only and not args.skip_extract):
        print("\n=== STEP 1: Extract triples ===")
        extractor.run()

    if args.only == "graph" or not args.only:
        print("\n=== STEP 2: Build graph ===")
        graph_builder.run(push_neo4j=args.neo4j)

    if args.only == "flat" or (not args.only and not args.skip_flat_index):
        print("\n=== STEP 3: Index Flat RAG ===")
        flat_rag.index()

    if args.only == "eval" or not args.only:
        print("\n=== STEP 4: Evaluate ===")
        evaluate.run()


if __name__ == "__main__":
    main()
