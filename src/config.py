import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gpt-4o-mini")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag123")

DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

CORPUS_PATH = DATA_DIR / "tech_corpus.json"
BENCHMARK_PATH = DATA_DIR / "benchmark_questions.json"
TRIPLES_PATH = OUTPUTS_DIR / "triples.json"
GRAPH_PATH = OUTPUTS_DIR / "graph.gpickle"
GRAPH_PNG = OUTPUTS_DIR / "graph.png"
COMPARISON_CSV = OUTPUTS_DIR / "comparison.csv"
CHROMA_DIR = OUTPUTS_DIR / "chroma_db"
