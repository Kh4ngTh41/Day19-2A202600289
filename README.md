# LAB DAY 19 — GraphRAG Tech Company Corpus

Pipeline: corpus → LLM trích xuất triples → đồ thị NetworkX (+ Neo4j) → 2-hop GraphRAG vs Flat RAG (ChromaDB).

## Cài đặt

```powershell
pip install -r requirements.txt
copy .env.example .env
# mở .env và điền OPENAI_API_KEY
```

## (Tùy chọn) Khởi động Neo4j

```powershell
docker compose up -d
# UI: http://localhost:7474  (user: neo4j  pass: graphrag123)
```

## Chạy toàn bộ pipeline

```powershell
python run_pipeline.py              # 4 bước: extract → graph → flat-index → eval
python run_pipeline.py --neo4j      # đẩy thêm vào Neo4j
python run_pipeline.py --skip-extract --skip-flat-index   # rerun eval
python run_pipeline.py --only eval  # chỉ eval
```

## Outputs
- `outputs/triples.json` — triples đã trích xuất + token usage
- `outputs/graph.gpickle`, `outputs/graph.png` — knowledge graph
- `outputs/comparison.csv` — bảng 20 Q × 2 systems
- `outputs/chroma_db/` — vector index Flat RAG

## Cấu trúc

```
src/
├── config.py          # đọc .env
├── extractor.py       # LLM → triples + dedup/aliases
├── graph_builder.py   # NetworkX + Neo4j + matplotlib viz
├── flat_rag.py        # Flat RAG baseline (numpy cosine similarity)
├── graph_rag.py       # entity-link + 2-hop BFS + textualize
└── evaluate.py        # benchmark + comparison.csv
```

## Test nhanh từng module
```powershell
python -m src.extractor
python -m src.graph_builder
python -m src.flat_rag
python -m src.graph_rag
python -m src.evaluate
```
