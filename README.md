# DocRAG - Document-Based RAG System

Local RAG system for querying technical books using Qwen 2.5, ChromaDB, and FlashRank reranking.

## Features

- **Multi-book search** with parallel retrieval
- **Intent detection** for followup questions, structure queries, cross-book comparisons
- **Semantic chunking** with TOC extraction
- **Reranking** using FlashRank
- **Rate limiting** and input sanitization

## Structure

```
src/
├── app.py           # Streamlit UI
├── config.py        # Configuration
├── ingest.py        # Document ingestion
├── core/
│   ├── intent.py    # Query intent detection
│   ├── retriever.py # Vector search with dedup
│   └── generator.py # LLM response generation
└── utils/
    ├── models.py    # Model loading
    └── security.py  # Rate limiting, sanitization
```

## Setup

```bash
conda create -n rag python=3.11
conda activate rag
pip install -r requirements.txt
```

## Usage

### Ingest Documents

```bash
python src/ingest.py
```

Place PDF/EPUB files in `data/` folder before running.

### Run App

```bash
streamlit run src/app.py --server.port 8501
```

## Configuration

Edit `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | Qwen/Qwen2.5-1.5B-Instruct | LLM model |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Embedding model |
| `CHUNK_SIZE` | 1500 | Characters per chunk |
| `TOP_K` | 5 | Results to return |

## Query Types

| Type | Example | Behavior |
|------|---------|----------|
| Specific book | "Who wrote AI Engineering?" | Filters to that book |
| Followup | "What chapters does it have?" | Uses previous book context |
| Cross-book | "What do all books say about RAG?" | Searches all books |
| Structure | "List the chapters" | Prioritizes TOC content |

## Tech Stack

- **LLM**: Qwen 2.5 1.5B (4-bit quantized)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: ChromaDB
- **Reranker**: FlashRank ms-marco-MultiBERT-L-12
- **UI**: Streamlit

## License

MIT
