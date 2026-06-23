# RAG Pipeline — Local Document Q&A

A fully local Retrieval-Augmented Generation (RAG) system built with LangChain,
ChromaDB, and Ollama. No API keys, no cloud, no cost.

## Architecture
PDF Documents
↓
Chunking (RecursiveCharacterTextSplitter)
↓
Embeddings (nomic-embed-text via Ollama)
↓
Vector Store (ChromaDB)
↓
Query
↓
Hybrid Retrieval (BM25 + Dense — Reciprocal Rank Fusion)
↓
LLM Reranking (qwen3 via Ollama)
↓
Generation (qwen3 via Ollama)

## Stack

| Component       | Technology                        |
|----------------|-----------------------------------|
| Orchestration   | LangChain 1.3                    |
| Embeddings      | nomic-embed-text (Ollama)        |
| Vector Store    | ChromaDB                         |
| Sparse Retrieval| BM25 (rank-bm25)                 |
| Fusion          | Reciprocal Rank Fusion (custom)  |
| Reranker        | LLM-based scoring (qwen3)        |
| LLM             | qwen3 (Ollama)                   |

## Key Design Decisions

**Hybrid Search over Dense-only**
Pure vector search misses exact keyword matches (e.g. legal codes, model numbers,
proper nouns). BM25 handles lexical matching while dense retrieval handles semantic
similarity. Reciprocal Rank Fusion combines both rankings without requiring score
normalization.

**Local-first**
All models run via Ollama — no data leaves the machine. Suitable for confidential
documents.

**Dedicated embedding model**
nomic-embed-text is used exclusively for embeddings. Using a general-purpose LLM
for embeddings produces lower-quality vector representations. Separation of concerns
between embedding and generation improves retrieval quality.

**LLM reranker**
After hybrid retrieval returns up to 8 candidate chunks, the LLM scores each chunk
against the query (0–10) and the top 2 are passed to generation. Reduces noise in
the context window.

**Prompt anchoring**
The system prompt instructs the model to answer exclusively from retrieved context
and explicitly return "not found" when the answer is absent — reducing hallucination.

## Setup

### Requirements
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Install models
```bash
ollama pull nomic-embed-text
ollama pull qwen3
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Add your documents
Place PDF files in the `docs/` folder.

### Index
```bash
python indexer.py
```

### Query
```bash
python query.py
```

## Usage
Pergunta: What is the main topic of the document?

Pergunta: /debug NBR 9050

Pergunta: /info

Pergunta: sair

### Debug mode
`/debug <query>` reveals the full retrieval pipeline:
- Hybrid retrieval results (BM25 + Dense)
- Reranker scores per chunk
- Final chunks sent to the LLM

## Project Structure
RAG/

├── indexer.py        # Document ingestion and vector store creation

├── query.py          # Query interface with hybrid retrieval

├── requirements.txt

├── README.md

└── docs/             # Place your PDFs here (gitignored)

## Retrieval Pipeline Detail
Query

├── BM25 → top-4 chunks (lexical match)

└── Dense → top-4 chunks (semantic match)

↓

Reciprocal Rank Fusion → deduplicated, re-ranked (up to 8 chunks)

↓

LLM Reranker → top-2 chunks scored by relevance

↓

Prompt + Generation

## What I learned building this

- Chunk size and overlap directly impact retrieval quality — smaller chunks improve
  precision but lose surrounding context
- Dense retrieval alone fails on exact keyword queries (codes, IDs, proper nouns)
- BM25 alone fails on paraphrased or semantically equivalent queries
- Hybrid search with RRF consistently outperforms either method in isolation
- A dedicated embedding model outperforms a general LLM for vector representations
