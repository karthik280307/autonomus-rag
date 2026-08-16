# Autonomus RAG

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-4B3A55?logo=python&logoColor=white)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-Vector%20DB-5B4B8A)](https://www.trychroma.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

Autonomus RAG is a Python-based retrieval-augmented generation (RAG) project that combines document loading, chunking, embeddings, vector storage, and query rewriting to support retrieval-oriented search workflows. The repository currently contains the core backend pipeline and exploratory notebooks rather than a production web application or API server.

## Table of Contents

- [Features](#features)
  - [Backend features](#backend-features)
  - [Frontend / UI features](#frontend--ui-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Repository Notes](#repository-notes)

## Features

### Backend features

- Document ingestion from PDF files
  - The project includes a document loader built on LangChain’s directory-based loader and PyMuPDF, configured to scan PDF files from a specified directory.

- Text chunking
  - A chunking component wraps LangChain’s recursive character splitter with configurable chunk size and overlap values.

- Embedding generation
  - The embedding layer uses SentenceTransformers and supports generating dense vector embeddings for arbitrary text batches.

- Persistent vector storage
  - The repository implements a Chroma-backed vector store with persistent storage on disk. Documents and embeddings are added into a collection and queried by embedding similarity.

- Query rewriting pipeline
  - The project includes a multi-step query rewriting workflow:
    - Reformulation: rewrites a user query into a clearer standalone search query.
    - Expansion: generates multiple alternative search queries to improve recall.
    - Step-back: creates a broader, more general query to retrieve supporting background knowledge.
  - These components are organized into a reusable query rewriter pipeline.

- Retrieval and reranking support
  - The retrieval package defines abstract retrieval interfaces and data models for dense, sparse, and hybrid retrieval strategies.
  - A cross-encoder reranker is also implemented using SentenceTransformers’ CrossEncoder model for re-scoring candidate passages.

- Notebook-based exploration
  - The repository includes a notebooks directory with modules related to chunking, document loading, embeddings, vector storage, retrieval, and query rewriting. These appear to be exploratory or instructional assets rather than application entry points.

### Frontend / UI features

- No frontend implementation was found in the current workspace.
- No web routes, React/Vue/Svelte components, REST or GraphQL API handlers, or browser-based UI were identified in the repository contents inspected.

## Tech Stack

| Area | Technology | Evidence in repository |
| --- | --- | --- |
| Language | Python 3.13 | [.python-version](.python-version) and [pyproject.toml](pyproject.toml) |
| Packaging | pyproject.toml, requirements.txt | [pyproject.toml](pyproject.toml) and [requirements.txt](requirements.txt) |
| LLM orchestration | LangChain, LangChain Core, LangChain Community, LangChain Groq | [pyproject.toml](pyproject.toml) and [src/query_rewriter/rewriter.py](src/query_rewriter/rewriter.py) |
| Embeddings | SentenceTransformers | [src/embeddings/embedding_manager.py](src/embeddings/embedding_manager.py) |
| Vector database | Chroma | [src/vector_db/vector_store.py](src/vector_db/vector_store.py) |
| PDF loading | PyMuPDF, PyPDF | [src/loaders/document_loader.py](src/loaders/document_loader.py) and [pyproject.toml](pyproject.toml) |
| ML runtime | PyTorch, Transformers | [pyproject.toml](pyproject.toml) |
| Numerical computing | NumPy | [src/embeddings/embedding_manager.py](src/embeddings/embedding_manager.py) and [src/vector_db/vector_store.py](src/vector_db/vector_store.py) |
| Notebook environment | Jupyter notebooks | [notebooks](notebooks) |

## Project Structure

```text
.
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── .env
├── data/
├── notebooks/
├── src/
│   ├── chunker/
│   ├── embeddings/
│   ├── loaders/
│   ├── query_rewriter/
│   ├── retrieval/
│   └── vector_db/
└── vector_store/
```

## Setup

The repository declares its Python dependencies in [pyproject.toml](pyproject.toml) and [requirements.txt](requirements.txt).

### Prerequisites

- Python 3.13
- Access to a Groq-compatible LLM environment, as the query rewriter imports Groq support through LangChain
- A local environment variable named `GROQ_API_KEY` is referenced in the repository’s local environment file

### Example setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell, the activation command is typically:

```powershell
.venv\Scripts\Activate.ps1
```

### Environment configuration

The repository includes a local environment file at [.env](.env). Its contents were not copied into this README because the file contains a real secret value; a safe example is shown below:

```env
GROQ_API_KEY=your-key-here
```

## Repository Notes

- The current codebase is primarily a backend Python package for RAG-related components.
- The entry point in [main.py](main.py) is a simple placeholder and does not start a web service.
- The repository contains a persistent Chroma database under [vector_store](vector_store), but no deployment or production infrastructure files were found.
- Some information could not be inferred from the codebase, including any production frontend, authentication model, API contract, or deployment configuration, because those components are not present in the repository.