# Autonomous RAG

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-4B3A55?logo=python\&logoColor=white)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-Vector%20DB-5B4B8A)](https://www.trychroma.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)

A Python-based Retrieval-Augmented Generation (RAG) system that retrieves relevant information from PDF documents and uses an LLM to generate answers based on the retrieved context.

## Features

* PDF document loading
* Text chunking
* Sentence Transformer embeddings
* ChromaDB vector storage
* Vector similarity search
* Cross-encoder reranking
* Query rewriting
* Groq LLM integration

## Tech Stack

* Python 3.13
* LangChain
* Sentence Transformers
* ChromaDB
* PyMuPDF
* PyTorch
* Groq

## Project Structure

```text
autonomus-rag/
├── main.py
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── notebooks/
├── data/
└── src/
    ├── chunker/
    ├── embeddings/
    ├── loaders/
    ├── query_rewriter/
    ├── retrieval/
    └── vector_db/
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/karthik280307/autonomus-rag.git
cd autonomus-rag
```

### 2. Create a virtual environment

Using Python:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Using uv

If you use `uv`:

```bash
uv sync
```

Then activate the environment if needed:

```powershell
.venv\Scripts\Activate.ps1
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key
```

Replace `your_api_key` with your Groq API key.

## Usage

### Add PDF Documents

Place your PDF documents inside the `data` directory.

```text
data/
├── document1.pdf
├── document2.pdf
└── document3.pdf
```

### Ingest Documents

Run the document ingestion pipeline:

```bash
python main.py --mode ingest
```

This loads the PDFs, splits the text into chunks, creates embeddings, and stores them in ChromaDB.

### Query the Documents

Run a query:

```bash
python main.py --mode query --query "What is RAG?"
```

The system retrieves relevant chunks, reranks them, rewrites the query when required, and generates the final answer using the Groq LLM.

## RAG Pipeline

```text
PDF Documents
      ↓
Document Loader
      ↓
Text Chunking
      ↓
Embeddings
      ↓
ChromaDB
      ↓
Vector Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Query Rewriting
      ↓
Groq LLM
      ↓
Final Answer
```

## How It Works

1. **Document Loading**
   PDF files are loaded and converted into text.

2. **Text Chunking**
   The extracted text is divided into smaller chunks for efficient retrieval.

3. **Embeddings**
   Sentence Transformers convert the chunks into vector representations.

4. **Vector Storage**
   The embeddings are stored in ChromaDB.

5. **Retrieval**
   Relevant chunks are retrieved based on vector similarity.

6. **Reranking**
   A cross-encoder reranks the retrieved chunks to improve relevance.

7. **Query Rewriting**
   The user query can be rewritten to improve retrieval quality.

8. **Generation**
   The retrieved context is passed to the Groq LLM to generate the final answer.

## Example

```text
Question:
What is Retrieval-Augmented Generation?

Retrieved Context:
Relevant information retrieved from the uploaded PDF documents.

Answer:
RAG is a technique that combines information retrieval with
language generation. It retrieves relevant external information
and provides it as context to an LLM before generating an answer.
```

## Requirements

* Python 3.13
* Groq API key
* PDF documents
* Internet connection for the LLM API
* Sufficient RAM for embedding and reranking models

## Future Improvements

* Web interface
* Conversation history
* Multiple document collections
* Better retrieval strategies
* Source citations in answers
* Local LLM support

## License

This project is for learning and experimentation.
