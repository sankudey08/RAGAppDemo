# 📄 Requirements.md — RAG Application

## 1. Overview

A Retrieval-Augmented Generation (RAG) application that:

- Parses **PDFs, text documents, and images** into embeddings.
- Stores embeddings in **ChromaDB** for semantic retrieval.
- Provides a **Streamlit UI** for user interaction.
- Implements **streaming responses** from the LLM.
- Returns **citations** for retrieved sources.
- Uses **semantic caching** to avoid redundant queries.
- Integrates **Ollama** for local LLM inference.
- Uses **LangGraph** for agent orchestration and workflow management.

---

## 2. Tech Stack

- **Language**: Python 3.10+
- **Vector DB**: ChromaDB
- **Frontend/UI**: Streamlit
- **LLM Integration**:
  - **Ollama** (local inference, streaming support)
  - OpenAI API or HuggingFace (optional fallback)
- **Agent Framework**: LangGraph (for conditional workflows, streaming, tool orchestration)
- **Parsing Libraries**:
  - `PyMuPDF` / `pdfplumber` → PDF parsing
  - `docx` / `python-docx` → DOCX parsing
  - `Pillow` / `pytesseract` → Image OCR parsing
- **Caching**: Semantic cache layer using FAISS or ChromaDB metadata
- **Streaming**: Async streaming via Ollama API or LangGraph callbacks

---

## 3. Functional Requirements

### Input Handling

- Upload **PDF, DOCX, TXT, or image files**.
- Extract text using appropriate parsers.
- Convert text into embeddings.

### Embedding & Storage

- Generate embeddings using Ollama-supported models or HuggingFace embeddings.
- Store embeddings in **ChromaDB collections** with metadata (filename, page, timestamp).

### Retrieval

- Perform **semantic search** over ChromaDB.
- Retrieve top-k relevant chunks with metadata.

### Response Generation

- Pass retrieved context to Ollama LLM via LangGraph orchestration.
- Stream responses to UI.
- Display **citations** (document name, page, snippet).

### Semantic Caching

- Before querying LLM, check if a **similar query** exists in cache.
- If found, return cached response with citations.
- If not cached, generate new response and store in cache.

### Agent Workflow (LangGraph)

- Define nodes for parsing, embedding, retrieval, and generation.
- Use conditional edges for decision-making (e.g., continue to tool node vs. end).
- Support streaming responses
