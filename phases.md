# Phased Implementation Plan

This plan turns the requirements in `requirements.md` into incremental, verifiable delivery phases. Each phase should leave the application in a runnable state and establish the behavior needed by the next phase.

## Phase 1 — Project foundation and configuration

- Confirm supported Python versions (minimum 3.10), dependency choices, and the initial Ollama model for both chat and embeddings.
- Define the application structure, dependency manifest, configuration approach, and local setup instructions.
- Add configuration for Ollama connectivity, model names, ChromaDB persistence, upload limits, retrieval settings, and cache behavior.
- Establish basic logging and clear startup/configuration errors.
- **Exit criteria:** A clean environment can install the declared dependencies, validate configuration, and start the application shell.

## Phase 2 — Document ingestion and text extraction

- Implement upload handling for PDF, DOCX, TXT, and supported image formats.
- Extract text with appropriate parsers: PDF page-aware extraction, DOCX and TXT extraction, and OCR for images.
- Normalize extracted text and preserve source metadata, including filename, page where available, and ingestion timestamp.
- Handle empty, unreadable, unsupported, and malformed files with actionable user-facing errors.
- **Exit criteria:** Each supported file type can be uploaded and yields inspectable extracted text and source metadata; failure cases are reported without stopping unrelated uploads.

## Phase 3 — Chunking, embeddings, and vector storage

- Split extracted text into retrieval-sized chunks while retaining document and page provenance.
- Generate embeddings using the configured Ollama-supported or HuggingFace embedding model.
- Persist chunks, embeddings, and metadata in ChromaDB, with a documented persistence and collection strategy.
- Define safe behavior for re-ingestion, duplicate uploads, and document removal so stored chunks do not become inconsistent.
- **Exit criteria:** Ingested chunks survive application restarts and can be traced back to their source document and page.

## Phase 4 — Semantic retrieval

- Implement top-k similarity search over the ChromaDB collection.
- Return relevant chunk text together with similarity information and citation metadata.
- Make retrieval settings configurable and handle an empty knowledge base or no sufficiently relevant results.
- **Exit criteria:** Representative queries return relevant, correctly attributed chunks; empty and low-relevance cases are explicit.

## Phase 5 — Grounded answer generation and citations

- Define the answer-generation prompt and context format so responses are grounded in retrieved material.
- Integrate the configured local Ollama chat model and pass retrieved context through the generation workflow.
- Format citations consistently with document name, page when available, and a supporting snippet.
- Define behavior when retrieval provides no useful evidence, including a clear response that the uploaded sources do not answer the question.
- **Exit criteria:** Answers use retrieved context and display verifiable citations; unsupported questions are not presented as source-grounded facts.

## Phase 6 — LangGraph workflow and streaming

- Represent the RAG lifecycle as a LangGraph workflow with clear state for the user query, retrieved context, generated answer, and citations.
- Add nodes for retrieval and generation, and include conditional routing for no-context or other supported outcomes.
- Stream generated response content through Ollama or LangGraph callbacks to the UI while preserving final citations and errors.
- Keep parsing and embedding in the ingestion path; connect them to the graph only where the chosen workflow requires it.
- **Exit criteria:** The graph handles successful, empty-retrieval, and model-error paths, and users see answer tokens as they are generated.

## Phase 7 — Streamlit user experience

- Build the Streamlit interface for document upload, ingestion status, asking questions, and viewing answers.
- Show streaming output, citations, and clear progress/error states.
- Expose essential configuration safely and avoid displaying secrets.
- Make document/source status understandable, including whether uploaded content has been indexed and is available for retrieval.
- **Exit criteria:** A user can upload supported files, wait for indexing, ask questions, and inspect cited streaming answers in one UI flow.

## Phase 8 — Semantic cache

- Select and document the cache implementation using ChromaDB metadata, including the similarity threshold and invalidation strategy.
- Check for semantically similar queries before invoking answer generation; on a cache hit, return the cached answer and its citations.
- Store successful generated answers with enough provenance to ensure citations remain valid.
- Avoid caching failures or answers whose source documents have changed or been removed.
- **Exit criteria:** Repeated or sufficiently similar queries can be served from cache, while cache misses generate and store fresh answers and document changes cannot return stale citations.

## Phase 9 — Validation, reliability, and release readiness

- Add focused automated tests for parsing, chunking, metadata, storage, retrieval, workflow routing, citation formatting, streaming, and cache hit/miss behavior.
- Test end-to-end flows using local fixtures and a controllable model boundary; keep tests independent of a live external API.
- Verify Ollama-unavailable, malformed-upload, empty-index, and ChromaDB persistence/restart behavior.
- Document installation, Ollama/model setup, configuration, supported files, limitations, and common troubleshooting steps.
- **Exit criteria:** The documented local setup works from a clean environment, core behavior is covered by tests, and expected failure modes are visible and recoverable.

## Suggested delivery order

Deliver Phases 1–5 first for a basic, testable RAG path; add LangGraph streaming and the UI in Phases 6–7; then add semantic caching and complete reliability work in Phases 8–9. Keep cache and optional fallback providers out of the initial critical path until retrieval, grounded answers, and citations are validated.
