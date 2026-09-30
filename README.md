# RAG App Demo

Local RAG application foundation built for Python 3.10+.

## Setup

1. Install Python 3.10 or newer and install Ollama for your operating system.
2. Create and activate a virtual environment:

   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the project and development dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   ```

4. Copy `.env.example` to `.env` and adjust values if needed:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Ensure Ollama is running, then download the models named in `.env`. If starting
   Ollama manually, run `ollama serve` in one terminal and the model commands in
   another:

   ```powershell
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

   If you changed `OLLAMA_CHAT_MODEL`, pull that model instead.

6. Validate configuration and prepare the ChromaDB directory:

   ```powershell
   python -m rag_app
   ```

7. Run the offline automated suite and start the application:

   ```powershell
   python -m pytest
   streamlit run streamlit_app.py
   ```

The example configuration uses `llama3.2:3b` chat and `nomic-embed-text` embeddings.
The UI accepts PDF, DOCX, UTF-8 TXT,
and PNG, JPEG, TIFF, BMP, GIF, or WebP images. PDFs retain 1-based page numbers;
DOCX, TXT, and image sources have no page number. Extracted text is normalized and
shown alongside its filename, media type, and UTC ingestion timestamp. Each upload
is processed independently, so a malformed file does not prevent other files from
being indexed.

Select **Index uploaded files** to extract, split, embed, and persist each document
in ChromaDB. Chunk size and overlap are character counts configured by `CHUNK_SIZE`
and `CHUNK_OVERLAP`. The persistent collection is named by
`CHROMA_COLLECTION_NAME` in `CHROMA_PERSIST_DIRECTORY`. Re-uploading a filename
(case-insensitive basename) replaces its prior chunks; files with different names
are kept as separate documents even if their contents match. Use the **Remove**
action to delete a document and all its chunks. These operations update one
collection; indexing computes embeddings before replacing stored chunks, so
embedding failures leave the prior version intact.

Use **Search indexed content** to run top-k semantic search. `RETRIEVAL_TOP_K`
controls the number of nearest chunks; `RETRIEVAL_MIN_SCORE` is the minimum
normalized relevance score (0 to 1) returned. New collections use cosine distance;
existing Phase 3 collections retain their L2 distance metric and are scored accordingly.
Search results show the text, similarity, distance,
filename, page where available, and chunk metadata. The UI distinguishes an empty
collection from a query with no matches above the configured relevance threshold.

Use **Ask a question** to run the LangGraph retrieval/generation workflow and stream
a response from the configured local Ollama chat model. The workflow routes empty
or low-relevance retrieval directly to the no-evidence response and surfaces model
or retrieval errors explicitly. The prompt instructs the model to use only the
retrieved excerpts, cite factual claims using markers such as `[S1]`, and say when
the excerpts do not support an answer. An empty index or query with no relevant
results returns a clear no-evidence message without calling the chat model. Source
citations are displayed with the filename, page when available, similarity, and a
supporting snippet. If Ollama is unavailable or the model returns an empty response,
the UI reports the generation error rather than presenting a fallback as an answer.

Image OCR requires the Tesseract executable in addition to the Python package. Install
Tesseract for your operating system (on Windows, install a Tesseract OCR distribution
and add its installation directory to `PATH`) and ensure `tesseract` is available there.
When it is missing, the affected image reports an actionable error while other files
continue processing.

The Streamlit interface groups the workflow into **Upload & index**, **Ask & search**,
and **Sources** tabs. Uploads show whether each file is awaiting indexing or is already
indexed; indexing reports per-file progress and errors and keeps extracted text
inspectable after completion. The Sources tab shows searchable chunk counts and allows
documents to be removed. The sidebar shows non-secret runtime settings; it does not
display environment variables or endpoint credentials. During questions, the interface
shows retrieval/generation progress, streams answer text, and then displays citations
or a clear no-evidence/error state.

Phase 8 uses a dedicated persistent ChromaDB collection (`CACHE_COLLECTION_NAME`)
for semantic query caching. Set `CACHE_ENABLED=true` to check the cache before
retrieval or answer generation. `CACHE_SIMILARITY_THRESHOLD` (0–1) controls cosine
similarity required for a hit. Only successfully generated answers with citations
are stored. Cache entries retain their answer, citation snippets, query, model and
retrieval settings, and source-document version timestamps. An entry is invalidated
when any cited source is changed or removed, or when the chat/embedding model or
retrieval settings differ. Cache entries that fail validation are discarded rather
than used; cache misses run the normal workflow and store successful results. A
cache write failure does not discard an already generated answer and is shown as a
warning. Disable caching with `CACHE_ENABLED=false`.

## Supported behavior and limitations

- PDFs are extracted page-by-page. Image-only/scanned PDFs are not OCR-processed;
  upload individual supported image files for OCR.
- DOCX, TXT, and image uploads are represented as a single source without page
  numbers. Page numbers are available for PDF sources.
- Text splitting uses character counts rather than model token counts.
- Ollama, ChromaDB, and document data run locally. Ollama must be available for
  indexing, uncached queries, and cache similarity checks.
- The UI provides streaming answers, citations, upload status, and source management;
  it does not provide authentication or multi-user isolation.
- OpenAI/HuggingFace fallbacks and semantic cache eviction/size limits are not
  implemented.

## Troubleshooting

| Symptom | Check / resolution |
| --- | --- |
| `Could not generate embeddings` or connection refused | Start Ollama (`ollama serve`), verify `OLLAMA_BASE_URL`, and confirm the configured embedding model is installed with `ollama list` or `ollama pull <model>`. |
| Chat generation reports a missing model | Check `OLLAMA_CHAT_MODEL` in `.env` and run `ollama pull <configured-model>`. |
| Tesseract OCR is not installed or is not on PATH | Install the Tesseract executable, add its directory to `PATH`, restart the terminal/app, and verify `tesseract --version`. |
| Unsupported, malformed, empty, or unreadable upload | Use PDF, DOCX, UTF-8 TXT, PNG, JPEG, TIFF, BMP, GIF, or WebP. Check the file opens correctly and is within `MAX_UPLOAD_SIZE_MB`. Each upload reports errors independently. |
| Search says there are no indexed documents | Confirm **Index uploaded files** completed successfully and check the displayed indexed-document/chunk counts and `CHROMA_PERSIST_DIRECTORY`. |
| Search finds no sufficiently relevant chunks | Try terms present in the documents or adjust `RETRIEVAL_MIN_SCORE` (0–1) and `RETRIEVAL_TOP_K`, then restart the app. |
| Cached answer is not reused | Confirm `CACHE_ENABLED=true`, check `CACHE_SIMILARITY_THRESHOLD`, and note cache entries are invalidated when source versions, models, or retrieval settings change. |
| Configuration validation fails | Compare `.env` values with `.env.example`; numeric settings must be in range and `CHUNK_OVERLAP` must be smaller than `CHUNK_SIZE`. |
| Tests try to access an external service | The test suite uses local fixtures and fake embedding/chat boundaries; run it with `python -m pytest` from the project root. No running Ollama server is required. |

## Validation

Run all automated tests from the repository root:

```powershell
python -m pytest
```

The suite covers parsers and malformed inputs, chunking and provenance, ChromaDB
persistence/restart, retrieval thresholds and empty indexes, citation formatting,
workflow routing and token streaming, semantic cache hit/miss and invalidation, and
offline end-to-end behavior. Live Ollama model availability is environment-specific;
network failure paths are tested with controlled errors instead.

## Configuration

All settings can be supplied as environment variables or in `.env`. Invalid values
produce an actionable startup error. The available settings are documented in
`.env.example`; secrets should not be committed. For any non-default model or
collection, make sure the local Ollama model and persistent ChromaDB collection
settings match your intended runtime.
