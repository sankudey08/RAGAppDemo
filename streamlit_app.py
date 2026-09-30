"""Streamlit user experience for document ingestion and grounded RAG questions."""

import logging
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from rag_app.config import Settings
from rag_app.ingestion import IngestionError, extract_document
from rag_app.logging_config import configure_logging
from rag_app.startup import prepare_runtime
from rag_app.vector_store import DocumentVectorStore, EmbeddingError, RetrievalResult, StorageError
from rag_app.workflow import RAGWorkflow, WorkflowResult


logger = logging.getLogger(__name__)
SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "txt", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp"]


def _safe_ollama_url(settings: Settings) -> str:
    """Display the Ollama endpoint without any configured credentials."""

    url = settings.ollama_base_url
    host = f"[{url.host}]" if ":" in url.host else url.host
    port = f":{url.port}" if url.port is not None else ""
    path = url.path.rstrip("/")
    return f"{url.scheme}://{host}{port}{path}"


def _format_timestamp(value: str) -> str:
    """Render an ISO 8601 timestamp in a friendly, human-readable form."""

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.strftime("%b %d, %Y at %H:%M")


def _humanize_snippet(text: str) -> str:
    """Collapse raw extraction whitespace/line breaks into readable paragraph text."""

    normalized = re.sub(r"\s*\n\s*", " ", text.strip())
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized


def _render_search_result(index: int, result: RetrievalResult) -> None:
    """Render a single semantic-search match as a readable, self-contained card."""

    location = f" · page {result.page}" if result.page is not None else ""
    with st.container(border=True):
        title_col, score_col = st.columns([4, 1])
        title_col.markdown(f"**{index}. {result.filename}**{location}")
        score_col.metric("Match", f"{result.similarity * 100:.0f}%")
        st.caption(f"{result.media_type.upper()} · indexed {_format_timestamp(result.ingested_at)}")
        st.markdown(f"> {_humanize_snippet(result.text)}")


st.set_page_config(page_title="Chat with Document", page_icon="📚", layout="wide")
st.caption("Upload documents, index their content, and ask questions grounded in your sources.")

try:
    settings = Settings()
    configure_logging(settings)
    prepare_runtime(settings)
except ValidationError as error:
    st.error("Configuration is invalid. Update your .env file and restart the application.")
    st.code(str(error))
    st.stop()
except OSError as error:
    st.error(f"Unable to prepare the application runtime: {error}")
    st.stop()

with st.sidebar:
    st.header("Chat with Document")

    with st.expander("⚙️ Configuration", expanded=False):
        ollama_url = _safe_ollama_url(settings)
        st.caption("Non-secret runtime settings")
        st.write(f"**Ollama:** `{ollama_url}`")
        st.write(f"**Chat model:** `{settings.ollama_chat_model}`")
        st.write(f"**Embedding model:** `{settings.ollama_embedding_model}`")
        st.write(f"**Chroma collection:** `{settings.chroma_collection_name}`")
        st.write(f"**Semantic cache:** {'enabled' if settings.cache_enabled else 'disabled'}")
        if settings.cache_enabled:
            st.write(f"**Cache threshold:** {settings.cache_similarity_threshold:.2f}")
        st.write(f"**Upload limit:** {settings.max_upload_size_mb} MB per file")
        st.write(f"**Retrieval:** top {settings.retrieval_top_k}, score ≥ {settings.retrieval_min_score:.2f}")
        st.caption("Secrets and environment-variable contents are not displayed.")

try:
    vector_store = DocumentVectorStore(settings)
except StorageError as error:
    logger.exception("Could not initialize persistent document storage")
    st.error(str(error))
    st.stop()

try:
    stored_documents = vector_store.list_documents()
except StorageError as error:
    logger.exception("Could not load the indexed-document list")
    st.error(f"Could not load document status: {error}")
    st.stop()

with st.sidebar:
    with st.expander("📤 Upload & index documents", expanded=True):
        st.caption(
            "Supported: PDF, DOCX, UTF-8 TXT, PNG, JPEG, TIFF, BMP, GIF, and WebP. "
            "Image OCR requires Tesseract to be installed."
        )
        uploads = st.file_uploader(
            "Choose one or more files",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            max_upload_size=settings.max_upload_size_mb,
            help=f"Maximum size per file: {settings.max_upload_size_mb} MB.",
            label_visibility="collapsed",
        )

        if uploads:
            indexed_by_name = {document.filename.casefold(): document for document in stored_documents}
            for upload in uploads:
                existing = indexed_by_name.get(Path(upload.name).name.casefold())
                if existing:
                    st.info(
                        f"**{upload.name}** — already indexed ({existing.chunk_count} chunks); "
                        "indexing again replaces this file's stored chunks."
                    )
                else:
                    st.warning(f"**{upload.name}** — uploaded, not indexed yet.")

            if st.button("Index uploaded files", type="primary", use_container_width=True):
                progress = st.progress(0, text="Starting ingestion...")
                ingestion_report: list[dict[str, object]] = []
                for upload_index, upload in enumerate(uploads, start=1):
                    progress.progress(
                        (upload_index - 1) / len(uploads),
                        text=f"Extracting and indexing {upload.name}...",
                    )
                    try:
                        extracted = extract_document(
                            upload.name,
                            upload.getvalue(),
                            max_upload_size_mb=settings.max_upload_size_mb,
                        )
                        stored = vector_store.index_document(extracted)
                    except (IngestionError, EmbeddingError, StorageError) as error:
                        logger.warning("Could not index uploaded file %s: %s", upload.name, error)
                        ingestion_report.append(
                            {"filename": upload.name, "error": str(error), "pages": []}
                        )
                    else:
                        ingestion_report.append(
                            {
                                "filename": stored.filename,
                                "chunk_count": stored.chunk_count,
                                "pages": [
                                    {
                                        "text": page.text,
                                        "page": page.source.page,
                                        "media_type": page.source.media_type,
                                        "ingested_at": page.source.ingested_at.isoformat(),
                                    }
                                    for page in extracted.pages
                                ],
                            }
                        )
                    progress.progress(
                        upload_index / len(uploads),
                        text=f"Processed {upload_index} of {len(uploads)} file(s).",
                    )
                progress.empty()
                st.session_state["last_ingestion_report"] = ingestion_report
                st.rerun()

        for report_index, report in enumerate(st.session_state.get("last_ingestion_report", [])):
            if report.get("error"):
                st.error(f"**{report['filename']}** — {report['error']}")
                continue
            st.success(f"**{report['filename']}** — indexed {report['chunk_count']} chunk(s).")
            if st.toggle("Show extracted text", key=f"show-preview-{report_index}-{report['filename']}"):
                for page in report.get("pages", []):
                    page_label = f"Page {page['page']}" if page["page"] is not None else "Text"
                    st.markdown(f"**{page_label}**")
                    st.caption(f"{page['media_type']} · Ingested {page['ingested_at']}")
                    st.text(page["text"])
        if "last_ingestion_report" in st.session_state:
            st.caption("Latest ingestion results are shown above. Selecting Index again replaces this report.")
        else:
            st.caption("Choose files above to see their upload/index status and start ingestion.")

with st.sidebar:
    with st.expander("📊 Knowledge base status", expanded=True):
        indexed_count = len(stored_documents)
        chunk_count = sum(document.chunk_count for document in stored_documents)
        st.metric("Indexed documents", indexed_count)
        st.metric("Searchable chunks", chunk_count)
        st.metric("Collection", settings.chroma_collection_name)

def _render_citations(citations: list[dict[str, object]]) -> None:
    """Render a response's source citations collapsed inside a single expander link."""

    with st.expander(f"📎 View sources ({len(citations)})"):
        for citation in citations:
            location = f", page {citation['page']}" if citation.get("page") is not None else ""
            st.markdown(
                f"**[{citation['reference']}] {citation['filename']}{location}** "
                f"· similarity {citation['similarity']:.3f}"
            )
            st.caption(citation["snippet"])


tab_ask, tab_sources = st.tabs(["Ask & search", "Sources"])

with tab_ask:
    header_col, clear_col = st.columns([5, 1])
    header_col.subheader("Chat with your sources")
    if not stored_documents:
        st.info("The knowledge base is empty. Upload and index documents from the sidebar before asking questions.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    if st.session_state["chat_messages"] and clear_col.button("Clear chat", use_container_width=True):
        st.session_state["chat_messages"] = []
        st.rerun()

    chat_container = st.container(height=480, border=True)
    with chat_container:
        for message in st.session_state["chat_messages"]:
            avatar = "🧑" if message["role"] == "user" else "📚"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message.get("cache_warning"):
                    st.warning(f"Answer is available, but could not be cached: {message['cache_warning']}")
                if message.get("citations"):
                    _render_citations(message["citations"])

    question = st.chat_input("Ask a question about the information in your documents...")

    if question:
        st.session_state["chat_messages"].append({"role": "user", "content": question})

        with chat_container:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="📚"):
                answer_area = st.empty()
                status_area = st.status("Searching indexed sources...", expanded=True)
                streamed_answer = ""
                final_result: WorkflowResult | None = None
                assistant_message: dict[str, object] = {"role": "assistant", "content": ""}
                try:
                    workflow = RAGWorkflow(settings, vector_store)
                    for event in workflow.stream(question):
                        if event.kind == "token":
                            status_area.update(label="Generating answer from retrieved sources...")
                            streamed_answer += event.token
                            answer_area.markdown(streamed_answer)
                        elif event.kind == "complete":
                            final_result = event.result
                except (EmbeddingError, StorageError, ValueError) as error:
                    logger.exception("RAG request failed")
                    status_area.update(label="Request failed", state="error")
                    answer_area.empty()
                    error_text = f"Could not complete the request: {error}"
                    st.error(error_text)
                    assistant_message["content"] = error_text
                else:
                    if final_result is None:
                        status_area.update(label="Workflow returned no result", state="error")
                        answer_area.empty()
                        error_text = "The workflow completed without a final result."
                        st.error(error_text)
                        assistant_message["content"] = error_text
                    elif final_result.status == "error":
                        status_area.update(label="Request failed", state="error")
                        answer_area.empty()
                        error_text = f"The RAG workflow failed: {final_result.error}"
                        st.error(error_text)
                        assistant_message["content"] = error_text
                    else:
                        if final_result.status == "cache_hit":
                            status_area.update(label="Semantic cache hit", state="complete", expanded=False)
                            answer_area.markdown(final_result.answer)
                            st.caption("Returned a cached answer because a sufficiently similar query was found.")
                        elif final_result.status == "answered":
                            status_area.update(label="Answer complete", state="complete", expanded=False)
                            answer_area.markdown(final_result.answer)
                        else:
                            status_area.update(label="No relevant source evidence", state="complete", expanded=False)
                            answer_area.info(final_result.answer)

                        assistant_message["content"] = final_result.answer
                        if final_result.cache_warning:
                            st.warning(
                                f"Answer is available, but could not be cached: {final_result.cache_warning}"
                            )
                            assistant_message["cache_warning"] = final_result.cache_warning
                        if final_result.citations:
                            citations_payload = [
                                {
                                    "reference": citation.reference,
                                    "filename": citation.filename,
                                    "page": citation.page,
                                    "similarity": citation.similarity,
                                    "snippet": citation.snippet,
                                }
                                for citation in final_result.citations
                            ]
                            _render_citations(citations_payload)
                            assistant_message["citations"] = citations_payload

                st.session_state["chat_messages"].append(assistant_message)

    with st.expander("🔍 Advanced: semantic search only (no answer generation)"):
        with st.form("search-form"):
            search_query = st.text_input("Semantic search", placeholder="Search for a phrase or topic")
            search_submitted = st.form_submit_button("Search sources")

        if search_submitted and not search_query.strip():
            st.warning("Enter a search phrase before searching.")
        elif search_submitted:
            with st.status("Searching the indexed collection...", expanded=False) as search_status:
                try:
                    retrieval = vector_store.search(search_query)
                except (EmbeddingError, StorageError, ValueError) as error:
                    logger.exception("Semantic search failed")
                    search_status.update(label="Search failed", state="error")
                    st.error(f"Search failed: {error}")
                else:
                    search_status.update(label="Search complete", state="complete")
                    if retrieval.status == "empty_collection":
                        st.info("No documents are indexed yet. Upload and index documents before searching.")
                    elif retrieval.status == "no_relevant_results":
                        st.info("No sufficiently relevant chunks were found. Try another search.")
                    else:
                        for index, result in enumerate(retrieval.results, start=1):
                            _render_search_result(index, result)

with tab_sources:
    st.subheader("Indexed sources")
    if not stored_documents:
        st.info("No indexed sources yet.")
    else:
        st.caption("Indexed content is persisted locally and available for semantic search.")
        for document in stored_documents:
            with st.container(border=True):
                details, actions = st.columns([5, 1])
                with details:
                    st.markdown(f"**{document.filename}**")
                    st.caption(
                        f"{document.chunk_count} searchable chunk(s) · "
                        f"indexed {document.ingested_at}"
                    )
                with actions:
                    if st.button("Remove", key=f"remove-{document.document_id}", type="secondary"):
                        try:
                            vector_store.delete_document(document.document_id)
                        except StorageError as error:
                            logger.exception("Could not remove document %s", document.filename)
                            st.error(f"Could not remove {document.filename}: {error}")
                        else:
                            st.success(f"Removed {document.filename} and its indexed chunks.")
                            st.rerun()
