from datetime import timezone
from io import BytesIO

import fitz
from docx import Document
from PIL import Image
import pytest
from pytesseract import TesseractNotFoundError

from rag_app.ingestion import IngestionError, extract_document, normalize_text


def make_pdf() -> bytes:
    document = fitz.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "First PDF page")
    second_page = document.new_page()
    second_page.insert_text((72, 72), "Second PDF page")
    content = document.tobytes()
    document.close()
    return content


def make_docx() -> bytes:
    document = Document()
    document.add_paragraph("DOCX paragraph")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Table left"
    table.cell(0, 1).text = "Table right"
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_image() -> bytes:
    image = Image.new("RGB", (2, 2), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_extract_pdf_preserves_page_numbers_and_timestamp() -> None:
    document = extract_document("source.pdf", make_pdf())

    assert [page.text for page in document.pages] == ["First PDF page", "Second PDF page"]
    assert [page.source.page for page in document.pages] == [1, 2]
    assert all(page.source.filename == "source.pdf" for page in document.pages)
    assert all(page.source.ingested_at.tzinfo == timezone.utc for page in document.pages)


def test_extract_docx_includes_paragraphs_and_tables() -> None:
    document = extract_document("source.docx", make_docx())

    assert len(document.pages) == 1
    assert "DOCX paragraph" in document.pages[0].text
    assert "Table left | Table right" in document.pages[0].text
    assert document.pages[0].source.page is None


def test_extract_utf8_text_normalizes_whitespace() -> None:
    document = extract_document("source.txt", b"\xef\xbb\xbf Hello  world \r\n\r\n")

    assert document.pages[0].text == "Hello world"
    assert document.pages[0].source.media_type == "txt"


def test_extract_image_uses_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_app.ingestion.pytesseract.image_to_string", lambda image: "  OCR   result ")

    document = extract_document("source.png", make_image())

    assert document.pages[0].text == "OCR result"
    assert document.pages[0].source.media_type == "png"
    assert document.pages[0].source.page is None


def test_missing_tesseract_reports_installation_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_tesseract(image: Image.Image) -> str:
        raise TesseractNotFoundError()

    monkeypatch.setattr("rag_app.ingestion.pytesseract.image_to_string", missing_tesseract)

    with pytest.raises(IngestionError, match="Install Tesseract"):
        extract_document("source.png", make_image())


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("source.exe", b"data", "Unsupported file type"),
        ("source.txt", b"", "is empty"),
        ("source.txt", b"\xff", "UTF-8 encoded"),
        ("source.pdf", b"not a PDF", "PDF"),
        ("source.docx", b"not a DOCX", "DOCX"),
        ("source.png", b"not an image", "readable image"),
    ],
)
def test_invalid_files_raise_actionable_errors(filename: str, content: bytes, message: str) -> None:
    with pytest.raises(IngestionError, match=message):
        extract_document(filename, content)


def test_oversized_file_is_rejected() -> None:
    with pytest.raises(IngestionError, match="upload limit is 1 MB"):
        extract_document("source.txt", b"x" * (1024 * 1024 + 1), max_upload_size_mb=1)


def test_empty_extraction_is_reported() -> None:
    with pytest.raises(IngestionError, match="No text could be extracted"):
        extract_document("source.txt", b"  \r\n  ")


def test_normalize_text_preserves_paragraphs_and_normalizes_unicode() -> None:
    assert normalize_text(" Ａ  B\r\n\r\n\r\nC ") == "A B\n\nC"
