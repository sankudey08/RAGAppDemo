"""File validation, text extraction, and source metadata for uploaded documents."""

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re
import unicodedata
from zipfile import BadZipFile

import fitz
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from lxml.etree import XMLSyntaxError
from PIL import Image, UnidentifiedImageError
import pytesseract
from pytesseract import TesseractNotFoundError


SUPPORTED_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
)
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"})


class IngestionError(ValueError):
    """An actionable error for a single file that could not be ingested."""


@dataclass(frozen=True)
class SourceMetadata:
    """Origin information retained for every extracted text page."""

    filename: str
    page: int | None
    ingested_at: datetime
    media_type: str


@dataclass(frozen=True)
class ExtractedPage:
    """Normalized text and provenance for a page or page-like source."""

    text: str
    source: SourceMetadata


@dataclass(frozen=True)
class ExtractedDocument:
    """All non-empty extracted pages from one uploaded file."""

    filename: str
    ingested_at: datetime
    pages: tuple[ExtractedPage, ...]


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""

    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in normalized.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def extract_document(
    filename: str,
    content: bytes,
    *,
    max_upload_size_mb: int = 25,
) -> ExtractedDocument:
    """Extract text and provenance from supported document bytes.

    Raises ``IngestionError`` for unsupported, oversized, unreadable, malformed,
    or text-empty files, allowing callers to continue processing other uploads.
    """

    if not filename or not filename.strip():
        raise IngestionError("The uploaded file has no filename.")

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestionError(f"Unsupported file type '{extension or '(none)'}'. Supported types: {supported}.")

    max_bytes = max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise IngestionError(
            f"'{filename}' is {len(content) / (1024 * 1024):.1f} MB; "
            f"the upload limit is {max_upload_size_mb} MB."
        )
    if not content:
        raise IngestionError(f"'{filename}' is empty.")

    ingested_at = datetime.now(timezone.utc)
    try:
        if extension == ".pdf":
            extracted_pages = _extract_pdf(content)
        elif extension == ".docx":
            extracted_pages = _extract_docx(content)
        elif extension == ".txt":
            extracted_pages = _extract_txt(content)
        else:
            extracted_pages = _extract_image(content, filename)
    except IngestionError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise IngestionError(f"Could not read '{filename}': {error}") from error

    pages = tuple(
        ExtractedPage(
            text=text,
            source=SourceMetadata(
                filename=Path(filename).name,
                page=page_number,
                ingested_at=ingested_at,
                media_type=extension.lstrip("."),
            ),
        )
        for page_number, raw_text in extracted_pages
        if (text := normalize_text(raw_text))
    )
    if not pages:
        raise IngestionError(
            f"No text could be extracted from '{filename}'. "
            "Check that the file contains readable text; images require Tesseract OCR."
        )

    return ExtractedDocument(filename=Path(filename).name, ingested_at=ingested_at, pages=pages)


def _extract_pdf(content: bytes) -> list[tuple[int | None, str]]:
    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            return [(index + 1, page.get_text("text")) for index, page in enumerate(pdf)]
    except (fitz.FileDataError, fitz.EmptyFileError) as error:
        raise IngestionError(f"Malformed or unreadable PDF: {error}") from error


def _extract_docx(content: bytes) -> list[tuple[int | None, str]]:
    try:
        document = Document(BytesIO(content))
    except (BadZipFile, KeyError, OSError, PackageNotFoundError, XMLSyntaxError, ValueError) as error:
        raise IngestionError(f"Malformed or unreadable DOCX file: {error}") from error

    blocks: list[str] = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        blocks.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
    return [(None, "\n".join(blocks))]


def _extract_txt(content: bytes) -> list[tuple[int | None, str]]:
    try:
        return [(None, content.decode("utf-8-sig"))]
    except UnicodeDecodeError as error:
        raise IngestionError("Text files must be UTF-8 encoded. Save the file as UTF-8 and try again.") from error


def _extract_image(content: bytes, filename: str) -> list[tuple[int | None, str]]:
    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            text = pytesseract.image_to_string(image)
    except TesseractNotFoundError as error:
        raise IngestionError(
            "Tesseract OCR is not installed or is not on PATH. Install Tesseract and try again."
        ) from error
    except UnidentifiedImageError as error:
        raise IngestionError(f"'{filename}' is not a readable image.") from error
    except OSError as error:
        raise IngestionError(f"'{filename}' is a malformed or unreadable image: {error}") from error
    return [(None, text)]
