"""Turn an uploaded file into plain-text sections, one per page for PDFs.

Keeping page boundaries lets every chunk carry its page number, so answers can cite "p. 3".
"""

import io
from dataclasses import dataclass
from pathlib import PurePath

TEXT_TYPES = {"text/plain", "text/markdown", "text/x-markdown"}
PDF_TYPE = "application/pdf"
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


class UnsupportedDocumentError(ValueError):
    """The file type isn't one we can extract text from."""


class EmptyDocumentError(ValueError):
    """The file parsed, but contains no extractable text (e.g. a scanned PDF without OCR)."""


@dataclass(frozen=True)
class Section:
    text: str
    page: int | None = None  # 1-based; None for formats without pages


def detect_kind(filename: str, content_type: str | None) -> str:
    """Return "pdf" or "text". The extension wins over a generic/missing content type."""
    suffix = PurePath(filename or "").suffix.lower()
    ctype = (content_type or "").split(";")[0].strip().lower()
    if suffix == ".pdf" or ctype == PDF_TYPE:
        return "pdf"
    if suffix in _TEXT_SUFFIXES or ctype in TEXT_TYPES:
        return "text"
    raise UnsupportedDocumentError(
        f"unsupported document type (filename={filename!r}, content_type={content_type!r}); "
        "upload PDF, plain text or Markdown"
    )


def parse_document(data: bytes, filename: str, content_type: str | None) -> list[Section]:
    kind = detect_kind(filename, content_type)
    sections = _parse_pdf(data) if kind == "pdf" else [Section(_decode_text(data))]
    sections = [s for s in sections if s.text.strip()]
    if not sections:
        raise EmptyDocumentError("document contains no extractable text")
    return sections


def _decode_text(data: bytes) -> str:
    # Real text never contains NUL bytes; Office files, images and archives almost always do.
    # This catches binaries uploaded with a text extension or content type.
    if b"\x00" in data[:8192]:
        raise UnsupportedDocumentError("file looks binary, not text; upload PDF, text or Markdown")
    try:
        text = data.decode("utf-8-sig")  # also strips a BOM
    except UnicodeDecodeError:
        text = data.decode("latin-1")  # never fails; better than rejecting a legacy file
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _parse_pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        return [
            Section(page.extract_text() or "", page=i) for i, page in enumerate(reader.pages, 1)
        ]
    except PdfReadError as e:
        raise UnsupportedDocumentError(f"could not read PDF: {e}") from e
