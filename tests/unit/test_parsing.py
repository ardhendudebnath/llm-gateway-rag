import pytest

from app.rag.parsing import (
    EmptyDocumentError,
    UnsupportedDocumentError,
    detect_kind,
    parse_document,
)
from tests.pdfgen import make_pdf


@pytest.mark.parametrize(
    ("filename", "content_type", "kind"),
    [
        ("notes.md", None, "text"),
        ("notes.txt", "application/octet-stream", "text"),  # extension wins over a generic type
        ("upload", "text/markdown; charset=utf-8", "text"),
        ("report.PDF", None, "pdf"),
        ("upload", "application/pdf", "pdf"),
    ],
)
def test_detect_kind(filename, content_type, kind):
    assert detect_kind(filename, content_type) == kind


def test_unsupported_type_is_rejected():
    with pytest.raises(UnsupportedDocumentError, match="unsupported"):
        detect_kind("slides.pptx", "application/vnd.ms-powerpoint")


def test_text_is_decoded_with_bom_and_normalised_newlines():
    [section] = parse_document("﻿line one\r\nline two\r".encode(), "a.txt", None)
    assert section.text == "line one\nline two\n"
    assert section.page is None


def test_non_utf8_text_falls_back_to_latin1():
    [section] = parse_document("café".encode("latin-1"), "a.txt", None)
    assert section.text == "café"


def test_binary_disguised_as_text_is_rejected():
    with pytest.raises(UnsupportedDocumentError, match="binary"):
        parse_document(b"PK\x03\x04\x14\x00\x06\x00", "notes.md", "text/markdown")


def test_pdf_pages_keep_their_page_numbers():
    sections = parse_document(make_pdf(["First page text", "Second page text"]), "a.pdf", None)
    assert [(s.page, s.text.strip()) for s in sections] == [
        (1, "First page text"),
        (2, "Second page text"),
    ]


def test_blank_pdf_pages_are_dropped():
    sections = parse_document(make_pdf(["Only text", ""]), "a.pdf", None)
    assert [s.page for s in sections] == [1]


def test_document_without_text_is_rejected():
    with pytest.raises(EmptyDocumentError):
        parse_document(b"  \n\t ", "empty.md", None)
    with pytest.raises(EmptyDocumentError):
        parse_document(make_pdf([""]), "scanned.pdf", None)


def test_corrupt_pdf_is_rejected():
    with pytest.raises(UnsupportedDocumentError, match="could not read PDF"):
        parse_document(b"%PDF-1.4 this is not really a pdf", "broken.pdf", None)
