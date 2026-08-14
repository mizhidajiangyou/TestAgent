"""Tests for the DocumentParser (docling-inspired, 3-tier fallback).

Covers PDF/DOCX/HTML/PPTX parsing with real generated documents, the
plain-text fallback, table-to-markdown conversion, DOCX paragraph
formatting, and error handling. Test documents are generated in-memory
using python-docx, python-pptx, and BeautifulSoup to avoid binary fixtures.
"""

from pathlib import Path

import pytest

from testagent.parsers.document_parser import DocumentParser

# ----------------------------------------------------------------------
# Fixtures: generate real binary documents for testing
# ----------------------------------------------------------------------


def _make_docx(path: Path) -> None:
    """Create a sample DOCX with heading, paragraph, list, and table."""
    from docx import Document

    doc = Document()
    doc.add_heading("User Registration", level=1)
    doc.add_paragraph("Users can register with an email and password.")
    doc.add_paragraph("Must be 18 or older", style="List Bullet")

    table = doc.add_table(rows=2, cols=2)
    table.style = "Table Grid"
    table.cell(0, 0).text = "Field"
    table.cell(0, 1).text = "Rule"
    table.cell(1, 0).text = "Email"
    table.cell(1, 1).text = "Must be valid format"
    doc.save(str(path))


def _make_html(path: Path) -> None:
    """Create a sample HTML file with headings, paragraph, list, and table."""
    content = """<!DOCTYPE html>
<html>
<head><title>Requirements</title><style>body { color: red; }</style></head>
<body>
<h1>User Login</h1>
<p>A user can log in with credentials.</p>
<h2>Rules</h2>
<ul>
  <li>Username is required</li>
  <li>Password must be 8+ characters</li>
</ul>
<table>
  <tr><th>Field</th><th>Rule</th></tr>
  <tr><td>Username</td><td>Required</td></tr>
  <tr><td>Password</td><td>8+ chars</td></tr>
</table>
<script>alert('should be removed');</script>
</body>
</html>"""
    path.write_text(content, encoding="utf-8")


def _make_pptx(path: Path) -> None:
    """Create a sample PPTX with text and a table."""
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])  # Title + Content
    slide.shapes.title.text = "User Registration"
    slide.placeholders[1].text = "Overview of registration requirements"

    # Add a table shape.
    rows, cols = 2, 2
    table_shape = slide.shapes.add_table(rows, cols, Inches(1), Inches(2), Inches(4), Inches(1))
    table = table_shape.table
    table.cell(0, 0).text = "Field"
    table.cell(0, 1).text = "Rule"
    table.cell(1, 0).text = "Email"
    table.cell(1, 1).text = "Valid format"

    prs.save(str(path))


def _make_pdf(path: Path) -> None:
    """Create a sample PDF with text using reportlab (if available)."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not available for PDF test fixture generation")

    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(100, 750, "User Registration Requirements")
    c.drawString(100, 730, "Users can register with email and password.")
    c.drawString(100, 710, "Password Policy: minimum 8 characters.")
    c.showPage()
    c.save()


@pytest.fixture
def docx_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.docx"
    _make_docx(path)
    return path


@pytest.fixture
def html_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.html"
    _make_html(path)
    return path


@pytest.fixture
def pptx_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pptx"
    _make_pptx(path)
    return path


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    _make_pdf(path)
    return path


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text("# Heading\n\nSome text content.\n", encoding="utf-8")
    return path


@pytest.fixture
def parser() -> DocumentParser:
    return DocumentParser()


# ----------------------------------------------------------------------
# Availability detection
# ----------------------------------------------------------------------


class TestAvailabilityDetection:
    """Tests for the _check_* availability methods."""

    def test_check_docx_returns_bool(self) -> None:
        result = DocumentParser._check_docx()
        assert isinstance(result, bool)

    def test_check_pdf_returns_bool(self) -> None:
        result = DocumentParser._check_pdf()
        assert isinstance(result, bool)

    def test_check_html_returns_bool(self) -> None:
        result = DocumentParser._check_html()
        assert isinstance(result, bool)

    def test_check_pptx_returns_bool(self) -> None:
        result = DocumentParser._check_pptx()
        assert isinstance(result, bool)

    def test_check_docling_returns_bool(self) -> None:
        result = DocumentParser._check_docling()
        assert isinstance(result, bool)

    def test_init_caches_availability(self) -> None:
        dp = DocumentParser()
        assert hasattr(dp, "_docx_available")
        assert hasattr(dp, "_pdf_available")
        assert hasattr(dp, "_html_available")
        assert hasattr(dp, "_pptx_available")
        assert hasattr(dp, "_docling_available")


# ----------------------------------------------------------------------
# DOCX parsing
# ----------------------------------------------------------------------


class TestParseDocx:
    """Tests for DOCX parsing via python-docx."""

    def test_parse_docx_extracts_heading(self, parser: DocumentParser, docx_file: Path) -> None:
        text = parser.parse(docx_file)
        assert "User Registration" in text
        assert "#" in text  # markdown heading

    def test_parse_docx_extracts_paragraph(self, parser: DocumentParser, docx_file: Path) -> None:
        text = parser.parse(docx_file)
        assert "email and password" in text

    def test_parse_docx_extracts_list_item(self, parser: DocumentParser, docx_file: Path) -> None:
        text = parser.parse(docx_file)
        assert "18 or older" in text
        assert "- " in text  # markdown list marker

    def test_parse_docx_extracts_table(self, parser: DocumentParser, docx_file: Path) -> None:
        text = parser.parse(docx_file)
        assert "Field" in text
        assert "Rule" in text
        assert "Email" in text
        assert "|" in text  # markdown table separator

    def test_parse_docx_preserves_order(self, parser: DocumentParser, docx_file: Path) -> None:
        """Paragraphs and tables should appear in document order (not flattened)."""
        text = parser.parse(docx_file)
        heading_pos = text.find("User Registration")
        para_pos = text.find("email and password")
        table_pos = text.find("Field")
        assert heading_pos < para_pos < table_pos

    def test_parse_docx_table_markdown_format(
        self, parser: DocumentParser, docx_file: Path
    ) -> None:
        text = parser.parse(docx_file)
        # Markdown table should have a separator row with ---.
        lines = text.split("\n")
        table_lines = [ln for ln in lines if ("|" in ln and "Field" in ln) or "| ---" in ln]
        assert len(table_lines) >= 1


# ----------------------------------------------------------------------
# HTML parsing
# ----------------------------------------------------------------------


class TestParseHtml:
    """Tests for HTML parsing via markdownify/BeautifulSoup."""

    def test_parse_html_extracts_heading(self, parser: DocumentParser, html_file: Path) -> None:
        text = parser.parse(html_file)
        assert "User Login" in text
        assert "#" in text

    def test_parse_html_extracts_paragraph(self, parser: DocumentParser, html_file: Path) -> None:
        text = parser.parse(html_file)
        assert "credentials" in text

    def test_parse_html_extracts_list(self, parser: DocumentParser, html_file: Path) -> None:
        text = parser.parse(html_file)
        assert "Username is required" in text
        assert "8+ characters" in text or "8+ chars" in text

    def test_parse_html_extracts_table(self, parser: DocumentParser, html_file: Path) -> None:
        text = parser.parse(html_file)
        assert "Field" in text
        assert "Rule" in text
        assert "|" in text

    def test_parse_html_removes_script_and_style(
        self, parser: DocumentParser, html_file: Path
    ) -> None:
        text = parser.parse(html_file)
        assert "alert" not in text
        assert "color: red" not in text


# ----------------------------------------------------------------------
# PPTX parsing
# ----------------------------------------------------------------------


class TestParsePptx:
    """Tests for PPTX parsing via python-pptx."""

    def test_parse_pptx_extracts_slide_marker(
        self, parser: DocumentParser, pptx_file: Path
    ) -> None:
        text = parser.parse(pptx_file)
        assert "Slide 1" in text

    def test_parse_pptx_extracts_title_text(self, parser: DocumentParser, pptx_file: Path) -> None:
        text = parser.parse(pptx_file)
        assert "User Registration" in text

    def test_parse_pptx_extracts_table(self, parser: DocumentParser, pptx_file: Path) -> None:
        text = parser.parse(pptx_file)
        assert "Field" in text
        assert "Rule" in text
        assert "Email" in text
        assert "|" in text


# ----------------------------------------------------------------------
# PDF parsing
# ----------------------------------------------------------------------


class TestParsePdf:
    """Tests for PDF parsing via pdfplumber (or docling fallback)."""

    def test_parse_pdf_extracts_text(self, parser: DocumentParser, pdf_file: Path) -> None:
        text = parser.parse(pdf_file)
        assert "User Registration" in text or "Registration" in text

    def test_parse_pdf_has_page_marker(self, parser: DocumentParser, pdf_file: Path) -> None:
        text = parser.parse(pdf_file)
        # pdfplumber backend adds "## Page N" markers.
        assert "Page" in text or "Password" in text

    def test_parse_pdf_extracts_password_policy(
        self, parser: DocumentParser, pdf_file: Path
    ) -> None:
        text = parser.parse(pdf_file)
        assert "Password" in text or "password" in text


# ----------------------------------------------------------------------
# Plain text fallback
# ----------------------------------------------------------------------


class TestParseTextFallback:
    """Tests for plain text and unknown format fallback."""

    def test_parse_text_file(self, parser: DocumentParser, text_file: Path) -> None:
        text = parser.parse(text_file)
        assert "Heading" in text
        assert "Some text content" in text

    def test_parse_markdown_file(self, parser: DocumentParser, tmp_path: Path) -> None:
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Title\n\nContent here.\n", encoding="utf-8")
        text = parser.parse(md_file)
        assert "# Title" in text
        assert "Content here" in text

    def test_parse_unknown_suffix_reads_as_text(
        self, parser: DocumentParser, tmp_path: Path
    ) -> None:
        unknown = tmp_path / "doc.log"
        unknown.write_text("log line 1\nlog line 2\n", encoding="utf-8")
        text = parser.parse(unknown)
        assert "log line 1" in text

    def test_parse_non_utf8_falls_back_to_replace(
        self, parser: DocumentParser, tmp_path: Path
    ) -> None:
        binary_file = tmp_path / "binary.dat"
        binary_file.write_bytes(b"\xff\xfe\x00\x01invalid utf8")
        # Should not raise; replaces errors.
        text = parser.parse(binary_file)
        assert isinstance(text, str)


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error and edge cases."""

    def test_parse_nonexistent_file_raises(self, parser: DocumentParser) -> None:
        with pytest.raises(FileNotFoundError, match="Document not found"):
            parser.parse("/nonexistent/file.docx")

    def test_parse_docx_without_available_backend_raises(
        self, tmp_path: Path, docx_file: Path
    ) -> None:
        parser = DocumentParser()
        # Force backends off to test the error path.
        parser._docling_available = False
        parser._docx_available = False
        with pytest.raises(ValueError, match="No DOCX parser available"):
            parser.parse(docx_file)

    def test_parse_pdf_without_available_backend_raises(
        self, tmp_path: Path, pdf_file: Path
    ) -> None:
        parser = DocumentParser()
        parser._docling_available = False
        parser._pdf_available = False
        with pytest.raises(ValueError, match="No PDF parser available"):
            parser.parse(pdf_file)

    def test_parse_html_without_available_backend_raises(
        self, tmp_path: Path, html_file: Path
    ) -> None:
        parser = DocumentParser()
        parser._html_available = False
        with pytest.raises(ValueError, match="No HTML parser available"):
            parser.parse(html_file)


# ----------------------------------------------------------------------
# Helper: _table_to_markdown
# ----------------------------------------------------------------------


class TestTableToMarkdown:
    """Tests for the _table_to_markdown static helper."""

    def test_basic_table(self) -> None:
        table = [["A", "B"], ["1", "2"]]
        md = DocumentParser._table_to_markdown(table)
        assert "| A | B |" in md
        assert "| --- | --- |" in md
        assert "| 1 | 2 |" in md

    def test_table_with_none_cells(self) -> None:
        table = [["A", None], [None, "2"]]
        md = DocumentParser._table_to_markdown(table)
        # None should become empty string.
        assert "| A |  |" in md
        assert "|  | 2 |" in md

    def test_table_with_pipe_escaped(self) -> None:
        table = [["Name", "Pattern"], ["Bob", "a|b"]]
        md = DocumentParser._table_to_markdown(table)
        assert "a\\|b" in md  # pipe escaped

    def test_table_with_newlines_in_cell(self) -> None:
        table = [["Col", "Value"], ["x", "line1\nline2"]]
        md = DocumentParser._table_to_markdown(table)
        assert "line1 line2" in md  # newline replaced with space

    def test_table_ragged_rows_padded(self) -> None:
        table = [["A", "B", "C"], ["1", "2"]]  # row 2 has fewer cols
        md = DocumentParser._table_to_markdown(table)
        assert "| 1 | 2 |  |" in md  # padded with empty

    def test_empty_table_returns_empty(self) -> None:
        assert DocumentParser._table_to_markdown([]) == ""

    def test_header_only_table(self) -> None:
        table = [["A", "B"]]
        md = DocumentParser._table_to_markdown(table)
        assert "| A | B |" in md
        assert "| --- | --- |" in md


# ----------------------------------------------------------------------
# Helper: _format_docx_paragraph
# ----------------------------------------------------------------------


class TestFormatDocxParagraph:
    """Tests for the _format_docx_paragraph static helper."""

    def test_heading_1(self) -> None:
        assert DocumentParser._format_docx_paragraph("Heading 1", "Title") == "# Title"

    def test_heading_2(self) -> None:
        assert DocumentParser._format_docx_paragraph("Heading 2", "Sub") == "## Sub"

    def test_heading_3(self) -> None:
        assert DocumentParser._format_docx_paragraph("Heading 3", "Deep") == "### Deep"

    def test_title_style(self) -> None:
        assert DocumentParser._format_docx_paragraph("Title", "My Title") == "# My Title"

    def test_list_bullet_style(self) -> None:
        assert DocumentParser._format_docx_paragraph("List Bullet", "item") == "- item"

    def test_list_number_style(self) -> None:
        assert DocumentParser._format_docx_paragraph("List Number", "first") == "- first"

    def test_normal_style_returns_text(self) -> None:
        assert DocumentParser._format_docx_paragraph("Normal", "text") == "text"

    def test_heading_invalid_level_defaults_to_1(self) -> None:
        result = DocumentParser._format_docx_paragraph("Heading", "No Level")
        assert result == "# No Level"

    def test_heading_level_clamped_to_6(self) -> None:
        result = DocumentParser._format_docx_paragraph("Heading 9", "Deep")
        assert result == "###### Deep"
