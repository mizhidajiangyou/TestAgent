"""
Binary document parser for PDF, DOCX, HTML, PPTX and other formats.

Inspired by docling's ``DocumentConverter`` pattern:
- Tries docling first (best quality: layout analysis, OCR, table structure).
- Falls back to format-specific libraries (python-docx, pdfplumber,
  beautifulsoup4/markdownify, python-pptx).
- Falls back to plain text as a last resort.
- Output is always markdown-formatted text preserving headings, tables and lists.

This parser is a utility that returns text; it does not inherit from
:class:`BaseParser` because it does not produce structured items.
"""

import logging
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parse binary document formats (PDF, DOCX, HTML, PPTX) into structured text.

    Inspired by docling's ``DocumentConverter`` pattern:
    - Tries docling first (best quality: layout analysis, OCR, table structure).
    - Falls back to python-docx for DOCX, pdfplumber for PDF.
    - Falls back to plain text as last resort.
    - Output is always markdown-formatted text preserving structure.
    """

    def __init__(self) -> None:
        self._docling_available = self._check_docling()
        self._docx_available = self._check_docx()
        self._pdf_available = self._check_pdf()
        self._html_available = self._check_html()
        self._pptx_available = self._check_pptx()
        # Lazily created docling converter (reused across calls for performance).
        self._docling_converter: Any = None

        if self._docling_available:
            logger.debug("DocumentParser: docling available (preferred backend)")
        else:
            logger.debug("DocumentParser: docling not available; using format-specific backends")

    def parse(self, source: str | Path) -> str:
        """Parse a document file and return markdown-formatted text.

        Args:
            source: Path to the document file (.pdf, .docx, .doc, .html, .pptx).

        Returns:
            Markdown-formatted text preserving headings, tables, and lists.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the format is unsupported and cannot be read as text.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {source}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(path)
        elif suffix in (".docx", ".doc"):
            return self._parse_docx(path)
        elif suffix in (".html", ".htm"):
            return self._parse_html(path)
        elif suffix == ".pptx":
            return self._parse_pptx(path)
        else:
            # Fallback: read as text for unknown/lightweight formats (.txt, .md, ...).
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("Could not decode %s as UTF-8; replacing errors", path)
                return path.read_text(encoding="utf-8", errors="replace")

    def _parse_pdf(self, path: Path) -> str:
        """Parse PDF file. Tries docling first, then pdfplumber."""
        if self._docling_available:
            try:
                return self._parse_with_docling(path)
            except Exception as e:
                logger.warning("docling failed for %s: %s, falling back to pdfplumber", path, e)

        if self._pdf_available:
            return self._parse_pdf_with_pdfplumber(path)

        raise ValueError(
            "No PDF parser available. Install docling or pdfplumber: pip install docling pdfplumber"
        )

    def _parse_docx(self, path: Path) -> str:
        """Parse DOCX file. Tries docling first, then python-docx."""
        if self._docling_available:
            try:
                return self._parse_with_docling(path)
            except Exception as e:
                logger.warning("docling failed for %s: %s, falling back to python-docx", path, e)

        if self._docx_available:
            return self._parse_docx_with_python_docx(path)

        raise ValueError(
            "No DOCX parser available. Install docling or python-docx: "
            "pip install docling python-docx"
        )

    def _parse_html(self, path: Path) -> str:
        """Parse HTML file. Prefers markdownify, falls back to BeautifulSoup."""
        if self._html_available:
            return self._parse_html_structured(path)

        raise ValueError(
            "No HTML parser available. Install beautifulsoup4 or markdownify: "
            "pip install beautifulsoup4 markdownify"
        )

    def _parse_pptx(self, path: Path) -> str:
        """Parse PPTX file using python-pptx."""
        if self._pptx_available:
            return self._parse_pptx_with_python_pptx(path)

        raise ValueError("No PPTX parser available. Install python-pptx: pip install python-pptx")

    def _parse_with_docling(self, path: Path) -> str:
        """Parse using docling's DocumentConverter."""
        from docling.document_converter import DocumentConverter

        if self._docling_converter is None:
            self._docling_converter = DocumentConverter()
        converter = self._docling_converter
        result = converter.convert(str(path))
        return cast("str", result.document.export_to_markdown())

    def _parse_pdf_with_pdfplumber(self, path: Path) -> str:
        """Parse PDF using pdfplumber. Extracts text and tables."""
        import pdfplumber

        pages_text: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text() or ""
                if text:
                    pages_text.append(f"## Page {page_num}\n\n{text}")

                # Extract tables
                tables = page.extract_tables() or []
                for table_idx, table in enumerate(tables):
                    if table:
                        md_table = self._table_to_markdown(table)
                        pages_text.append(f"\n\n**Table {table_idx + 1}:**\n\n{md_table}")
        return "\n\n".join(pages_text) if pages_text else ""

    def _parse_docx_with_python_docx(self, path: Path) -> str:
        """Parse DOCX using python-docx. Preserves headings, paragraphs, and tables.

        Iterates ``doc.element.body`` children so paragraphs and tables stay in
        document order (interleaved), rather than the separate ``doc.paragraphs``
        / ``doc.tables`` lists which flatten structure.
        """
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = Document(str(path))
        parts: list[str] = []

        for element in doc.element.body:
            tag = element.tag
            if tag.endswith("}p"):
                para = Paragraph(element, doc)
                text = para.text.strip()
                if not text:
                    continue
                style_name = para.style.name if para.style else ""
                formatted = self._format_docx_paragraph(style_name, text)
                if formatted:
                    parts.append(formatted)
            elif tag.endswith("}tbl"):
                table = Table(element, doc)
                rows_data: list[list[str | None]] = []
                for row in table.rows:
                    rows_data.append([cell.text for cell in row.cells])
                if rows_data:
                    parts.append(self._table_to_markdown(rows_data))

        return "\n\n".join(parts)

    def _parse_html_structured(self, path: Path) -> str:
        """Parse HTML, preferring markdownify, falling back to BeautifulSoup."""
        content = path.read_text(encoding="utf-8")

        # Preferred: markdownify produces clean markdown directly.
        try:
            from markdownify import markdownify as md

            return md(content, heading_style="ATX")
        except ImportError:
            pass

        # Fallback: BeautifulSoup-based structured extraction.
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        parts: list[str] = []
        heading_levels = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
        for el in soup.find_all([*heading_levels, "p", "li", "table"]):
            # Skip block elements nested inside a table; the table branch handles them.
            if el.name != "table" and el.find_parent("table") is not None:
                continue
            name = el.name
            if name == "table":
                rows = el.find_all("tr")
                table_data: list[list[str | None]] = []
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    table_data.append([c.get_text(" ", strip=True) for c in cells])
                if table_data:
                    parts.append(self._table_to_markdown(table_data))
                continue
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            if name in heading_levels:
                parts.append(f"{'#' * heading_levels[name]} {text}")
            elif name == "li":
                parts.append(f"- {text}")
            else:
                parts.append(text)
        return "\n\n".join(parts)

    def _parse_pptx_with_python_pptx(self, path: Path) -> str:
        """Parse PPTX using python-pptx. Extracts text frames and tables."""
        from pptx import Presentation

        prs = Presentation(str(path))
        parts: list[str] = []
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_parts: list[str] = [f"## Slide {slide_num}"]
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            slide_parts.append(text)
                if shape.has_table:
                    table = shape.table
                    rows_data: list[list[str | None]] = []
                    for row in table.rows:
                        rows_data.append([cell.text for cell in row.cells])
                    if rows_data:
                        slide_parts.append(self._table_to_markdown(rows_data))
            parts.append("\n\n".join(slide_parts))
        return "\n\n".join(parts)

    @staticmethod
    def _format_docx_paragraph(style_name: str, text: str) -> str:
        """Format a DOCX paragraph as markdown based on its style name."""
        style_lower = style_name.lower()
        if style_lower.startswith("heading"):
            parts = style_lower.split()
            try:
                level = int(parts[-1])
            except (ValueError, IndexError):
                level = 1
            level = max(1, min(level, 6))
            return f"{'#' * level} {text}"
        if style_lower == "title":
            return f"# {text}"
        if style_lower.startswith("list"):
            return f"- {text}"
        return text

    @staticmethod
    def _table_to_markdown(table: list[list[str | None]]) -> str:
        """Convert a table (list of rows) to markdown format.

        Handles ``None`` cells, escapes pipe characters, and pads ragged rows to
        match the header width.
        """
        if not table:
            return ""
        # First row is header
        header = table[0]
        rows = table[1:] if len(table) > 1 else []
        width = len(header)

        def clean(cell: str | None) -> str:
            return (cell or "").replace("|", "\\|").replace("\n", " ").strip()

        md = "| " + " | ".join(clean(c) for c in header) + " |\n"
        md += "| " + " | ".join("---" for _ in header) + " |\n"
        for row in rows:
            # Pad row to match header length (handles ragged tables).
            padded = list(row) + [""] * (width - len(row))
            md += "| " + " | ".join(clean(c) for c in padded[:width]) + " |\n"
        return md

    @staticmethod
    def _check_docling() -> bool:
        """Check if docling is available."""
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _check_docx() -> bool:
        """Check if python-docx is available."""
        try:
            from docx import Document  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _check_pdf() -> bool:
        """Check if pdfplumber is available."""
        try:
            import pdfplumber  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _check_html() -> bool:
        """Check if an HTML parser (beautifulsoup4) is available."""
        try:
            import bs4  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _check_pptx() -> bool:
        """Check if python-pptx is available."""
        try:
            import pptx  # noqa: F401

            return True
        except ImportError:
            return False
