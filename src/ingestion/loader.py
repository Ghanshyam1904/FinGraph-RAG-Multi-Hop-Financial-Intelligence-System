"""
Phase 1 — Data Ingestion: loader.py

Multi-format document loaders for financial documents:
- PDF (SEC 10-K / 10-Q / 8-K filings) via PyMuPDF
- HTML (earnings call transcripts, news, EDGAR filings) via BeautifulSoup
- SEC EDGAR API for programmatic filing retrieval
- Folder loading — pick up everything already downloaded
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


@dataclass
class Document:
    """A loaded document with content and rich metadata."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Loads financial documents from disk or the SEC EDGAR API."""

    SEC_EDGAR_BASE = "https://data.sec.gov"

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or os.getenv(
            "SEC_USER_AGENT", "FinGraphRAG contact@example.com"
        )

    # ------------------------------------------------------------------ #
    # PDF loading (10-K, 10-Q, 8-K filings)
    # ------------------------------------------------------------------ #
    def load_pdf(self, path: str | Path, *, company: str = "", doc_type: str = "10-K",
                 year: int | None = None) -> list[Document]:
        """Parse a PDF filing into one Document per page, with metadata."""
        if fitz is None:
            raise ImportError("PyMuPDF is required: pip install pymupdf")

        path = Path(path)
        docs: list[Document] = []
        with fitz.open(path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                docs.append(
                    Document(
                        content=text,
                        metadata={
                            "source": str(path),
                            "company": company,
                            "doc_type": doc_type,
                            "year": year,
                            "page_num": page_num,
                            "section": self._guess_section(text),
                        },
                    )
                )
        return docs

    # ------------------------------------------------------------------ #
    # HTML loading (earnings call transcripts, news articles, EDGAR filings)
    # ------------------------------------------------------------------ #
    def load_html(self, source: str, *, company: str = "", doc_type: str = "transcript",
                   year: int | None = None, is_url: bool = False) -> Document:
        """Parse an HTML file or URL into a single cleaned-text Document."""
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4")

        if is_url:
            resp = requests.get(source, headers={"User-Agent": self.user_agent}, timeout=30)
            resp.raise_for_status()
            html = resp.text
        else:
            html = Path(source).read_text(encoding="utf-8", errors="ignore")

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n")).strip()

        return Document(
            content=text,
            metadata={
                "source": source,
                "company": company,
                "doc_type": doc_type,
                "year": year,
                "section": None,
            },
        )

    # ------------------------------------------------------------------ #
    # Folder loading — pick up everything already downloaded
    # ------------------------------------------------------------------ #
    def load_directory(self, dir_path: str | Path, *, company: str = "", doc_type: str = "10-K",
                        year: int | None = None) -> list[Document]:
        """Load every .htm/.html/.pdf file in a folder (e.g. data/raw/sec_filings/)."""
        dir_path = Path(dir_path)
        all_docs: list[Document] = []

        for file_path in sorted(dir_path.iterdir()):
            suffix = file_path.suffix.lower()
            if suffix in (".htm", ".html"):
                doc = self.load_html(
                    str(file_path), company=company, doc_type=doc_type, year=year, is_url=False
                )
                all_docs.append(doc)
            elif suffix == ".pdf":
                docs = self.load_pdf(str(file_path), company=company, doc_type=doc_type, year=year)
                all_docs.extend(docs)

        return all_docs

    # ------------------------------------------------------------------ #
    # SEC EDGAR API
    # ------------------------------------------------------------------ #
    def fetch_edgar_filings(self, cik: str, forms: tuple[str, ...] = ("10-K", "10-Q")) -> list[dict]:
        """Fetch recent filing metadata for a company (CIK) from SEC EDGAR."""
        cik_padded = str(cik).zfill(10)
        url = f"{self.SEC_EDGAR_BASE}/submissions/CIK{cik_padded}.json"
        resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        results = []
        for i, form in enumerate(recent.get("form", [])):
            if form in forms:
                results.append(
                    {
                        "form": form,
                        "accession_number": recent["accessionNumber"][i],
                        "filing_date": recent["filingDate"][i],
                        "primary_document": recent["primaryDocument"][i],
                        "company_name": data.get("name"),
                        "cik": cik_padded,
                    }
                )
        return results

    def download_filing(self, cik: str, accession_number: str, primary_document: str,
                         dest_dir: str | Path) -> Path:
        """Download a single filing document to dest_dir."""
        acc_no_dashes = accession_number.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_dashes}/{primary_document}"
        )
        resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=60)
        resp.raise_for_status()

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / primary_document
        dest_path.write_bytes(resp.content)
        return dest_path

    # ------------------------------------------------------------------ #
    @staticmethod
    def _guess_section(text: str) -> str | None:
        """Cheap heuristic to tag common 10-K section headers."""
        headers = {
            "risk factors": "Item 1A",
            "management's discussion": "Item 7 (MD&A)",
            "financial statements": "Item 8",
            "controls and procedures": "Item 9A",
        }
        lower = text.lower()
        for key, label in headers.items():
            if key in lower[:500]:
                return label
        return None