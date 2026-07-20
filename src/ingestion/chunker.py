"""
Phase 1 — Data Ingestion: chunker.py

Recursive character text splitting with metadata propagation, plus
lightweight semantic-boundary detection using sentence embeddings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .loader import Document

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict[str, Any]


class SemanticChunker:
    """Splits Documents into overlapping chunks tagged with source metadata."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64,
                 use_semantic_boundaries: bool = False,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_boundaries = use_semantic_boundaries
        self._embedder = None
        self._embedding_model_name = embedding_model

        if RecursiveCharacterTextSplitter is not None:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:  # fallback if langchain isn't installed
            self._splitter = None

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a single Document into metadata-tagged Chunks."""
        pieces = self._split_text(doc.content)
        chunks = []
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = (
                f"{doc.metadata.get('company', 'unk')}_{doc.metadata.get('doc_type', 'doc')}"
                f"_{doc.metadata.get('page_num', 0)}_{i}"
            )
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=piece,
                    metadata={**doc.metadata, "chunk_index": i},
                )
            )
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Chunk a list of Documents (e.g. every page/file from load_directory)."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # ------------------------------------------------------------------ #
    def _split_text(self, text: str) -> list[str]:
        if self._splitter is not None:
            return self._splitter.split_text(text)
        return self._naive_split(text)

    def _naive_split(self, text: str) -> list[str]:
        """Fallback splitter if LangChain isn't installed: sentence windows w/ overlap."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) <= self.chunk_size:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current)
                current = (current[-self.chunk_overlap:] + " " + sentence) if self.chunk_overlap else sentence
        if current:
            chunks.append(current)
        return chunks

    def detect_semantic_boundaries(self, sentences: list[str], threshold: float = 0.3) -> list[int]:
        """
        Optional: use embedding cosine-distance jumps between consecutive
        sentences to find natural topic breaks. Returns indices where a
        new semantic section likely begins.
        """
        if not self.use_semantic_boundaries:
            return []
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            return []

        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embedding_model_name)

        embeddings = self._embedder.encode(sentences, normalize_embeddings=True)
        boundaries = []
        for i in range(1, len(embeddings)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            if (1 - sim) > threshold:
                boundaries.append(i)
        return boundaries