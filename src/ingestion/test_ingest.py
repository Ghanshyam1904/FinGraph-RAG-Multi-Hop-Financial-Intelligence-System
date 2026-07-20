from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import SemanticChunker

loader = DocumentLoader()
chunker = SemanticChunker(chunk_size=512, chunk_overlap=64)

docs = loader.load_directory("data/raw/sec_filings", company="Apple", doc_type="10-K", year=2025)
print(f"Loaded {len(docs)} document(s)")

chunks = chunker.chunk_documents(docs)
print(f"Created {len(chunks)} chunks")
print(chunks[0].content[:200])
print(chunks[0].metadata)