"""Build the Chroma vector store from the book PDF.

Usage:
    python ingest.py --pdf data/book.pdf
    python ingest.py --pdf data/book.pdf --reset    # rebuild from scratch
"""
import argparse
import logging
import os
import re
import unicodedata
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import CHROMA_DIR, COLLECTION_NAME, get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")

WATERMARK = re.compile(r"(Download from\s+)?www\.finelybook\.com", re.IGNORECASE)


def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\x00", "")
    t = WATERMARK.sub("", t)              # remove the watermark text, keep the page
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=os.getenv("PDF_PATH", "data/book.pdf"))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    store = get_vector_store()
    existing = store._collection.count()
    if args.reset and existing:
        log.info("Resetting collection '%s' (%d chunks)", COLLECTION_NAME, existing)
        store.delete_collection()
        store = get_vector_store()
        existing = 0
    if existing:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' already has {existing} chunks in {CHROMA_DIR}. "
            "Use --reset to rebuild it."
        )

    pages = PyPDFLoader(str(pdf)).load()
    log.info("Loaded %d pages", len(pages))

    for p in pages:
        p.page_content = clean_text(p.page_content)
    pages = [p for p in pages if p.page_content]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    assert all(isinstance(c.page_content, str) for c in chunks)
    log.info("Created %d chunks", len(chunks))

    total = len(chunks)
    for start in range(0, total, args.batch_size):
        batch = chunks[start : start + args.batch_size]
        store.add_documents(batch)
        log.info("Stored %d / %d", min(start + len(batch), total), total)

    log.info("Done. Collection count: %d", store._collection.count())


if __name__ == "__main__":
    main()