"""Text ingestion, cleaning, and chunking for raw corpus files."""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_index: int


def load_corpus(corpus_dir: str | Path) -> list[str]:
    """Load all .txt files from a corpus directory, return list of raw texts."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    texts = []
    for txt_file in sorted(corpus_dir.glob("*.txt")):
        texts.append(txt_file.read_text(encoding="utf-8"))
    if not texts:
        raise ValueError(f"No .txt files found in {corpus_dir}")
    return texts


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, remove artifacts."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks by character count, breaking at paragraph boundaries."""
    text = clean_text(text)
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        if current_len + para_len > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Keep last paragraph(s) for overlap
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current_chunk):
                if overlap_len + len(p) > overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_len += len(p)
            current_chunk = overlap_parts
            current_len = overlap_len

        current_chunk.append(para)
        current_len += para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def load_and_chunk(corpus_dir: str | Path, chunk_size: int = 1500, overlap: int = 200) -> list[Chunk]:
    """Load corpus files, clean, and chunk them. Returns list of Chunk objects."""
    corpus_dir = Path(corpus_dir)
    texts = load_corpus(corpus_dir)

    all_chunks = []
    for txt_file, raw_text in zip(sorted(corpus_dir.glob("*.txt")), texts):
        text_chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk_text_ in enumerate(text_chunks):
            all_chunks.append(Chunk(text=chunk_text_, source_file=txt_file.name, chunk_index=i))

    return all_chunks
