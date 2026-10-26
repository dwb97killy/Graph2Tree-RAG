"""Document splitting, shared by every stage so chunk ids always line up.

chunk i of the whole corpus has: text  = all_texts[i]            (faiss_document.json, FAISS row i)
                                 title = chunk_titles[i]         (chunk_title.json)
"""
from __future__ import annotations

import re

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:  # newer LangChain releases moved the splitters to their own package
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from gtrag.config import Config


def make_splitter(cfg: Config):
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunking.chunk_size, chunk_overlap=cfg.chunking.chunk_overlap)


def split_document(splitter, text: str) -> list[str]:
    """Split one document into chunks; newlines inside a chunk become single spaces."""
    return [re.sub(r"\n+", " ", s) for s in splitter.split_text(text)]


def doc_title(doc: dict, i: int, cfg: Config) -> str:
    field = cfg.data.title_field
    try:
        return doc[field]
    except KeyError:
        raise KeyError(f"Document {i} has no '{field}' field; "
                       f"set data.title_field=null if the corpus has no titles") from None


def split_corpus(corpus: list[dict], splitter, cfg: Config) -> tuple[list[str], list[str]]:
    """Split a whole corpus without touching the LLM: -> (chunk texts, chunk titles)."""
    texts, titles = [], []
    for i, doc in enumerate(corpus):
        chunks = split_document(splitter, doc[cfg.data.text_field])
        texts += chunks
        titles += [doc_title(doc, i, cfg)] * len(chunks)
    return texts, titles
