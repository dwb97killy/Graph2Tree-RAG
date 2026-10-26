"""Small FAISS helpers shared by the pipeline stages."""
from __future__ import annotations

import faiss


def faiss_index_save(faiss_index, path) -> None:
    faiss.write_index(faiss_index, str(path))


def faiss_index_load(path):
    return faiss.read_index(str(path))


def search(faiss_index, embedder, query: str, top_k: int):
    """(scores, row indices) of the top_k vectors closest to the query, as returned by FAISS."""
    query_embedding = embedder.encode([query])
    return faiss_index.search(query_embedding, top_k)


def data_recall(faiss_index, embedder, query: str, top_k: int):
    """Row indices of the top_k vectors closest to the query (`embedder` has .encode(list[str]))."""
    query_embedding = embedder.encode([query])
    _, index = faiss_index.search(query_embedding, top_k)
    return index
