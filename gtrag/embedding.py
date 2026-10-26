"""One embedding model for every stage, so chunk vectors and entity vectors always match."""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from gtrag.config import Config


class Embedder:
    def __init__(self, cfg: Config, device):
        self.model = SentenceTransformer(
            cfg.embedding.model, trust_remote_code=cfg.embedding.trust_remote_code).to(device)
        self.normalize = cfg.embedding.normalize

    def encode(self, texts: list[str]):
        # With normalize=false the call is exactly model.encode(texts), as in the original scripts.
        kwargs = {"normalize_embeddings": True} if self.normalize else {}
        return self.model.encode(texts, **kwargs)
