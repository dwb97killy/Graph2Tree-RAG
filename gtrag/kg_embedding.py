"""Step 2: embed the entities of the knowledge graph.

Reads   <run.output_dir>/all_entities.json          (written by gtrag.kg_construction)
            key  "('NAME', 'TYPE')"  ->  [[description, entity_id, chunk_id], ...]
Writes  <run.output_dir>/entity_faiss_index         row i = vector of the entity with id i
        <run.output_dir>/entity_document.json       [[entity_id, "NAME TYPE description", chunk_id], ...]
                                                    sorted by entity_id, so row i is entry i

Usage:
    python -m gtrag.kg_embedding --config configs/hotpotqa.yaml
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import faiss

from gtrag.config import Config, build_parser, load_config, save_config
from gtrag.embedding import Embedder
from gtrag.index_utils import faiss_index_save
from gtrag.llm import get_device


# ----------------------------------------------------------------------------- key parsing
def parse_key_legacy(key: str) -> tuple:
    """Original parser: strip the brackets and split on ', '. Breaks on names that contain
    ', ' (e.g. 'Washington, D.C.') or an apostrophe (the key is then written with double quotes)."""
    return tuple(x.strip("'") for x in key.strip("()").split(", "))


def parse_key_exact(key: str) -> tuple:
    """Exact inverse of str((name, type))."""
    name, entity_type = ast.literal_eval(key)
    return name, entity_type


def load_entity_texts(raw_data: dict, parser: str) -> list[list]:
    """-> [[entity_id, "NAME TYPE description", chunk_id], ...] sorted by entity id."""
    if parser not in ("legacy", "literal_eval"):
        raise ValueError(f"kg_embedding.key_parser must be 'legacy' or 'literal_eval', got '{parser}'")
    texts, mismatches = [], []
    for key, mentions in raw_data.items():
        legacy = parse_key_legacy(key)
        try:
            exact = parse_key_exact(key)
        except (ValueError, SyntaxError):
            exact = None
        if exact is not None and tuple(legacy[:2]) != exact:
            mismatches.append((key, legacy[:2], exact))
        if parser == "legacy":
            name, entity_type = legacy[0], legacy[1]
        elif exact is None:
            raise ValueError(f"Cannot parse entity key {key!r}")
        else:
            name, entity_type = exact
        for description in mentions:  # [description, entity_id, chunk_id]
            texts.append([description[1], name + " " + entity_type + " " + description[0], description[2]])

    if parser == "legacy" and mismatches:
        print(f"WARNING: {len(mismatches)} of {len(raw_data)} entity keys are parsed differently by the "
              f"legacy parser than by literal_eval (names with ', ' or apostrophes). Examples:")
        for key, legacy, exact in mismatches[:3]:
            print(f"  {key}  legacy={legacy}  exact={exact}")
        print("  Set kg_embedding.key_parser=literal_eval to fix this (changes the embedded texts).")

    texts.sort(key=lambda x: x[0])
    return texts


def check_ids(texts: list[list]) -> None:
    """Retrieval maps a FAISS row number straight to an entity id, so ids must be 0..N-1."""
    if [t[0] for t in texts] != list(range(len(texts))):
        raise ValueError("Entity ids are not contiguous 0..N-1, so FAISS row numbers would not match "
                         "entity ids. Re-run kg_construction from scratch.")


# ----------------------------------------------------------------------------- main
def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg: Config = load_config(args.config, args.overrides)
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir, name="resolved_config_kg_embedding.yaml")
    o = cfg.outputs

    embedder = Embedder(cfg, get_device())

    with open(out_dir / o.entities, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    entity_texts = load_entity_texts(raw_data, cfg.kg_embedding.key_parser)
    check_ids(entity_texts)
    print(len(entity_texts))

    entity_faiss_index = faiss.IndexFlatIP(cfg.embedding.dim)
    batch_size = cfg.kg_embedding.batch_size
    for start in range(0, len(entity_texts), batch_size):
        batch = [t[1] for t in entity_texts[start:start + batch_size]]
        entity_faiss_index.add(embedder.encode(batch))

    faiss_index_save(entity_faiss_index, out_dir / o.entity_faiss_index)
    with open(out_dir / o.entity_document, "w") as f:
        json.dump(entity_texts, f)


if __name__ == "__main__":
    main()
