"""Write chunk_title.json (title of the source document of every chunk) without running the LLM.

Normally kg_construction writes this file itself. Use this module to create it for a knowledge
graph that was built earlier, or to build titles on their own. If the chunks of an existing
run are present, they are re-split and compared, so a mismatch (changed chunking settings or
corpus) is reported instead of silently producing misaligned titles.

Usage:
    python -m gtrag.chunk_titles --config configs/hotpotqa.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

from gtrag.chunking import make_splitter, split_corpus
from gtrag.config import build_parser, load_config


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if not cfg.data.title_field:
        raise ValueError("data.title_field is null; nothing to do")
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.data.corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    texts, titles = split_corpus(corpus, make_splitter(cfg), cfg)

    chunks_file = out_dir / cfg.outputs.chunks
    if chunks_file.exists():
        with open(chunks_file, "r") as f:
            built = json.load(f)
        if texts[:len(built)] != built:
            raise ValueError(f"Re-splitting the corpus does not reproduce {chunks_file}; "
                             "titles would be misaligned (different chunking settings or corpus?)")
        print(f"Chunks match the existing run ({len(built)} chunks).")

    with open(out_dir / cfg.outputs.chunk_titles, "w") as f:
        json.dump(titles, f, indent=4)
    print(len(titles))


if __name__ == "__main__":
    main()
