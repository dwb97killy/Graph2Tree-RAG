"""Answer questions with GTRAG (the step after gtrag.kg_construction / gtrag.kg_embedding).

Usage:
    python -m gtrag.answer --config configs/hotpotqa.yaml \
        --set llm.model_path=/path/to/Llama-3-8B-Instruct
    python -m gtrag.answer --config configs/hotpotqa.yaml \
        --set llm.backend=openai --set llm.api.model=<model name>      # API instead of a local model

Writes to <run.output_dir>/records/ after every question, so a run can be resumed with
--set run.resume=true:
    answer_record_kg.json     answers from the tree-structured generation (the method)
    answer_record_rag.json    answers from the flat concatenation of the same chunks (generation.also_flat)
    kg_document_record.json   the chunk tree of every question
    rag_document_record.json  (kept for compatibility with the original script)
    init_rag_record.json      the top-k chunks recalled by plain embedding search for every question
    consume.json              prompt tokens / LLM calls per stage and failure counters (paper Table 8)
    count.json                number of questions finished
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from gtrag import prompts as prompt_lib
from gtrag.config import Config, build_parser, load_config, save_config
from gtrag.embedding import Embedder
from gtrag.llm import build_chat, get_device
from gtrag.pipeline import GTRAG, Stats, load_kg_data
from gtrag.questions import load_questions

RECORD_FILES = ("answer_record_kg", "answer_record_rag", "rag_document_record", "kg_document_record",
                "init_rag_record", "count", "consume")


def _dump(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)  # an interrupted run never leaves a half-written record


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg: Config = load_config(args.config, args.overrides)
    cfg.llm.max_new_tokens = cfg.inference.max_new_tokens
    out_dir = Path(cfg.run.output_dir)
    rec_dir = out_dir / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir, name="resolved_config_answer.yaml")

    questions = load_questions(cfg)
    kg = load_kg_data(cfg)
    print(f"relation set size: {len(kg.relations)}  entity set size: {len(kg.entity_document)}  "
          f"chunk set size: {len(kg.chunks)}  questions: {len(questions)}")

    device = get_device()
    gtrag = GTRAG(cfg, build_chat(cfg, device), Embedder(cfg, device), prompt_lib.from_config(cfg), kg)

    rec = {name: [] for name in RECORD_FILES[:5]}
    count = 0
    if cfg.run.resume and (rec_dir / "count.json").exists():
        for name in RECORD_FILES[:5]:
            with open(rec_dir / f"{name}.json") as f:
                rec[name] = json.load(f)
        with open(rec_dir / "count.json") as f:
            count = int(json.load(f))
        with open(rec_dir / "consume.json") as f:
            gtrag.stats = Stats.from_saved(json.load(f))
        print(f"resuming after question {count}")

    for single_question in questions[count:]:
        print(count, single_question)
        count += 1
        result = gtrag.answer_question(single_question)
        rec["answer_record_kg"].append(result["answer_kg"])
        rec["answer_record_rag"].append(result["answer_rag"])
        rec["rag_document_record"].append(result["rag_result"])
        rec["kg_document_record"].append(result["kg_result"])
        if result["query_candidates"]:
            rec["init_rag_record"].append(result["query_candidates"])

        if count % cfg.run.save_every == 0 or count == len(questions):
            for name in RECORD_FILES[:5]:
                _dump(rec[name], rec_dir / f"{name}.json")
            _dump(count, rec_dir / "count.json")
            _dump(gtrag.stats.to_dict(), rec_dir / "consume.json")

    print("done:", count, "questions;", gtrag.stats.to_dict())


if __name__ == "__main__":
    main()
