"""Split a text corpus into chunks, index them, and build a knowledge graph with an LLM.

Usage:
    python -m gtrag.kg_construction --config configs/hotpotqa.yaml \
        --set llm.model_path=<HF id or local path>          # local model
    python -m gtrag.kg_construction --config configs/hotpotqa.yaml \
        --set llm.backend=openai --set llm.api.model=<model name>   # API (key in $OPENAI_API_KEY)

All hyperparameters live in configs/*.yaml and all prompts in prompts/*.txt. The extraction
logic is the same as in the original notebook-style script (see the NOTE in process_chunk).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import faiss

from gtrag.chunking import make_splitter, split_corpus, split_document, doc_title
from gtrag.config import Config, build_parser, load_config, save_config
from gtrag.embedding import Embedder
from gtrag.index_utils import faiss_index_load, faiss_index_save
from gtrag.llm import build_chat, get_device
from gtrag.prompts import PromptLibrary, from_config


# ----------------------------------------------------------------------------- KG state
@dataclass
class KGState:
    all_texts: list = field(default_factory=list)
    chunk_titles: list | None = field(default_factory=list)  # source-document title of every chunk (parallel to all_texts)
    all_entities: dict = field(default_factory=dict)   # "(name, type)" -> [[description, entity_id, chunk_id], ...]
    all_relations: dict = field(default_factory=dict)  # "(src_id, tgt_id)" -> [[description, strength, chunk_id], ...]
    entity_size: int = 0   # running entity-id counter
    chunk_count: int = 0   # running chunk counter; also the chunk id stored with every entity / relation
    doc_count: int = 0     # number of documents processed so far
    sample_question: list = field(default_factory=list)
    sample_answer: list = field(default_factory=list)


# ----------------------------------------------------------------------------- parsing
def extract_substrings(s: str, begin_delimiter: str, completion_delimiter: str) -> list[str]:
    pattern = re.compile(f"{re.escape(begin_delimiter)}(.*?){re.escape(completion_delimiter)}", re.DOTALL)
    return pattern.findall(s)


def parse_records(response: str, delims: Config) -> list[list[str]]:
    """LLM reply -> list of records, each a list of fields split on the tuple delimiter."""
    records = []
    for substring in (s.strip() for s in extract_substrings(response, delims.begin_delimiter,
                                                            delims.completion_delimiter)):
        for item in (x.strip() for x in substring.split(delims.record_delimiter)):
            records.append(item.strip().split(delims.tuple_delimiter))
    return records


def deduplicate(records: list[list[str]]) -> list[list[str]]:
    """The LLM sometimes repeats records; keep the first occurrence of each."""
    return [x for i, x in enumerate(records) if x not in records[:i]]


def update_kg(state: KGState, records: list[list[str]]) -> bool:
    """Add the records of ONE chunk to the graph.

    Returns True if the chunk yielded at least one entity record and one relationship record
    (a False return is treated as a format error by the caller).
    """
    entity_id_list = []  # [entity name, entity id] for the entities of this chunk
    entity_num = 0
    relation_num = 0

    for element in records:
        if "entity" in element[0] and len(element) == 4:
            entity_num += 1
            key = str((element[1], element[2]))  # must be a string to be JSON-serialisable
            if key not in state.all_entities:
                state.all_entities[key] = [[element[3], state.entity_size, state.chunk_count]]
            else:
                state.all_entities[key].append([element[3], state.entity_size, state.chunk_count])
            entity_id_list.append([element[1], state.entity_size])
            state.entity_size += 1

    for element in records:
        if "relationship" in element[0] and len(element) == 5:
            flag = 0  # counts how many of (source, target) were resolved to entity ids
            for i in range(len(entity_id_list)):
                if element[1] == entity_id_list[i][0]:
                    element[1] = entity_id_list[i][1]
                    flag += 1
                if element[2] == entity_id_list[i][0]:
                    element[2] = entity_id_list[i][1]
                    flag += 1
            if flag == 2:
                key = str((element[1], element[2]))
                relation_num += 1
                if key not in state.all_relations:
                    state.all_relations[key] = [[element[3], element[4], state.chunk_count]]
                else:
                    state.all_relations[key].append([element[3], element[4], state.chunk_count])

    return relation_num != 0 and entity_num != 0


# ----------------------------------------------------------------------------- per-chunk extraction
def process_chunk(text: str, chat, prompts: PromptLibrary, state: KGState, cfg: Config) -> None:
    delims = cfg.prompts.delimiters
    kcfg = cfg.kg_extraction
    verbose = cfg.run.verbose

    def log_records(records):
        if verbose:
            for record in records:
                print(record)

    conv = [
        {"role": "system", "content": prompts.render("kg_extraction_system")},
        {"role": "user", "content": prompts.render(
            "kg_extraction_user", entity_types=prompts.raw("kg_entity_types"), text=text)},
    ]
    response = chat(conv)
    conv.append({"role": "assistant", "content": response})
    records = deduplicate(parse_records(response, delims))
    log_records(records)
    ok = update_kg(state, records)

    if not ok and kcfg.retry_on_format_error:
        # Nothing usable was extracted: ask once more with the format reminder.
        conv.append({"role": "user", "content": prompts.render("kg_format_error")})
        response = chat(conv)
        conv.append({"role": "assistant", "content": response})
        records = deduplicate(parse_records(response, delims))
        log_records(records)
        update_kg(state, records)

    # Optional "gleaning" rounds (kg_extraction.loop_num, 0 = off).
    for _ in range(kcfg.loop_num):
        conv.append({"role": "user", "content": prompts.render("kg_loop")})
        response = chat(conv)
        conv.append({"role": "assistant", "content": response})
        if response == "Y":
            conv.append({"role": "user", "content": prompts.render("kg_continue")})
            response = chat(conv)
            conv.append({"role": "assistant", "content": response})
            # NOTE (kept from the original script): gleaned records are parsed and printed
            # but are NOT added to the graph. This is a no-op while loop_num is 0.
            log_records(parse_records(response, delims))
        else:
            break

    state.chunk_count += 1


# ----------------------------------------------------------------------------- persistence
def _dump(obj, path: Path, **kwargs) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, **kwargs)


def _load(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_state(cfg: Config, state: KGState, faiss_index, out_dir: Path) -> None:
    o = cfg.outputs
    faiss_index_save(faiss_index, out_dir / o.faiss_index)
    _dump(state.all_texts, out_dir / o.chunks)
    if cfg.data.title_field:
        _dump(state.chunk_titles, out_dir / o.chunk_titles, indent=4)
    _dump(state.all_entities, out_dir / o.entities, indent=4)
    _dump(state.all_relations, out_dir / o.relations, indent=4)
    _dump(state.doc_count, out_dir / o.processed_docs)
    _dump(state.entity_size, out_dir / o.processed_entities)
    _dump(state.sample_question, out_dir / o.sample_question)
    _dump(state.sample_answer, out_dir / o.sample_answer)


def load_or_init_state(cfg: Config, out_dir: Path):
    o = cfg.outputs
    if cfg.run.resume:
        faiss_index = faiss_index_load(out_dir / o.faiss_index)
        state = KGState(
            all_texts=_load(out_dir / o.chunks),
            all_entities=_load(out_dir / o.entities),
            all_relations=_load(out_dir / o.relations),
            doc_count=int(_load(out_dir / o.processed_docs)),
            entity_size=int(_load(out_dir / o.processed_entities)),
            sample_question=_load(out_dir / o.sample_question),
            sample_answer=_load(out_dir / o.sample_answer),
        )
        state.chunk_count = len(state.all_texts)
        titles_file = out_dir / o.chunk_titles
        # None = run made before titles existed; main() backfills them from the corpus.
        state.chunk_titles = _load(titles_file) if (cfg.data.title_field and titles_file.exists()) else None
    else:
        # Brute-force inner-product index; equals cosine similarity for L2-normalised embeddings.
        faiss_index = faiss.IndexFlatIP(cfg.embedding.dim)
        state = KGState()
    return state, faiss_index


def print_status(state: KGState, faiss_index) -> None:
    print("total doc: ", len(state.sample_question))
    print("total chunk: ", state.chunk_count)
    print("entity_size: ", state.entity_size)
    print("relation_size: ", len(state.all_relations))
    print(f"vectors in index: {faiss_index.ntotal}")


# ----------------------------------------------------------------------------- main
def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if not cfg.data.corpus_path:
        raise ValueError("Set data.corpus_path (use a dataset config such as configs/hotpotqa.yaml)")

    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir)  # keep the resolved config next to the results

    prompts = from_config(cfg)
    device = get_device()
    chat = build_chat(cfg, device)  # local Hugging Face model or API, see gtrag/llm.py
    embedder = Embedder(cfg, device)
    text_splitter = make_splitter(cfg)

    state, faiss_index = load_or_init_state(cfg, out_dir)
    print(state.doc_count)
    print_status(state, faiss_index)

    with open(cfg.data.corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    if cfg.data.title_field and state.chunk_titles is None:
        # Resuming a run that has no chunk_title.json: rebuild it, but only if re-splitting
        # reproduces the chunks already built (otherwise titles would be misaligned).
        texts, titles = split_corpus(corpus[:state.doc_count], text_splitter, cfg)
        if texts != state.all_texts:
            raise ValueError("Cannot backfill chunk titles: re-splitting the corpus does not reproduce the "
                             "chunks of this run (changed chunking settings or corpus?). "
                             "Set data.title_field=null or rebuild from scratch.")
        state.chunk_titles = titles

    for i in range(state.doc_count, len(corpus)):
        # Split the document into chunks and index them.
        texts = split_document(text_splitter, corpus[i][cfg.data.text_field])
        faiss_index.add(embedder.encode(texts))
        state.all_texts = state.all_texts + texts
        if cfg.data.title_field:
            state.chunk_titles += [doc_title(corpus[i], i, cfg)] * len(texts)

        # Extract entities and relations chunk by chunk.
        for text in texts:
            process_chunk(text, chat, prompts, state, cfg)

        state.doc_count += 1
        if state.doc_count % cfg.run.save_every == 0:
            save_state(cfg, state, faiss_index, out_dir)
            print_status(state, faiss_index)


if __name__ == "__main__":
    main()
