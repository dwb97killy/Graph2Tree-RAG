Graph2Tree-RAG (GTRAG)


## Pipeline ##

GTRAG has three stages:

1. **Pre-processing:** split the corpus into chunks, extract entities and relations with an LLM, and fuse duplicate entities into a unified knowledge graph.
2. **Chunk selection:** BFS retrieval, LLM and semantic pruning over a Dual-Signal Relation Chain (DSRC), chunk indexing, cross-relation exploration, and iterative tree update.
3. **Answer generation:** bottom-up recursive generation over the tree.


########### Repository layout ###########

```
<repo>/
├── configs/                 # default.yaml (every hyperparameter) + one small YAML per dataset
├── prompts/                 # every prompt and message template sent to the LLM (plain text)
├── gtrag/
│   ├── kg_construction.py   # step 1: chunking, chunk index, entity/relation extraction
│   ├── kg_embedding.py      # step 2: entity index
│   ├── answer.py            # step 3: retrieval + answer generation (command-line entry point)
│   ├── pipeline.py          # BFS retrieval, pruning, cross-relation exploration, tree generation
│   ├── llm.py               # local Hugging Face model or any OpenAI-compatible API
│   └── ...                  # config, prompts loader, embedding, FAISS helpers, question loader
├── scripts/                 # build_kg.sh, answer.sh
└── requirements.txt
```


########### Setup ###########

```bash
pip install -r requirements.txt
```

- Python: `<VERSION>`
- Backbone LLM: `<EXACT CHECKPOINT, e.g. base vs. Instruct>`, temperature 0
- Embedding model: `nomic-ai/nomic-embed-text-v1`, 768-dim (set in `embedding.model`)
- Chunking: LangChain `RecursiveCharacterTextSplitter`, chunk size `<N>`, overlap `<M>`
- Open-ended judge: `<JUDGE MODEL>`, temperature `<T>`
- Hardware / approximate runtime: `<GPU TYPE, HOURS>`

Do **not** commit API keys. Set them as environment variables:

```bash
export OPENAI_API_KEY=...   # only if llm.backend=openai (or the judge / a baseline uses an API model)
```


########### Quick start ###########

```bash
# 1-2. build the knowledge graph and the vector indexes for one dataset
bash scripts/build_kg.sh configs/hotpotqa.yaml --set llm.model_path=<LOCAL MODEL OR HF ID>

# 3. answer the questions (local model ...)
bash scripts/answer.sh configs/hotpotqa.yaml --set llm.model_path=<LOCAL MODEL OR HF ID>
#    ... or through an API
bash scripts/answer.sh configs/hotpotqa.yaml --set llm.backend=openai --set llm.api.model=<MODEL NAME>
```

Results are written to `outputs/<dataset>/records/` (`answer_record_kg.json` holds the answers; `consume.json`
the tokens and LLM calls per stage). A run can be continued with `--set run.resume=true`.

**Changing a setting or a prompt does not require touching the code.** Every hyperparameter is in
`configs/default.yaml` (override it in a dataset YAML or with `--set section.key=value`; a misspelled key is an
error). Every prompt is a text file in `prompts/`; `{{name}}` marks a value filled in by the code and
`{SEP}`, `{BEG}`, ... mark the output delimiters defined under `prompts.delimiters`. Prompt files are read
byte for byte, so a trailing newline added by an editor becomes part of the prompt.


########### Data ###########

We use the same 1,000-question samples and retrieval corpora as HippoRAG for MuSiQue, HotpotQA, and 2WikiMultihopQA. Open-ended QA uses TraCR, QA-Expert-multi-hop-qa-V1.0, and MultiHop-RAG (500-query random samples for the latter two, seed `<SEED>`).

Put the files named in the dataset config (`configs/<dataset>.yaml`) next to where you run the commands:

| Dataset         | Corpus (`data.corpus_path`)                                | Questions (`questions.path`)                                          |
|-----------------|------------------------------------------------------------|-----------------------------------------------------------------------|
| HotpotQA        | `hotpotqa_corpus.json`: list of `{"idx", "title", "text"}` | `hotpotqa.json`: HippoRAG file, list of records with a `question` key |
| 2WikiMultiHopQA | `2wikimultihopqa_corpus.json`: list of `{"title", "text"}` | `2wikimultihopqa.json`: HippoRAG file, `question` keym                |
| MuSiQue         | `musique_corpus.json`: list of `{"title", "text"}`         | `musique.json`: HippoRAG file, `question` key                         |

Only `text` and `title` of each corpus entry are read (`idx` is ignored), and the text is used exactly as it appears in
the file. Every other field of a question record is ignored. The three datasets use the same pipeline and settings,
so switching dataset is just a different config: `bash scripts/build_kg.sh configs/musique.yaml ...`.

Dataset licenses and download sources: `<LINKS OR CITATIONS>`.


########### Hyperparameters ###########

All values live in `configs/default.yaml`; the default ones are:

| Setting (config key) | Value |
|----------------------|-------|
| Relation pruning (`relation_pruning`) | cosine similarity to the query and relation chain, keep top 5 of every batch of 10 relations with score >= 0.5 (`use_llm: true` switches to LLM scores 0-5, keep >= 3) |
| Search depth (`retrieval.max_hop`) | 2 |
| Chunks added per node (`retrieval.kg_neighbors_per_node`, `cross_neighbors_per_node`) | 1 through KG relations, 1 through cross-relation exploration |
| Candidates for cross-relation exploration (`retrieval.semantic_candidates`) | top-10 chunks by embedding similarity |
| Entities matched per query entity (`query_extraction`) | closest 5, plus any of the closest 50 with inner product >= 0.90 |
| Generation (`generation`) | tree-structured; the flat baseline over the same chunks is also produced (`also_flat`) |
| Decoding (`llm`, `inference`) | greedy, at most 1024 new tokens |


########### Prompts ###########

All prompts are in `prompts/`:

| File | Used for |
|------|----------|
| `kg_extraction_system.txt`, `kg_extraction_user.txt`, `kg_entity_types.txt`, `kg_continue.txt`, `kg_loop.txt`, `kg_format_error.txt` | entity/relation extraction with 1-10 confidence score (step 1) |
| `query_extraction_system.txt`, `user_query_extraction.txt` | entities of the question |
| `relation_pruning_system.txt`, `user_relation_pruning.txt` | LLM relation pruning (only with `relation_pruning.use_llm: true`) |
| `cross_first_system.txt`, `cross_next_system.txt`, `user_cross_*.txt` | cross-chunk relation extraction from the question / from a relation chain |
| `chunk_merge_system.txt`, `user_merge_header.txt` | intermediate answer at internal nodes |
| `answer_short_system.txt`, `answer_open_system.txt`, `user_answer_header.txt` | final answer at the root (`generation.answer_mode`: short or open) |
| `user_node_text.txt`, `user_child_text.txt`, `user_flat_*.txt` | how chunk texts are put into those messages |
| `format_error_*.txt` | re-ask messages when a reply does not follow the required format |
| `frag_*.txt` | one relation / entity as it appears inside a message |
| `llm_judge.txt` | open-ended evaluation (4 dimensions); added together with the evaluation script |


########### Notes ###########

- Results can vary slightly with LLM version and serving stack; temperature is 0 for all generation.
- Knowledge-graph construction is run once per corpus; `<COST NOTE: tokens / time>`.

