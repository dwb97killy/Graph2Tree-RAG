"""Load the questions to answer, as configured in `questions:` (see configs/default.yaml)."""
from __future__ import annotations

import csv
import json

from gtrag.config import Config


def load_questions(cfg: Config) -> list[str]:
    q = cfg.questions
    if q.source == "json":
        if not q.path:
            raise ValueError("Set questions.path (a JSON file), e.g. --set questions.path=hotpotqa.json")
        with open(q.path, "r", encoding="utf-8") as f:
            records = json.load(f)
        questions = [r[q.field] for r in records] if q.field else list(records)
    elif q.source == "csv":
        if not q.path:
            raise ValueError("Set questions.path (a CSV file)")
        questions = []
        with open(q.path, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if q.csv_skip_header_value is not None and row[0] == q.csv_skip_header_value:
                    continue
                questions.append(row[q.csv_column])
    elif q.source == "hf":
        from datasets import load_dataset  # imported lazily: only this source needs it

        if not q.hf_name:
            raise ValueError("Set questions.hf_name, e.g. yixuantt/MultiHopRAG")
        ds = load_dataset(q.hf_name, q.hf_config) if q.hf_config else load_dataset(q.hf_name)
        questions = list(ds[q.hf_split][q.hf_field])
    else:
        raise ValueError(f"questions.source must be json, csv or hf, got '{q.source}'")
    return questions[:q.limit] if q.limit else questions
