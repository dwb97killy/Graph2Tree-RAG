"""GTRAG question answering: query-entity extraction, BFS over chunks, pruning, cross-relation
exploration, and bottom-up answer generation.

This is the logic of the original GTRAG.py with every prompt moved to prompts/*.txt and every
constant moved to configs/default.yaml. Behaviour is unchanged unless a config value says otherwise
(the few places where the original could loop forever now stop after a configurable number of
retries; see the comments marked "CAP").

Typical use (see gtrag/answer.py for the command-line entry point):

    gtrag = GTRAG(cfg, chat, embedder, prompts, kg_data)
    result = gtrag.answer_question("Which river flows through ...?")
    result["answer_kg"]       # tree-structured answer
"""
from __future__ import annotations

import copy
import heapq
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gtrag.config import Config
from gtrag.index_utils import faiss_index_load, search
from gtrag.prompts import PromptLibrary

STAGES = ("query_extraction", "relation_pruning", "cross_exploration",
          "answer_generation_kg", "answer_generation_rag")


# ----------------------------------------------------------------------------- small helpers
def can_convert_to_float(s) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def can_be_converted_to_int(s) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def chunk_list(lst, chunk_size=10):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def extract_substrings(s: str, begin_delimiter: str, completion_delimiter: str) -> list[str]:
    pattern = re.compile(f"{re.escape(begin_delimiter)}(.*?){re.escape(completion_delimiter)}", re.DOTALL)
    return pattern.findall(s)


def extract_number(s):
    match = re.search(r"\d+(?:\.\d+)?", s)
    return float(match.group()) if match else None


def merge_entities(list1, list2):
    """list1 followed by the elements of list2 that are not already in list1."""
    list1_copy = copy.deepcopy(list1)
    for element in list2:
        if element not in list1:
            list1_copy.append(element)
    return list1_copy


# ----------------------------------------------------------------------------- knowledge graph data
@dataclass
class KGData:
    entity_document: list   # [[entity_id, "NAME TYPE description", chunk_id], ...]; row i == entity id i
    entity_index: Any       # FAISS index over entity_document
    chunks: list            # chunk texts; row i == chunk id i
    chunk_index: Any        # FAISS index over chunks
    relations: list         # [[source_id, target_id, description, strength, chunk_id], ...]


def load_kg_data(cfg: Config) -> KGData:
    out, o = Path(cfg.run.output_dir), cfg.outputs
    with open(out / o.entity_document, "r") as f:
        entity_document = json.load(f)
    with open(out / o.chunks, "r") as f:
        chunks = json.load(f)
    with open(out / o.relations, "r") as f:
        all_relations = json.load(f)
    relations = []
    for relation in all_relations:  # key "(source_id, target_id)"
        entity_obj = tuple(x.strip("'") for x in relation.strip("()").split(", "))
        for description in all_relations[relation]:  # [description, strength, chunk_id]
            relations.append([int(entity_obj[0]), int(entity_obj[1]), description[0],
                              float(extract_number(description[1])), float(description[2])])
    return KGData(entity_document, faiss_index_load(out / o.entity_faiss_index), chunks,
                  faiss_index_load(out / o.faiss_index), relations)


# ----------------------------------------------------------------------------- cost / failure counters
class Stats:
    """Prompt tokens and LLM calls per stage (paper Table 8) plus failure counters."""

    COUNTERS = ("fail_pruning_time", "fail_pruning_query", "fail_expansion_time",
                "fail_expansion_query", "expansion_refuse_query")

    def __init__(self):
        self.tokens = {s: 0 for s in STAGES}
        self.calls = {s: 0 for s in STAGES}
        for c in self.COUNTERS:
            setattr(self, c, 0)

    def to_dict(self) -> dict:
        d = {}
        for s in STAGES:
            d[f"tokens_{s}"], d[f"calls_{s}"] = self.tokens[s], self.calls[s]
        d.update({c: getattr(self, c) for c in self.COUNTERS})
        return d

    def to_legacy_list(self, count: int) -> list:
        """The list layout written by the original GTRAG.py (consume.json)."""
        out = []
        for s in STAGES:
            out += [self.tokens[s], self.calls[s]]
        return out + [getattr(self, c) for c in self.COUNTERS] + [count]

    @classmethod
    def from_saved(cls, saved) -> "Stats":
        st = cls()
        if isinstance(saved, dict):
            for s in STAGES:
                st.tokens[s], st.calls[s] = saved[f"tokens_{s}"], saved[f"calls_{s}"]
            for c in cls.COUNTERS:
                setattr(st, c, saved[c])
        else:  # list written by the original script
            for k, s in enumerate(STAGES):
                st.tokens[s], st.calls[s] = saved[2 * k], saved[2 * k + 1]
            for k, c in enumerate(cls.COUNTERS):
                setattr(st, c, saved[10 + k])
        return st


# ----------------------------------------------------------------------------- the pipeline
class GTRAG:
    def __init__(self, cfg: Config, chat: Callable, embedder, prompts: PromptLibrary, kg: KGData,
                 stats: Stats | None = None, log: Callable = print):
        self.cfg, self.llm, self.embedder, self.prompts, self.kg = cfg, chat, embedder, prompts, kg
        self.stats = stats or Stats()
        self.log = log if cfg.run.verbose else (lambda *a, **k: None)
        self.d = cfg.prompts.delimiters
        self._rows = None  # entity id -> rows of kg.relations that mention it (built on first use)

    # ------------------------------------------------------------------ plumbing
    def render(self, name: str, **kw) -> str:
        return self.prompts.render(name, **kw)

    def _chat(self, messages: list[dict], stage: str | None = None) -> str:
        """Call the LLM; only calls made with a `stage` are counted (as in the original script)."""
        response = self.llm(messages)
        if stage is not None:
            self.stats.tokens[stage] += int(getattr(self.llm, "last_prompt_tokens", 0))
            self.stats.calls[stage] += 1
        return response

    def _search(self, index, text: str, k: int):
        return search(index, self.embedder, text, k)

    def _select_entities(self, scores, rows) -> list[list]:
        """Closest `always_keep` entities, plus further ones whose score reaches the threshold."""
        q = self.cfg.query_extraction
        doc, out = self.kg.entity_document, []
        for ii in range(q.always_keep):
            index = rows[0][ii]
            out.append([int(index), doc[int(index)][2]])
        for ii in range(q.always_keep, len(rows[0])):
            index = rows[0][ii]
            if scores[0][ii] >= q.similarity_threshold:
                out.append([int(index), doc[int(index)][2]])
        return out

    def _relation_rows(self, entity_id: int) -> list[int]:
        if self._rows is None:
            rows = defaultdict(set)
            for i, rel in enumerate(self.kg.relations):
                rows[rel[0]].add(i)
                rows[rel[1]].add(i)
            self._rows = {k: sorted(v) for k, v in rows.items()}
        return self._rows.get(entity_id, [])

    # ------------------------------------------------------------------ 1. entities of the question
    def extract_query_entity(self, question: str):
        q = self.cfg.query_extraction
        begin, tuple_d = self.d["begin_delimiter"], self.d["tuple_delimiter"]
        record, completion = self.d["record_delimiter"], self.d["completion_delimiter"]

        extract_entities, entities_from_query, all_final = [], [], []
        conv = [{"role": "system", "content": self.render("query_extraction_system")},
                {"role": "user", "content": self.render("user_query_extraction", question=question)}]
        response = self._chat(conv, "query_extraction")
        conv.append({"role": "assistant", "content": response})

        def parse(resp, skip_seen):
            for substring in [s.strip() for s in extract_substrings(resp, begin, completion)]:
                for item in [i.strip() for i in substring.split(record)]:
                    parts = item.strip().split(tuple_d)
                    if not skip_seen or parts not in all_final:
                        all_final.append(parts)

        exist_entity, retries = False, 0
        while not exist_entity:
            parse(response, skip_seen=False)

            for _ in range(q.loop_num):  # optional extra "anything missed?" rounds
                conv.append({"role": "user", "content": self.render("query_loop")})
                response = self._chat(conv, "query_extraction")
                conv.append({"role": "assistant", "content": response})
                if response == "Y":
                    conv.append({"role": "user", "content": self.render("query_continue")})
                    response = self._chat(conv, "query_extraction")
                    conv.append({"role": "assistant", "content": response})
                    parse(response, skip_seen=True)

            for element in all_final:
                if len(element) == 4 and "entity" in element[0]:
                    exist_entity = True
                    text = element[1] + " " + element[2] + " " + element[3]
                    if text not in entities_from_query:
                        entities_from_query.append(element[3])
                        scores, rows = self._search(self.kg.entity_index, text, q.pool_size)
                        extract_entities += self._select_entities(scores, rows)

            if not exist_entity:
                if retries >= q.max_format_retries:  # CAP: the original retried forever
                    self.log("WARNING: no entity could be parsed from the question; continuing without entities")
                    break
                retries += 1
                conv.append({"role": "user", "content": self.render("format_error_query")})
                response = self._chat(conv, "query_extraction")
                conv.append({"role": "assistant", "content": response})

        extract_entities = [list(x) for x in set(tuple(x) for x in extract_entities)]  # de-duplicate
        return extract_entities, entities_from_query

    def extract_similar_entities(self, entity_id) -> list[list]:
        """Entities of the KG that are close to the given entity (an id, or a free-text entity)."""
        text = self.kg.entity_document[int(entity_id)][1] if can_be_converted_to_int(entity_id) else entity_id
        scores, rows = self._search(self.kg.entity_index, text, self.cfg.query_extraction.pool_size)
        return self._select_entities(scores, rows)

    # ------------------------------------------------------------------ 2. relations -> neighbour chunks
    def extract_neighbor_relation(self, entity_id_list, query, existing_relation, visited_document,
                                  current_chunk_id, document_queue, entities_from_query_string, user_input_1):
        rp = self.cfg.relation_pruning
        sep = self.d["SEP"]
        neighbors, unique_extract_entities = {}, []

        relation_set = ""
        for rel in existing_relation:
            relation_set += self.render("frag_chain_relation", description=rel[2])

        new_relation, source_id_list, target_id_list = [], [], []
        description_list, strength_list, chunk_id_list = [], [], []
        table = self.kg.relations

        def in_queue(chunk_id) -> bool:
            return any(chunk_id == int(single_document[0]) for single_document in document_queue)

        entity_id = None
        for x in entity_id_list:
            entity_id = int(x)
            for i in self._relation_rows(entity_id):
                if table[i][0] == entity_id:  # relation leaves this entity
                    target_id, description, strength = table[i][1], table[i][2], table[i][3]
                    chunk_id = int(table[i][4])
                    if int(chunk_id) not in visited_document and int(chunk_id) != current_chunk_id \
                            and not in_queue(chunk_id):
                        new_relation.append(self.render("frag_new_relation", description=description))
                        source_id_list.append(entity_id)
                        target_id_list.append(target_id)
                        description_list.append(description)
                        strength_list.append(strength)
                        chunk_id_list.append(chunk_id)
                if table[i][1] == entity_id:  # relation arrives at this entity
                    source_id, description, strength = table[i][0], table[i][2], table[i][3]
                    chunk_id = table[i][4]
                    if int(chunk_id) not in visited_document and int(chunk_id) != current_chunk_id \
                            and not in_queue(chunk_id):
                        new_relation.append(self.render("frag_new_relation", description=description))
                        source_id_list.append(source_id)
                        target_id_list.append(entity_id)
                        description_list.append(description)
                        strength_list.append(strength)
                        chunk_id_list.append(chunk_id)

        if not new_relation:
            return neighbors, [], unique_extract_entities

        # Score the new relations in small batches so the LLM can keep track of them.
        n = rp.batch_size
        new_relation_chunks = chunk_list(new_relation, n)
        source_chunks, target_chunks = chunk_list(source_id_list, n), chunk_list(target_id_list, n)
        description_chunks, strength_chunks = chunk_list(description_list, n), chunk_list(strength_list, n)
        chunk_id_chunks = chunk_list(chunk_id_list, n)
        fail_this_call = 0

        for j in range(len(new_relation_chunks)):
            format_flag, format_check_time = 0, 0
            new_relation_prompt = "".join(r + ";\n" for r in new_relation_chunks[j])
            user_input = self.render(
                "user_relation_pruning", n=len(new_relation_chunks[j]), query=query,
                query_entities=entities_from_query_string, relation_chain=relation_set,
                new_relations=new_relation_prompt)

            if rp.use_llm:
                conv = [{"role": "system", "content": self.render("relation_pruning_system")},
                        {"role": "user", "content": user_input}]
                response = self._chat(conv, "relation_pruning")
                conv.append({"role": "assistant", "content": response})
                substrings = [s for s in response.split(sep) if s]
            else:
                query_vec = self.embedder.encode([user_input_1])[0]
                text_vecs = self.embedder.encode(new_relation_chunks[j])
                dot_products = np.dot(text_vecs, query_vec)
                cosine_scores = dot_products / (np.linalg.norm(query_vec) * np.linalg.norm(text_vecs, axis=1))
                substrings_float = cosine_scores.tolist()
                format_flag = 1

            while format_flag == 0 and format_check_time <= rp.max_format_retries:
                substrings_float, skip_num_check = [], False
                for score in substrings:  # format check
                    if can_convert_to_float(score):
                        substrings_float.append(float(score))
                    else:
                        conv.append({"role": "user", "content": self.render("format_error_scores")})
                        response = self._chat(conv)
                        conv.append({"role": "assistant", "content": response})
                        substrings = [s for s in response.split(sep) if s]
                        format_check_time += 1
                        skip_num_check = True
                        break
                if not skip_num_check:  # count check
                    if len(substrings) != len(new_relation_chunks[j]):
                        msg = self.render("format_error_scores_count", n_scores=len(substrings),
                                          n_relations=len(new_relation_chunks[j]))
                        conv.append({"role": "user", "content": msg})
                        response = self._chat(conv)
                        conv.append({"role": "assistant", "content": response})
                        substrings = [s for s in response.split(sep) if s]
                        format_check_time += 1
                    else:
                        format_flag = 1

            if format_flag == 1:  # only well-formed scores are used
                top_k = heapq.nlargest(rp.top_k_per_batch, enumerate(substrings_float), key=lambda x: x[1])
                threshold = rp.min_score_llm if rp.use_llm else rp.min_score_cosine
                for i, _ in top_k:
                    if substrings_float[i] >= threshold:
                        chunk_id = chunk_id_chunks[j][i]
                        source_id = source_chunks[j][i]
                        target_id = target_chunks[j][i]
                        description = description_chunks[j][i]
                        strength = strength_chunks[j][i]
                        # GTRAG.py stored the *last entity of the node* here instead of source_id.
                        stored_source = source_id if self.cfg.retrieval.fix_relation_source else entity_id
                        entry = [stored_source, target_id, description, strength, substrings_float[i]]
                        if chunk_id not in neighbors:
                            neighbors[chunk_id] = [entry]
                        elif entry not in neighbors[chunk_id]:
                            neighbors[chunk_id].append(entry)
                        if source_id not in entity_id_list and source_id not in unique_extract_entities:
                            unique_extract_entities.append(source_id)
                        if target_id not in entity_id_list and target_id not in unique_extract_entities:
                            unique_extract_entities.append(target_id)
            else:
                self.stats.fail_pruning_time += 1
                fail_this_call += 1
                if fail_this_call == 1:
                    self.stats.fail_pruning_query += 1

        return neighbors, [], unique_extract_entities

    # ------------------------------------------------------------------ 3. BFS over chunks
    def retrieve(self, question: str):
        """Build the chunk tree for one question.

        Returns (kg_result, rag_result, query_candidates) where kg_result maps a node ('query' or a
        chunk id) to {"hop": h, "children": [[chunk_id, entities, relations, hop], ...]}.
        """
        cfg = self.cfg.retrieval
        max_hop = cfg.max_hop
        doc, chunks = self.kg.entity_document, self.kg.chunks

        unique_extract_entities, entities_from_query = self.extract_query_entity(question)
        entities_from_query_string, entities_from_query_string_pure = "", ""
        for e in entities_from_query:
            entities_from_query_string += self.render("frag_query_entity", entity=e)
            entities_from_query_string_pure += e + "\n"
        self.log("query entities:", entities_from_query)

        kg_result = defaultdict(lambda: {"hop": None, "children": []})
        rag_result = defaultdict(lambda: {"hop": None, "children": []})
        visited_document = set()
        document_queue = [["query", unique_extract_entities, [], 0]]  # [chunk id, entities, relations, hop]
        query_candidates = None
        current_query_expansion_fail = 0

        while document_queue:
            current_document = document_queue.pop(0)
            cur_id, cur_entities, existing_relation, cur_hop = current_document

            if cur_id != "query" and int(cur_id) not in visited_document:
                visited_document.add(int(cur_id))
            elif cur_id == "query":
                visited_document.add(cur_id)
            if cur_hop >= max_hop:
                continue

            kg_result[cur_id]["hop"] = cur_hop
            rag_result[cur_id]["hop"] = cur_hop
            temp_graph_document = []  # next-level chunks chosen through the knowledge graph

            if cur_id != "query":  # query entities were already expanded to similar entities
                all_extract_entities = []
                for entity_id in cur_entities:
                    all_extract_entities += self.extract_similar_entities(entity_id)
                all_unique_extract_entities = [list(x) for x in set(tuple(x) for x in all_extract_entities)]
            else:
                all_unique_extract_entities = cur_entities

            if cfg.entity_matching_only:
                self._entity_matching_step(question, current_document, all_unique_extract_entities,
                                           temp_graph_document, document_queue, visited_document, kg_result)
                continue

            kg_result[cur_id]["children"] = []
            if cur_id == "query":
                user_input_1 = entities_from_query_string_pure + "\n"
            else:
                relation_set_pure = "".join(rel[2] + "\n" for rel in existing_relation)
                user_input_1 = entities_from_query_string_pure + "\n" + "\n" + relation_set_pure + "\n"

            # ---- (a) neighbours through KG relations
            if cfg.kg_neighbors_per_node > 0:
                entity_id_list = [entity_id for entity_id, _ in all_unique_extract_entities]
                neighbors, _, unique_extract_entities = self.extract_neighbor_relation(
                    entity_id_list, question, existing_relation, visited_document, cur_id,
                    document_queue, entities_from_query_string, user_input_1)
                for key in neighbors:
                    temp_flag = 0
                    for i in range(len(temp_graph_document)):
                        if temp_graph_document[i][0] == key:
                            temp_graph_document[i][1] = merge_entities(temp_graph_document[i][1], unique_extract_entities)
                            temp_graph_document[i][2] = merge_entities(temp_graph_document[i][2], neighbors[key])
                            temp_flag = 1
                            break
                    if temp_flag == 0:
                        temp_graph_document.append([int(key), unique_extract_entities, neighbors[key], cur_hop + 1])

                # keep the best chunks: score = best relation (sqrt(strength * relevance)) x cosine to the chain
                temp_temp_graph_document, neighbor_score = [], []
                query_embedding = self.embedder.encode([user_input_1]) if temp_graph_document else None
                for chunk in temp_graph_document:
                    total_score, relation_temp = 0, ""
                    for relation in chunk[2]:
                        total_score = max(total_score, math.sqrt(relation[3] * relation[4]))
                        if math.sqrt(relation[3] * relation[4]) >= total_score:
                            relation_temp = relation[2]
                    text_embedding = self.embedder.encode([relation_temp])
                    dot_product = np.dot(query_embedding.flatten(), text_embedding.flatten())
                    cosine_score = dot_product / (np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding))
                    neighbor_score.append(total_score * cosine_score)

                k = cfg.kg_neighbors_per_node
                sorted_indices = sorted(range(len(neighbor_score)), key=lambda i: neighbor_score[i], reverse=True)
                if len(sorted_indices) > k:
                    threshold = neighbor_score[sorted_indices[k - 1]]
                    selected = [i for i in sorted_indices if neighbor_score[i] >= threshold]  # keeps ties
                else:
                    selected = sorted_indices
                for chunk_idx in selected:
                    if neighbor_score[chunk_idx] != 0:
                        temp_temp_graph_document.append(temp_graph_document[chunk_idx])
                temp_graph_document = temp_temp_graph_document

            for next_document in temp_graph_document:
                next_existing_relation = merge_entities(existing_relation, next_document[2])
                if not any(child[0] == int(next_document[0]) for child in kg_result[cur_id]["children"]):
                    kg_result[cur_id]["children"].append(
                        [int(next_document[0]), next_document[1], next_existing_relation, next_document[3]])
                if int(next_document[0]) not in visited_document:
                    flag_in_queue = 0
                    for i in range(len(document_queue)):
                        single_document = document_queue[i]
                        if single_document[0] != "query" and int(next_document[0]) == int(single_document[0]) \
                                and next_document[3] == single_document[3]:
                            flag_in_queue = 1
                            document_queue[i][1] = merge_entities(single_document[1], next_document[1])
                            document_queue[i][2] = merge_entities(single_document[2], next_document[2])
                            break
                    if flag_in_queue == 0:
                        document_queue.append(
                            [int(next_document[0]), next_document[1], next_existing_relation, next_document[3]])

            # ---- (b) neighbours through semantic similarity + cross-relation exploration
            top = cfg.semantic_candidates
            _, rows = self._search(self.kg.chunk_index, question if cur_id == "query" else user_input_1, top)
            initial_chunk_list = [int(index) for index in rows[0]]
            if initial_chunk_list and cur_id == "query":
                query_candidates = initial_chunk_list

            across_number = 0  # chunks already added through cross-relations
            for idi_chunk in initial_chunk_list:
                in_queue = any(int(idi_chunk) == int(sd[0]) for sd in document_queue)
                if not (across_number < cfg.cross_neighbors_per_node and int(idi_chunk) not in visited_document
                        and not in_queue):
                    continue
                conv = []
                if cur_id == "query":
                    conv.append({"role": "system", "content": self.render("cross_first_system")})
                    user_input_0 = self.render("user_cross_first", query=question,
                                               query_entities=entities_from_query_string, text=chunks[int(idi_chunk)])
                else:
                    relation_set = "".join(self.render("frag_cross_chain_relation", description=rel[2])
                                           for rel in existing_relation)
                    conv.append({"role": "system", "content": self.render("cross_next_system")})
                    user_input_0 = self.render("user_cross_next", query=question, relation_chain=relation_set,
                                               text=chunks[int(idi_chunk)])
                conv.append({"role": "user", "content": user_input_0})
                response = self._chat(conv, "cross_exploration")
                conv.append({"role": "assistant", "content": response})

                flag_format_check, format_count = 0, 0
                while flag_format_check == 0:
                    substrings = [s.strip() for s in extract_substrings(response, self.d["BEG"], self.d["END"])]
                    if substrings and substrings[0] == "N":
                        flag_format_check = 1
                        self.stats.expansion_refuse_query += 1
                    elif substrings:
                        all_final = []
                        for substring in substrings:
                            for item in [i.strip() for i in substring.split(self.d["RECORD_SEP"])]:
                                all_final.append(item.strip().split(self.d["SEP"]))
                        unique_all_final = [x for i, x in enumerate(all_final) if x not in all_final[:i]]
                        temp_entities, temp_relations = [], []
                        for element in unique_all_final:
                            if len(element) == 4 and "entity" in element[0]:
                                temp_entities.append(element[1] + " " + element[2] + " " + element[3])
                            if len(element) == 4 and "relationship" in element[0]:
                                temp_relations.append([element[1], element[2], element[3]])
                        if temp_entities and temp_relations:
                            kg_result[cur_id]["children"].append([idi_chunk, temp_entities, temp_relations, cur_hop + 1])
                            document_queue.append([int(idi_chunk), temp_entities, temp_relations, cur_hop + 1])
                            flag_format_check = 1
                            across_number += 1

                    format_count += 1
                    if format_count > self.cfg.cross_exploration.max_format_retries:
                        if flag_format_check == 0:
                            self.stats.fail_expansion_time += 1
                            if current_query_expansion_fail == 0:
                                current_query_expansion_fail = 1
                                self.stats.fail_expansion_query += 1
                        break
                    if flag_format_check == 0:
                        conv.append({"role": "user", "content": self.render("format_error_cross")})
                        response = self._chat(conv, "cross_exploration")
                        conv.append({"role": "assistant", "content": response})

        return kg_result, rag_result, query_candidates

    def _entity_matching_step(self, question, current_document, all_unique_extract_entities,
                              temp_graph_document, document_queue, visited_document, kg_result):
        """Ablation: ignore relations; rank chunks by similarity between the question+entity and the chunk."""
        import torch
        import torch.nn.functional as F

        cfg, doc, chunks = self.cfg.retrieval, self.kg.entity_document, self.kg.chunks
        cur_id, cur_hop = current_document[0], current_document[3]
        for entity_id, entity_chunk_id in all_unique_extract_entities:
            entity_embedding = torch.from_numpy(self.embedder.encode([question + " " + doc[int(entity_id)][1]]))
            chunk_embedding = torch.from_numpy(self.embedder.encode([chunks[int(entity_chunk_id)]]))
            relevance_score = F.cosine_similarity(entity_embedding, chunk_embedding, dim=1).item()
            found = False
            for i in range(len(temp_graph_document)):
                if temp_graph_document[i][0] == entity_chunk_id:
                    temp_graph_document[i][2] = temp_graph_document[i][2] + relevance_score
                    found = True
                    break
            if not found:
                temp_graph_document.append([entity_chunk_id, [], relevance_score, cur_hop + 1])
        temp_graph_document.sort(key=lambda x: x[2], reverse=True)
        if len(temp_graph_document) >= cfg.kg_neighbors_per_node:
            del temp_graph_document[cfg.kg_neighbors_per_node:]
        for next_document in temp_graph_document:
            kg_result[cur_id]["children"].append([next_document[0], next_document[1], next_document[2], next_document[3]])
            if next_document[3] < cfg.max_hop and next_document[0] not in visited_document:
                flag_in_queue = 0
                for single_document in document_queue:
                    if next_document[0] == single_document[0] and next_document[3] == single_document[3]:
                        flag_in_queue = 1
                        single_document[1] = merge_entities(single_document[1], next_document[1])
                        single_document[2] = merge_entities(single_document[2], next_document[2])
                        break
                if flag_in_queue == 0:
                    document_queue.append(list(next_document[:4]))

    # ------------------------------------------------------------------ 4. answer generation
    def _short_answer(self, response, conv_history, stage):
        """Pull the answer out of the answer delimiters; re-ask when they are missing."""
        begin, completion = self.d["begin_delimiter"], self.d["completion_delimiter"]
        retries = 0
        while True:
            substrings = [s.strip() for s in extract_substrings(response, begin, completion)]
            if substrings:
                return substrings[0]
            if retries > self.cfg.generation.max_format_retries:
                # CAP: the original looped forever here; return the raw reply instead
                self.log("WARNING: answer delimiters missing; returning the raw reply")
                return response
            conv_history.append({"role": "user", "content": self.render("format_error_answer")})
            response = self._chat(conv_history, stage)
            conv_history.append({"role": "assistant", "content": response})
            retries += 1

    def generate_answer_tree(self, kg_result, rag_result, query):
        """Bottom-up: summarise the leaves, feed the summaries to their parents, answer at the root."""
        if kg_result == {} and rag_result == {}:
            return None
        short = self.cfg.generation.answer_mode == "short"
        chunks = self.kg.chunks
        current_hop = self.cfg.retrieval.max_hop - 1
        temp_answer, response, conv_history = {}, None, []
        while current_hop >= 0:
            for key in kg_result:
                if kg_result[key]["hop"] != current_hop:
                    continue
                conv_history = []
                if current_hop == 0:  # root: final answer
                    conv_history.append({"role": "system", "content": self.render(
                        "answer_short_system" if short else "answer_open_system")})
                    user_input = self.render("user_answer_header", query=query)
                else:
                    conv_history.append({"role": "system", "content": self.render("chunk_merge_system")})
                    relation_set = ""  # relation chain that led to this chunk
                    for farther_key in kg_result:
                        if kg_result[farther_key]["hop"] == current_hop - 1:
                            for farther_child in kg_result[farther_key]["children"]:
                                if int(farther_child[0]) == int(key):
                                    # entity_matching_only stores a score here instead of relations; the
                                    # original crashed on it, now the chain is simply empty
                                    for rel in (farther_child[2] if isinstance(farther_child[2], list) else []):
                                        relation_set += self.render("frag_merge_chain_relation", description=rel[2])
                                    break
                    user_input = self.render("user_merge_header", query=query,
                                             relation_chain=relation_set if relation_set != "" else "None\n")
                if key != "query":
                    user_input += self.render("user_node_text", text=chunks[int(key)])
                for child in kg_result[key]["children"]:
                    text = temp_answer[int(child[0])] if int(child[0]) in temp_answer else chunks[int(child[0])]
                    user_input += self.render("user_child_text", text=text)

                if user_input != "":
                    conv_history.append({"role": "user", "content": user_input})
                    response = self._chat(conv_history, "answer_generation_kg")
                    self.log("GENERATED ANSWER:", response)
                    if key != "query":
                        temp_answer[int(key)] = response
            current_hop -= 1

        if short:
            return self._short_answer(response, conv_history, "answer_generation_kg")
        return response

    def generate_answer_flat(self, kg_result, rag_result, query):
        """Flat baseline: put every retrieved chunk into one prompt, no tree."""
        if kg_result == {} and rag_result == {}:
            return None
        short = self.cfg.generation.answer_mode == "short"
        texts = ""
        for hop in range(self.cfg.retrieval.max_hop - 1, -1, -1):
            for key in kg_result:
                if kg_result[key]["hop"] == hop:
                    for child in kg_result[key]["children"]:
                        texts = self.render("user_flat_text", text=self.kg.chunks[int(child[0])]) + texts
        conv_history = [
            {"role": "system", "content": self.render("answer_short_system" if short else "answer_open_system")},
            {"role": "user", "content": self.render("user_flat_header", query=query, texts=texts)}]
        response = self._chat(conv_history, "answer_generation_rag")
        self.log("GENERATED ANSWER (flat):", response)
        if short:
            return self._short_answer(response, conv_history, "answer_generation_rag")
        return response

    # ------------------------------------------------------------------ one question
    def answer_question(self, question: str) -> dict:
        kg_result, rag_result, query_candidates = self.retrieve(question)
        answer_kg = self.generate_answer_tree(kg_result, rag_result, question)
        answer_rag = self.generate_answer_flat(kg_result, rag_result, question) \
            if self.cfg.generation.also_flat else None
        return {"answer_kg": answer_kg, "answer_rag": answer_rag, "kg_result": kg_result,
                "rag_result": rag_result, "query_candidates": query_candidates}
