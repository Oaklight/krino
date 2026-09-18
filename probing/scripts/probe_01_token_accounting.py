#!/usr/bin/env python3
"""Probe 1: Token accounting — verify tool-call template hypothesis.

Systematically varies question IDs, question counts, and question types
to fit a linear model to token usage and identify the prompt template.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "01-token-accounting.jsonl")

FIXED_STATE = "The customer has been waiting for three days and wants a status update on order #12345."


def run_probe(client: JevClient, probe_name: str, questions: dict, state: str = FIXED_STATE) -> dict:
    t0 = time.monotonic()
    resp = client.ask(state, questions)
    elapsed = time.monotonic() - t0
    record = {
        "probe": probe_name,
        "state_len": len(state),
        "num_questions": len(questions),
        "question_ids": list(questions.keys()),
        "question_types": [q["type"] for q in questions.values()],
        "input_tokens": resp["usage"]["input_tokens"],
        "output_tokens": resp["usage"]["output_tokens"],
        "elapsed_s": round(elapsed, 4),
        "model": resp["model"],
    }
    return record


def main():
    client = JevClient()
    results = []

    def emit(record):
        results.append(record)
        with open(OUTFILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  {record['probe']:40s} in={record['input_tokens']:4d} out={record['output_tokens']:3d} ({record['elapsed_s']:.2f}s)")

    print("=" * 70)
    print("PROBE 1: Token Accounting")
    print("=" * 70)

    # --- 1a: Vary question ID length ---
    print("\n--- 1a: Vary question ID length (single noul) ---")
    for id_len in [1, 2, 3, 5, 10, 15, 20, 30, 50]:
        qid = "q" * id_len
        questions = {qid: {"type": "noul", "instructions": "Is this urgent?"}}
        emit(run_probe(client, f"id_len_{id_len}", questions))

    # --- 1b: Vary question count (fixed IDs) ---
    print("\n--- 1b: Vary question count (noul, short IDs) ---")
    base_questions_pool = [
        ("q1", "Is this urgent?"),
        ("q2", "Is the customer angry?"),
        ("q3", "Is this a refund request?"),
        ("q4", "Is this a complaint?"),
        ("q5", "Does this mention a product defect?"),
        ("q6", "Is the customer a repeat buyer?"),
        ("q7", "Is shipping mentioned?"),
        ("q8", "Is there a billing issue?"),
        ("q9", "Is this time-sensitive?"),
        ("q10", "Does this need escalation?"),
        ("q11", "Is the tone polite?"),
        ("q12", "Is this about an exchange?"),
        ("q13", "Does this mention a competitor?"),
        ("q14", "Is there a safety concern?"),
        ("q15", "Is this from an enterprise customer?"),
        ("q16", "Does this mention social media?"),
        ("q17", "Is legal action threatened?"),
        ("q18", "Is this a duplicate message?"),
        ("q19", "Does this need a manager?"),
        ("q20", "Is the customer satisfied?"),
    ]
    for count in [1, 2, 3, 5, 10, 15, 20]:
        questions = {qid: {"type": "noul", "instructions": instr} for qid, instr in base_questions_pool[:count]}
        emit(run_probe(client, f"count_{count}_noul", questions))

    # --- 1c: Vary question type (fixed count=1, same ID) ---
    print("\n--- 1c: Vary question type (single question, id='q') ---")
    noul_q = {"q": {"type": "noul", "instructions": "Is this urgent?"}}
    emit(run_probe(client, "type_noul", noul_q))

    choice_q = {"q": {"type": "choice", "instructions": "Which department?",
                       "criteria": {"billing": "Payment issues", "shipping": "Delivery issues", "returns": "Returns and exchanges"}}}
    emit(run_probe(client, "type_choice_3opt", choice_q))

    choice_q5 = {"q": {"type": "choice", "instructions": "Which department?",
                        "criteria": {"billing": "Payment", "shipping": "Delivery", "returns": "Returns", "tech": "Technical", "sales": "Sales"}}}
    emit(run_probe(client, "type_choice_5opt", choice_q5))

    score_q = {"q": {"type": "score", "instructions": "How frustrated is the customer?",
                      "criteria": ["Calm", "Mildly frustrated", "Very frustrated"]}}
    emit(run_probe(client, "type_score_3lvl", score_q))

    score_q5 = {"q": {"type": "score", "instructions": "How frustrated is the customer?",
                       "criteria": ["Calm", "Slightly annoyed", "Frustrated", "Very frustrated", "Furious"]}}
    emit(run_probe(client, "type_score_5lvl", score_q5))

    # --- 1d: Mixed question types ---
    print("\n--- 1d: Mixed question types ---")
    mixed = {
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "dept": {"type": "choice", "instructions": "Which department?",
                 "criteria": {"billing": "Payment", "shipping": "Delivery", "returns": "Returns"}},
        "frustration": {"type": "score", "instructions": "How frustrated?",
                        "criteria": ["Calm", "Frustrated", "Furious"]},
    }
    emit(run_probe(client, "mixed_3types", mixed))

    # --- 1e: Vary state length (fixed question) ---
    print("\n--- 1e: Vary state length ---")
    base_sentence = "The quick brown fox jumps over the lazy dog. "
    for multiplier in [1, 5, 10, 20, 50, 100]:
        state = base_sentence * multiplier
        questions = {"q": {"type": "noul", "instructions": "Is this urgent?"}}
        emit(run_probe(client, f"state_len_{len(state)}", questions, state=state))

    # --- 1f: Constant instruction, vary only question ID ---
    print("\n--- 1f: Same instruction, different ID lengths (isolate ID token cost) ---")
    for id_len in [1, 5, 10, 20, 50, 100]:
        qid = "x" * id_len
        questions = {qid: {"type": "noul", "instructions": "Is this urgent?"}}
        emit(run_probe(client, f"id_isolate_{id_len}", questions))

    client.close()

    print(f"\n{'=' * 70}")
    print(f"Done. {len(results)} records written to {OUTFILE}")

    # Quick analysis
    print("\n--- Quick Analysis ---")
    id_probes = [r for r in results if r["probe"].startswith("id_len_")]
    if len(id_probes) >= 2:
        first, last = id_probes[0], id_probes[-1]
        id_diff = len(last["question_ids"][0]) - len(first["question_ids"][0])
        tok_diff = last["input_tokens"] - first["input_tokens"]
        print(f"ID length {len(first['question_ids'][0])}→{len(last['question_ids'][0])}: "
              f"token diff = {tok_diff}, chars/token ≈ {id_diff / tok_diff:.2f}" if tok_diff > 0 else "no token change")

    count_probes = [r for r in results if r["probe"].startswith("count_")]
    if len(count_probes) >= 2:
        for i in range(1, len(count_probes)):
            prev, curr = count_probes[i - 1], count_probes[i]
            dq = curr["num_questions"] - prev["num_questions"]
            dt = curr["input_tokens"] - prev["input_tokens"]
            print(f"Questions {prev['num_questions']}→{curr['num_questions']}: "
                  f"+{dt} input tokens ({dt / dq:.1f} tokens/question)" if dq > 0 else "")


if __name__ == "__main__":
    main()
