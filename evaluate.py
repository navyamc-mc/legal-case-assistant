"""
evaluate.py - Mini BENCHMARK for the retrieval part of the RAG system. No AI key needed.
Run:  python evaluate.py
For each question in questions.json it checks whether the right text appears in the top-K chunks,
and compares the two chunking strategies. Metrics: Hit@K and MRR.
"""
import json
import os

import chromadb

from ingest import CASES_DIR, build_records

K = 3


def build_collection(strategy):
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("eval")
    except Exception:
        pass
    col = client.create_collection("eval")
    for case in sorted(os.listdir(CASES_DIR)):
        cp = os.path.join(CASES_DIR, case)
        if not os.path.isdir(cp):
            continue
        for fn in sorted(os.listdir(cp)):
            ids, docs, metas = build_records(case, fn, os.path.join(cp, fn), strategy)
            if docs:
                col.add(ids=ids, documents=docs, metadatas=metas)
    return col


def evaluate(col, questions):
    hits, rr, misses = 0, 0.0, []
    for q in questions:
        res = col.query(query_texts=[q["question"]], n_results=K, where={"case": q["case"]})
        rank = next((i + 1 for i, d in enumerate(res["documents"][0])
                     if q["keyword"].lower() in d.lower()), None)
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            misses.append(q["question"])
    n = len(questions)
    return hits / n, rr / n, misses


if __name__ == "__main__":
    with open("questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"{len(questions)} test questions, top-{K} retrieval\n")
    print(f"{'Chunking strategy':<20}{'Hit@' + str(K):<10}{'MRR':<8}")
    for strategy in ["words", "section"]:
        hit, mrr, misses = evaluate(build_collection(strategy), questions)
        print(f"{strategy:<20}{hit:<10.2f}{mrr:<8.2f}")
        for m in misses:
            print(f"    missed: {m}")
    print("\nHit@K = how often the right chunk is in the top K. MRR = how high it is ranked (1.0 is best).")
