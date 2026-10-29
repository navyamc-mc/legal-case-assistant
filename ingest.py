"""
ingest.py - Read case documents, CHUNK them, and store them in the VECTOR DATABASE (ChromaDB).

Folder layout:  cases/<Case_Name>/judgment.pdf, order_xxx.pdf, proof_xxx.pdf (or .txt)
Run:  python ingest.py
Chunking strategy (set CHUNK_STRATEGY):  "section" (default) or "words"
"""
import os
import re
import sys

import chromadb
from pypdf import PdfReader

CASES_DIR = "cases"
DB_DIR = "chroma_db"
COLLECTION = "cases"
CHUNK_WORDS = 300
OVERLAP_WORDS = 60
STRATEGY = os.environ.get("CHUNK_STRATEGY", "section")
SECTION_RE = re.compile(r"\b(FACTS|ISSUES|ISSUE|ARGUMENTS|REASONING|HELD|FINAL ORDER)\s*:")


def guess_doc_type(filename):
    n = filename.lower()
    if "judg" in n:
        return "judgment"
    if "order" in n:
        return "order"
    if "proof" in n or "exhibit" in n or "evidence" in n:
        return "proof"
    return "other"


def read_pages(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return [(i, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages, start=1)]
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [(1, f.read())]
    return []


def chunk_text(text):
    """Fixed-size word chunks with overlap."""
    words = [w for w in re.sub(r"\s+", " ", text).strip().split(" ") if w]
    chunks, step = [], CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def split_sections(text):
    """Split a judgment at headings such as FACTS:, ISSUES:, REASONING:, FINAL ORDER:."""
    marks = list(SECTION_RE.finditer(text))
    if not marks:
        return [("General", text)]
    parts = []
    if text[:marks[0].start()].strip():
        parts.append(("Header", text[:marks[0].start()]))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        parts.append((m.group(1).title(), text[m.end():end]))
    return parts


def chunk_document(text, strategy=STRATEGY):
    """Return a list of (section, chunk_text)."""
    if strategy == "words":
        return [("General", c) for c in chunk_text(text)]
    out = []
    for section, body in split_sections(text):
        for c in chunk_text(body):
            out.append((section, f"{section}: {c}"))
    return out


def build_records(case_name, filename, file_path, strategy=STRATEGY):
    if not filename.lower().endswith((".pdf", ".txt")):
        return [], [], []
    doc_type = guess_doc_type(filename)
    ids, docs, metas = [], [], []
    for page_no, page_text in read_pages(file_path):
        for i, (section, chunk) in enumerate(chunk_document(page_text, strategy)):
            ids.append(f"{case_name}|{filename}|{page_no}|{i}")
            docs.append(chunk)
            metas.append({"case": case_name, "file": filename, "page": page_no,
                          "doc_type": doc_type, "section": section})
    return ids, docs, metas


def main():
    if not os.path.isdir(CASES_DIR):
        print(f"Folder '{CASES_DIR}/' not found. Run this from inside the legal-case-qa folder.")
        sys.exit(1)
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    total = 0
    for case_name in sorted(os.listdir(CASES_DIR)):
        case_path = os.path.join(CASES_DIR, case_name)
        if not os.path.isdir(case_path):
            continue
        n = 0
        for filename in sorted(os.listdir(case_path)):
            ids, docs, metas = build_records(case_name, filename, os.path.join(case_path, filename))
            if docs:
                collection.add(ids=ids, documents=docs, metadatas=metas)
                n += len(docs)
            elif filename.lower().endswith((".pdf", ".txt")):
                print(f"  ! No text found in {filename} (scanned PDF? it needs OCR)")
        print(f"{case_name}: {n} chunks")
        total += n
    print(f"\nDone. {total} chunks stored in '{DB_DIR}/' using the '{STRATEGY}' strategy.")


if __name__ == "__main__":
    main()
