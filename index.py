"""Step 3: build the index (do the expensive embedding work once, save it).

Run: python index.py
Creates:  index/vectors.npy  (one row per chunk)   index/chunks.json  (the text for each row)
Row i of vectors.npy is the embedding of chunks[i] -- that's the whole "database".
"""
import json
import os

import numpy as np
import ollama
from tqdm import tqdm

from data import load_questions

MODEL = "nomic-embed-text"
BATCH = 32


def build_corpus(questions):
    """Pool every question's paragraphs into one list, dropping duplicates.

    Chunking: HotpotQA paragraphs are already short, so 1 paragraph = 1 chunk.
    (With long documents you'd split them into pieces here.)
    """
    seen, chunks = set(), []
    for q in questions:
        for p in q["paragraphs"]:
            if p["title"] not in seen:
                seen.add(p["title"])
                chunks.append(p)
    return chunks


def embed_documents(texts):
    out = []
    for i in tqdm(range(0, len(texts), BATCH), desc="embedding"):
        batch = ["search_document: " + t for t in texts[i:i + BATCH]]
        out.extend(ollama.embed(model=MODEL, input=batch)["embeddings"])
    vecs = np.array(out, dtype=np.float32)
    # Normalise each vector to length 1. Then cosine similarity == plain dot product,
    # which makes search a single matrix multiplication.
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


if __name__ == "__main__":
    chunks = build_corpus(load_questions(200))
    print(f"{len(chunks)} unique paragraphs to embed")
    vecs = embed_documents([c["title"] + ". " + c["text"] for c in chunks])

    os.makedirs("index", exist_ok=True)
    np.save("index/vectors.npy", vecs)
    with open("index/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    print("vectors shape:", vecs.shape, "->", f"{vecs.nbytes / 1e6:.1f} MB")
