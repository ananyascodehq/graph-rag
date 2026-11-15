"""Step 4: retrieval = embed the question, find the k closest stored vectors.

Run: python retrieve.py            (demo on a multi-hop question)
"""
import json

import numpy as np
import ollama

MODEL = "nomic-embed-text"

vectors = np.load("index/vectors.npy")                     # (n_chunks, 768), each row length 1
with open("index/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)                                  # chunks[i] is the text of vectors[i]


def embed_query(question):
    q = np.array(ollama.embed(model=MODEL, input="search_query: " + question)["embeddings"][0],
                 dtype=np.float32)
    return q / np.linalg.norm(q)


def retrieve(question, k=10):
    """Return the k most similar chunks as (score, chunk) pairs, best first."""
    q = embed_query(question)
    scores = vectors @ q                                  # cosine similarity with every chunk at once
    top = np.argsort(-scores)[:k]                          # indices of the k highest scores
    return [(float(scores[i]), chunks[i]) for i in top]


if __name__ == "__main__":
    question = ("What government position was held by the woman who portrayed "
                "Corliss Archer in the film Kiss and Tell?")
    needed = {"Kiss and Tell (1945 film)", "Shirley Temple"}
    print("Q:", question, "\nNeeded:", sorted(needed), "\n")
    for rank, (score, c) in enumerate(retrieve(question, k=3), 1):
        mark = "  <-- NEEDED" if c["title"] in needed else ""
        print(f"{rank:2d}. {score:.3f}  {c['title']}{mark}")
