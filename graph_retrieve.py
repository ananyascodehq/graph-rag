"""Step 8: graph retrieval = vector search for seeds, then walk the graph to their neighbours.

Run: python graph_retrieve.py     (same demo question as retrieve.py, side by side)

Algorithm:
  1. SEEDS     : top n_seeds paragraphs by vector similarity (normal retrieval).
  2. EXPAND    : collect every paragraph linked to a seed (either direction).
  3. RANK      : score those neighbours against the question; the best ones fill the remaining slots.
Result = seeds + best neighbours, k total. Seeds come first because we trust vector search most.
"""
import json
from collections import defaultdict

from retrieve import chunks, embed_query, retrieve, vectors

with open("index/graph.json", encoding="utf-8") as f:
    out_links = json.load(f)

index_of = {c["title"]: i for i, c in enumerate(chunks)}          # title -> row number

# Edges are stored one-way (A mentions B). For walking we want both ways:
# a seed might be the paragraph being mentioned, not the one doing the mentioning.
neighbours = defaultdict(set)
for a, targets in out_links.items():
    for b in targets:
        neighbours[a].add(b)
        neighbours[b].add(a)


def graph_retrieve(question, k=10, n_seeds=2):
    q = embed_query(question)
    scores = vectors @ q
    ranked = [int(i) for i in (-scores).argsort()]                # all chunks, best first

    seeds = ranked[:min(n_seeds, k)]
    seed_set = set(seeds)

    # 2. expand: every neighbour of a seed that isn't already a seed
    candidates = {index_of[t] for s in seeds for t in neighbours[chunks[s]["title"]]} - seed_set
    # 3. rank neighbours by how well they match the question itself
    best_neighbours = sorted(candidates, key=lambda i: -scores[i])

    result = seeds + best_neighbours[:k - len(seeds)]
    for i in ranked:                                              # not enough neighbours? fill with plain vector hits
        if len(result) >= k:
            break
        if i not in result:
            result.append(i)
    return [(float(scores[i]), chunks[i]) for i in result]


if __name__ == "__main__":
    question = ("What government position was held by the woman who portrayed "
                "Corliss Archer in the film Kiss and Tell?")
    needed = {"Kiss and Tell (1945 film)", "Shirley Temple"}
    print("Q:", question, "\nNeeded:", sorted(needed))
    for name, fn in (("vector only", retrieve), ("vector + graph", graph_retrieve)):
        print(f"\n{name}, k=3")
        for rank, (score, c) in enumerate(fn(question, 3), 1):
            mark = "  <-- NEEDED" if c["title"] in needed else ""
            print(f"{rank:2d}. {score:.3f}  {c['title']}{mark}")
