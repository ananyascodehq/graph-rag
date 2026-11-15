"""Step 7: build the graph.

Run: python graph.py
Creates: index/graph.json   {title: [titles it is linked to]}

Graph = nodes + edges.
  node : one paragraph (identified by its title, e.g. "Shirley Temple")
  edge : paragraph A --> paragraph B  if A's text mentions B's title

Example: "Kiss and Tell is a 1945 film starring Shirley Temple as Corliss Archer."
         mentions "Shirley Temple"  =>  edge  Kiss and Tell --> Shirley Temple
That edge is exactly the hop that vector search can't see.

(No LLM needed. Real GraphRAG systems often use an LLM to extract (subject, relation, object)
triples instead -- richer, but ~2000 slow LLM calls. This is the cheap version of the same idea.)
"""
import json
import re
from collections import defaultdict


def short_name(title):
    """'Kiss and Tell (1945 film)' -> 'kiss and tell'  (drop the disambiguation, lowercase)."""
    return re.sub(r"\s*\(.*?\)\s*$", "", title).lower()


def build_graph(chunks):
    names = {c["title"]: short_name(c["title"]) for c in chunks}
    links = defaultdict(set)
    for a in chunks:
        text = a["text"].lower()
        for title, name in names.items():
            if title == a["title"] or len(name) < 4:        # skip self-links and tiny names like "Q"
                continue
            # cheap substring test first, then a whole-word check so "Ed" doesn't match "Edward"
            if name in text and re.search(r"\b" + re.escape(name) + r"\b", text):
                links[a["title"]].add(title)
    return {t: sorted(v) for t, v in links.items()}


if __name__ == "__main__":
    with open("index/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    graph = build_graph(chunks)

    with open("index/graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f)

    n_edges = sum(len(v) for v in graph.values())
    print(f"{len(chunks)} nodes, {n_edges} edges "
          f"({len(graph)} nodes have at least one outgoing link)")
    for t in ("Kiss and Tell (1945 film)", "Shirley Temple"):
        print(f"\n{t} -->", graph.get(t, []))
