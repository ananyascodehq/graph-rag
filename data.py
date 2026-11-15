"""Step 1: load a small slice of HotpotQA.

HotpotQA 'distractor' setting: every question comes with ~10 paragraphs,
2+ of which are needed to answer it ("supporting") and the rest are distractors.
"""
from datasets import load_dataset


def load_questions(n=200):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    questions = []
    for row in ds.select(range(n)):
        # context = {"title": [...], "sentences": [[...], ...]}  -> one paragraph per title
        paragraphs = [
            {"title": t, "text": " ".join(s)}
            for t, s in zip(row["context"]["title"], row["context"]["sentences"])
        ]
        questions.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],                                  # "bridge" (multi-hop chain) or "comparison"
            "supporting_titles": sorted(set(row["supporting_facts"]["title"])),
            "paragraphs": paragraphs,
        })
    return questions


if __name__ == "__main__":
    qs = load_questions()
    print(f"Loaded {len(qs)} questions\n")
    for q in qs[:3]:
        print("Q:", q["question"])
        print("A:", q["answer"], f"  [{q['type']}]")
        print("Needs paragraphs:", q["supporting_titles"])
        print(f"(question comes with {len(q['paragraphs'])} paragraphs; first one:)")
        print("  ", q["paragraphs"][0]["title"], "->", q["paragraphs"][0]["text"][:150], "...\n")
