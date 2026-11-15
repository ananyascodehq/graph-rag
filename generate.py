"""Step 5: generation = put the retrieved chunks + question into a prompt, ask the LLM.

Run: python generate.py                     (demo question, k=3)
     python generate.py "your question" 5   (your question, k=5)
"""
import sys

import ollama

from retrieve import retrieve

LLM = "llama3.1"

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain enough information to answer, reply exactly: I don't know.
Keep the answer short (a few words).

Context:
{context}

Question: {question}
Answer:"""


def answer(question, k=3, show=True, retriever=retrieve):
    hits = retriever(question, k)
    context = "\n\n".join(f"[{c['title']}] {c['text']}" for _, c in hits)
    prompt = PROMPT.format(context=context, question=question)
    resp = ollama.chat(model=LLM, messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0})          # 0 = deterministic, same answer each run
    text = resp["message"]["content"].strip()
    if show:
        print("Retrieved:", [c["title"] for _, c in hits])
    return text, hits


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else (
        "What government position was held by the woman who portrayed "
        "Corliss Archer in the film Kiss and Tell?")
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print("Q:", q, f"(k={k})")
    print("A:", answer(q, k)[0])
