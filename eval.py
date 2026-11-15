"""Step 6: measure the system instead of eyeballing single questions.

Run: python eval.py                              retrieval only (fast, no LLM)
     python eval.py --answers --n 40 --ks 3 10   also generate answers (slow: one LLM call per question per k)

Retrieval metric: for each question we know which paragraphs are NEEDED (supporting_titles).
  all-found % : fraction of questions where EVERY needed paragraph is in the top k  <- the multi-hop number
  avg found % : average fraction of needed paragraphs that made it into the top k
"""
import argparse
import re
import string
from collections import Counter, defaultdict

from tqdm import tqdm

from data import load_questions
from generate import answer
from graph_retrieve import graph_retrieve
from retrieve import retrieve

KS = [1, 3, 5, 10, 20]


def retrieval_eval(questions, retriever=retrieve):
    # stats[type][k] = [n_questions, sum of fraction found, n_all_found]
    stats = defaultdict(lambda: {k: [0, 0.0, 0] for k in KS})
    for q in tqdm(questions, desc="retrieving"):
        needed = set(q["supporting_titles"])
        # NOTE: graph retrieval's result depends on k (seeds + neighbours fill k slots), so for it
        # we must call once per k. Plain vector search is k-independent, so one call is enough.
        if retriever is retrieve:
            full = [c["title"] for _, c in retriever(q["question"], max(KS))]
            by_k = {k: full[:k] for k in KS}
        else:
            by_k = {k: [c["title"] for _, c in retriever(q["question"], k)] for k in KS}
        for k in KS:
            found = needed & set(by_k[k])
            for key in (q["type"], "ALL"):
                s = stats[key][k]
                s[0] += 1
                s[1] += len(found) / len(needed)
                s[2] += len(found) == len(needed)
    return stats


def print_retrieval(stats):
    for title, idx, fmt in (("all-found %  (both/every needed paragraph in top k)", 2, "all"),
                            ("avg found %  (average share of needed paragraphs in top k)", 1, "avg")):
        print(f"\n{title}")
        print(f"{'type':<12}{'n':>5}" + "".join(f"{'k=' + str(k):>8}" for k in KS))
        for key in sorted(stats, key=lambda t: (t == "ALL", t)):
            n = stats[key][KS[0]][0]
            print(f"{key:<12}{n:>5}" + "".join(f"{100 * stats[key][k][idx] / n:>7.1f}%" for k in KS))


# --- answer scoring (standard HotpotQA-style normalisation) ---------------
def normalize(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_eval(questions, ks, retriever=retrieve):
    for k in ks:
        agg = defaultdict(lambda: [0, 0, 0.0, 0])        # n, exact matches, f1 sum, "I don't know" count
        for q in tqdm(questions, desc=f"answering k={k}"):
            pred, _ = answer(q["question"], k, show=False, retriever=retriever)
            for key in (q["type"], "ALL"):
                a = agg[key]
                a[0] += 1
                a[1] += normalize(pred) == normalize(q["answer"])
                a[2] += f1(pred, q["answer"])
                a[3] += pred.lower().startswith("i don't know")
        print(f"\nAnswers, k={k}")
        print(f"{'type':<12}{'n':>5}{'EM':>8}{'F1':>8}{'IDK':>8}")
        for key in sorted(agg, key=lambda t: (t == "ALL", t)):
            n, em, f, idk = agg[key]
            print(f"{key:<12}{n:>5}{100 * em / n:>7.1f}%{100 * f / n:>7.1f}%{100 * idk / n:>7.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also run the LLM on the first --n questions")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 10])
    ap.add_argument("--graph", action="store_true", help="use vector+graph retrieval instead of vector only")
    args = ap.parse_args()

    retriever = graph_retrieve if args.graph else retrieve
    print("Retriever:", "vector + graph" if args.graph else "vector only")
    questions = load_questions(200)
    print_retrieval(retrieval_eval(questions, retriever))
    if args.answers:
        answer_eval(questions[:args.n], args.ks, retriever)
