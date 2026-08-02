"""
Is the lexical-vs-semantic result an artifact of picking one small model?

Runs the same recall measurement across several open-source embedding models of
different sizes and families. If the numbers cluster, the conclusion is a
property of semantic-vs-lexical scoring rather than of any one model, and it is
reasonable to expect a larger or hosted model to land in the same place.

    python eval_across_models.py

Downloads each model on first use (roughly 0.07GB to 0.64GB each), then caches.
"""
import time

import pandas as pd

import sim
from sim import QUERIES, PARAPHRASED_QUERIES, run_eval
from local_backend import LocalEmbeddingScorer

MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "snowflake/snowflake-arctic-embed-s",
    "sentence-transformers/all-MiniLM-L6-v2",
]

TURNS = [300]


def measure(scorer):
    out = {}
    for set_name, qset in [("original", QUERIES), ("paraphrased", PARAPHRASED_QUERIES)]:
        rows = run_eval(conversation_lengths=TURNS, token_budget=120, seed=7,
                        max_scorable_history=300, query_set=qset, scorer=scorer)
        df = pd.DataFrame(rows)
        agg = df.groupby("approach")["hit"].mean()
        out[(set_name, "packing")] = agg["context_packing"]
        out[(set_name, "memory")] = agg["memory_retrieval"]
    return out


def main():
    results = []

    base = measure(None)  # sim.rank_by_relevance, the TF-IDF baseline
    results.append(("TF-IDF (lexical baseline)", "n/a", base))

    for name in MODELS:
        t0 = time.time()
        scorer = LocalEmbeddingScorer(model_name=name)
        load = time.time() - t0
        results.append((name, f"{load:.0f}s", measure(scorer.rank_by_relevance)))

    print(f"\nRecall at {TURNS[0]} turns, 40 facts, 120-token budget\n")
    hdr = (f"{'scorer':<42} {'load':>6} | {'orig pack':>9} {'orig mem':>8} | "
           f"{'para pack':>9} {'para mem':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, load, r in results:
        print(f"{name:<42} {load:>6} | {r[('original','packing')]:>9.2f} "
              f"{r[('original','memory')]:>8.2f} | {r[('paraphrased','packing')]:>9.2f} "
              f"{r[('paraphrased','memory')]:>8.2f}")


if __name__ == "__main__":
    main()
