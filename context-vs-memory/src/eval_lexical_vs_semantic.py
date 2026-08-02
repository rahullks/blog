"""
Does swapping the TF-IDF stand-in for real embeddings actually change anything?

sim.py scores relevance with TF-IDF, which matches on literal word overlap. The
obvious worry is that this flatters the results: QUERIES in sim.py mostly reuse
the words of the fact they are asking about, so a lexical matcher gets the
answer for free.

This script runs the same recall measurement twice, once with the original
QUERIES and once with PARAPHRASED_QUERIES (same facts, none of the same words),
against both scorers, so the two effects can be told apart.

    python eval_lexical_vs_semantic.py

Downloads ~130MB on first run (the local embedding model), then runs offline.
"""
import time

import sim
from sim import (QUERIES, PARAPHRASED_QUERIES, generate_conversation,
                 extract_memory_store, approach_context_packing,
                 approach_memory_retrieval)

CONVERSATION_LENGTHS = [120, 300]
TOKEN_BUDGET = 120
MAX_SCORABLE_HISTORY = 300


def recall_for(scorer, query_set, num_turns):
    """Fraction of facts whose expected snippet lands in the assembled context."""
    conversation = generate_conversation(num_turns, seed=7)
    memory_store = extract_memory_store(conversation)

    packing_hits = 0
    memory_hits = 0
    for _, question, expected in query_set:
        text, _, _ = approach_context_packing(
            conversation, question, TOKEN_BUDGET,
            max_scorable_history=MAX_SCORABLE_HISTORY, scorer=scorer)
        packing_hits += expected.lower() in text.lower()

        text, _, _ = approach_memory_retrieval(
            memory_store, question, TOKEN_BUDGET, scorer=scorer)
        memory_hits += expected.lower() in text.lower()

    n = len(query_set)
    return packing_hits / n, memory_hits / n


def main():
    from local_backend import LocalEmbeddingScorer

    t0 = time.time()
    local = LocalEmbeddingScorer()
    print(f"loaded {local.model_name} in {time.time() - t0:.1f}s\n")

    scorers = [
        ("TF-IDF (lexical)", sim.rank_by_relevance),
        (f"{local.model_name} (semantic)", local.rank_by_relevance),
    ]
    query_sets = [
        ("original QUERIES (shares words with the fact)", QUERIES),
        ("PARAPHRASED (shares no words with the fact)", PARAPHRASED_QUERIES),
    ]

    header = f"{'scorer':<40} {'query set':<46} {'turns':>6} {'packing':>9} {'memory':>8}"
    print(header)
    print("-" * len(header))
    for scorer_name, scorer in scorers:
        for set_name, query_set in query_sets:
            for num_turns in CONVERSATION_LENGTHS:
                packing, memory = recall_for(scorer, query_set, num_turns)
                print(f"{scorer_name:<40} {set_name:<46} {num_turns:>6} "
                      f"{packing:>9.2f} {memory:>8.2f}")

    print(f"\nembedding calls made: {local.texts_embedded} unique texts "
          f"in {local.embed_batches} batches")
    print("(the filler pool is small and turns repeat, so unique texts stay far "
          "below turn count)")


if __name__ == "__main__":
    main()
