"""
What happens to memory retrieval when the extractor is not a perfect oracle.

sim.extract_memory_store() reads a flag set by the conversation generator, so it
never has to decide what is worth remembering. A production memory-writing step
is an LLM call that misses things. This script simulates that by dropping a
share of facts before they ever reach the store, which is the optimistic model
of a bad extractor: everything that survives is recorded perfectly.

Context packing is shown alongside it for comparison and is unaffected, because
it reads the raw conversation and has no extraction step to lose anything in.

    python eval_lossy_extraction.py
"""
import collections
import random

from sim import (QUERIES, generate_conversation, extract_memory_store,
                 approach_memory_retrieval, run_eval)

LENGTHS = [80, 240, 480, 800]
EXTRACTION_RECALL = [1.0, 0.9, 0.8, 0.6, 0.4]
TOKEN_BUDGET = 120
SEED = 7


def memory_recall_with_lossy_extractor(num_turns: int, keep: float, seed: int = 11) -> float:
    conversation = generate_conversation(num_turns, seed=SEED)
    store = extract_memory_store(conversation)
    rng = random.Random(seed)
    surviving = [t for t in store if rng.random() < keep]
    fact_turn = {t.fact_key: t.text for t in conversation if t.is_fact}

    hits = 0
    for fact_key, question, _ in QUERIES:
        text, _, _ = approach_memory_retrieval(surviving, question, TOKEN_BUDGET)
        hits += fact_turn[fact_key] in text
    return hits / len(QUERIES)


def main():
    packing = collections.defaultdict(list)
    for row in run_eval(LENGTHS, token_budget=TOKEN_BUDGET, seed=SEED,
                        max_scorable_history=300):
        if row["approach"] == "context_packing":
            packing[row["conversation_length"]].append(row["hit"])

    header = f"{'':<26}" + "".join(f"{n:>8}" for n in LENGTHS)
    print("\nRecall by conversation length\n")
    print(header)
    print("-" * len(header))
    print(f"{'context packing':<26}" +
          "".join(f"{sum(packing[n]) / len(packing[n]):>8.3f}" for n in LENGTHS))
    print()
    for keep in EXTRACTION_RECALL:
        label = f"memory, extractor {int(keep * 100)}%"
        print(f"{label:<26}" +
              "".join(f"{memory_recall_with_lossy_extractor(n, keep):>8.3f}" for n in LENGTHS))

    print("\nMemory retrieval's recall tracks extraction quality and stays flat across")
    print("length, because a fact the extractor dropped is absent rather than hard to")
    print("find. Context packing has no extraction step, so the two are not penalised")
    print("symmetrically: the crossover moves later as the extractor gets worse.")


if __name__ == "__main__":
    main()
