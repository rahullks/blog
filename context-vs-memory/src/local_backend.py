"""
Local open-source embeddings backend, as a drop-in replacement for the TF-IDF
stand-in in sim.py.

Uses fastembed (https://github.com/qdrant/fastembed), which runs ONNX models on
CPU. The default model (BAAI/bge-small-en-v1.5, ~130MB) downloads on first use
and is cached in ~/.cache/fastembed, so later runs need no network.

    pip install fastembed

Why this exists: the TF-IDF scorer in sim.py is a *lexical* matcher: it finds
a fact because the fact and the query happen to share literal words. That is
fine for testing the retrieval/scaling logic, but it is not what a real system
does. This backend swaps in real semantic embeddings so the comparison holds up
when a query is worded nothing like the fact it is asking about.

    from local_backend import LocalEmbeddingScorer
    import sim

    scorer = LocalEmbeddingScorer()
    text, tokens, scored = sim.approach_context_packing(
        conversation, query, token_budget=120,
        max_scorable_history=300, scorer=scorer.rank_by_relevance)
"""
import numpy as np

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class LocalEmbeddingScorer:
    """Scores (query, candidate turns) pairs using local ONNX embeddings.

    Exposes rank_by_relevance(query, candidates) with the exact same contract
    as sim.rank_by_relevance, so it can be passed directly as the `scorer`
    argument to sim.approach_context_packing / sim.approach_memory_retrieval.

    Tracks `texts_embedded` (a real cost counter, not a proxy) so the
    context-packing vs. memory-retrieval cost gap can be measured in actual
    embedding calls rather than inferred from turns_scored.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model=None):
        if model is not None:
            self.model = model  # allows injecting a fake model for testing
        else:
            # imported lazily so the rest of the repo doesn't need fastembed installed
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name=model_name)
        self.model_name = model_name
        self._cache = {}
        self.texts_embedded = 0
        self.embed_batches = 0

    def _embed_batch(self, texts: list) -> None:
        """Embed every text not already cached, in a single batched call."""
        missing = []
        seen = set()
        for t in texts:
            if t not in self._cache and t not in seen:
                seen.add(t)
                missing.append(t)
        if not missing:
            return
        vectors = list(self.model.embed(missing))
        if len(vectors) != len(missing):
            raise ValueError(
                f"embedding backend returned {len(vectors)} vectors for {len(missing)} texts")
        for text, vec in zip(missing, vectors):
            v = np.asarray(vec, dtype=np.float32)
            self._cache[text] = v
        self.texts_embedded += len(missing)
        self.embed_batches += 1

    def rank_by_relevance(self, query: str, candidates: list):
        """candidates: list[Turn]. Returns [(Turn, score)] sorted by descending
        cosine similarity to the query, using real embeddings."""
        self._embed_batch([query] + [t.text for t in candidates])
        q_vec = self._cache[query]
        q_norm = float(np.linalg.norm(q_vec))
        scored = []
        for turn in candidates:
            v = self._cache[turn.text]
            sim = float(np.dot(q_vec, v) / (q_norm * float(np.linalg.norm(v)) + 1e-8))
            scored.append((turn, sim))
        return sorted(scored, key=lambda x: -x[1])
