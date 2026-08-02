"""
Unit tests for local_backend.py using a fake embedding model, so no model
download or network access is needed and these run anywhere in under a second.

These cover the plumbing: batching, caching, cosine similarity math, and sort
order. For a check against the real model, run eval_lexical_vs_semantic.py,
which downloads it and scores the actual conversations.
"""

import numpy as np

from local_backend import LocalEmbeddingScorer


class FakeEmbeddingModel:
    """Returns a deterministic pseudo-embedding derived from the input text,
    so similarity ordering is stable and assertable without a real model."""

    def __init__(self, dims=32):
        self.dims = dims
        self.batches = []

    def embed(self, texts):
        texts = list(texts)
        self.batches.append(texts)
        for text in texts:
            rng = np.random.RandomState(abs(hash(text)) % (2**32))
            vec = rng.normal(size=self.dims)
            yield vec / np.linalg.norm(vec)


class FakeTurn:
    def __init__(self, text):
        self.text = text


def test_rank_by_relevance_returns_sorted_scores():
    model = FakeEmbeddingModel()
    scorer = LocalEmbeddingScorer(model=model)

    candidates = [FakeTurn("my employee ID is E-77210"),
                  FakeTurn("what's the weather like today"),
                  FakeTurn("the budget code is BC-4471")]
    ranked = scorer.rank_by_relevance("what is my employee id", candidates)

    assert len(ranked) == 3
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True), "results must be sorted descending by score"
    for _, score in ranked:
        assert -1.0001 <= score <= 1.0001, "cosine similarity must be in [-1, 1]"


def test_same_text_scores_perfect_similarity():
    scorer = LocalEmbeddingScorer(model=FakeEmbeddingModel())
    ranked = scorer.rank_by_relevance("exact match text", [FakeTurn("exact match text")])
    _, score = ranked[0]
    assert abs(score - 1.0) < 1e-4, "identical text must score ~1.0"


def test_embeds_in_one_batch_and_deduplicates():
    model = FakeEmbeddingModel()
    scorer = LocalEmbeddingScorer(model=model)

    candidates = [FakeTurn("same text"), FakeTurn("same text"), FakeTurn("different text")]
    scorer.rank_by_relevance("same text", candidates)

    assert len(model.batches) == 1, "one ranking pass must issue exactly one batched embed call"
    assert sorted(model.batches[0]) == ["different text", "same text"], \
        "duplicate texts (and the query echoing a candidate) must be embedded once"
    assert scorer.texts_embedded == 2


def test_cache_persists_across_calls():
    model = FakeEmbeddingModel()
    scorer = LocalEmbeddingScorer(model=model)

    candidates = [FakeTurn("alpha"), FakeTurn("beta")]
    scorer.rank_by_relevance("query one", candidates)
    scorer.rank_by_relevance("query one", candidates)

    assert len(model.batches) == 1, "second identical pass must be served entirely from cache"
    assert scorer.texts_embedded == 3, "alpha, beta, and the query: embedded once each"


def test_raises_when_backend_returns_wrong_vector_count():
    class ShortModel(FakeEmbeddingModel):
        def embed(self, texts):
            texts = list(texts)
            yield np.ones(self.dims) / np.sqrt(self.dims)  # one vector, however many asked for

    scorer = LocalEmbeddingScorer(model=ShortModel())
    try:
        scorer.rank_by_relevance("q", [FakeTurn("a"), FakeTurn("b")])
    except ValueError as e:
        assert "vectors for" in str(e)
    else:
        raise AssertionError("expected ValueError on vector/text count mismatch")


if __name__ == "__main__":
    test_rank_by_relevance_returns_sorted_scores()
    test_same_text_scores_perfect_similarity()
    test_embeds_in_one_batch_and_deduplicates()
    test_cache_persists_across_calls()
    test_raises_when_backend_returns_wrong_vector_count()
    print("all local_backend tests passed")
