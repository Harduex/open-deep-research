from __future__ import annotations

import math

from open_deep_research.models import Finding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FindingDeduplicator:
    """Embedding-based finding deduplication using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.85) -> None:
        self._threshold = similarity_threshold
        self._model = None
        self._model_name = model_name

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            self._model = None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        if self._model is None:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def deduplicate(self, findings: list[Finding]) -> list[Finding]:
        """Remove duplicate findings based on semantic similarity.

        When two findings are similar above threshold, keep the longer one
        and merge source_ids.
        """
        if len(findings) <= 1:
            return findings

        texts = [f.content for f in findings]
        embeddings = self._embed(texts)

        # If embeddings unavailable, fall back to returning all findings
        if not embeddings:
            return findings

        merged_indices: set[int] = set()
        result: list[Finding] = []

        for i, finding in enumerate(findings):
            if i in merged_indices:
                continue

            merged_sources = list(finding.source_ids)
            best_content = finding.content
            best_confidence = finding.confidence

            for j in range(i + 1, len(findings)):
                if j in merged_indices:
                    continue

                sim = _cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self._threshold:
                    merged_indices.add(j)
                    # Keep longer content
                    if len(findings[j].content) > len(best_content):
                        best_content = findings[j].content
                    # Keep higher confidence
                    conf_order = {"high": 3, "medium": 2, "low": 1}
                    if conf_order.get(findings[j].confidence, 0) > conf_order.get(best_confidence, 0):
                        best_confidence = findings[j].confidence
                    # Merge sources
                    merged_sources.extend(findings[j].source_ids)

            result.append(Finding(
                content=best_content,
                source_ids=list(set(merged_sources)),
                confidence=best_confidence,
            ))

        return result
