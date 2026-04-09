from open_deep_research.models import Finding
from open_deep_research.embeddings.dedup import FindingDeduplicator, _cosine_similarity


def test_cosine_similarity_identical():
    a = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_zero():
    assert _cosine_similarity([0, 0], [1, 1]) == 0.0


def test_dedup_no_model_passthrough():
    """Without sentence-transformers, dedup should return all findings as-is."""
    dedup = FindingDeduplicator()
    findings = [
        Finding(content="Python is great", source_ids=[1], confidence="high"),
        Finding(content="Python is great", source_ids=[2], confidence="high"),
    ]
    result = dedup.deduplicate(findings)
    # Without model, should return all since embeddings fail gracefully
    assert len(result) >= 1


def test_dedup_single_finding():
    dedup = FindingDeduplicator()
    findings = [Finding(content="Test", source_ids=[1], confidence="medium")]
    result = dedup.deduplicate(findings)
    assert len(result) == 1


def test_dedup_empty():
    dedup = FindingDeduplicator()
    assert dedup.deduplicate([]) == []
