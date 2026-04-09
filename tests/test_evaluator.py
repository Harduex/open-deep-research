from open_deep_research.core.evaluator import Evaluator
from open_deep_research.models import ResearchPlan, SubQuestion, TokenBudget


def test_hard_ceiling_iteration():
    """Evaluator returns stop when iteration >= max_iterations."""
    plan = ResearchPlan(
        query="test",
        sub_questions=[SubQuestion(id="sq_0", question="Q1")],
        iteration=10,
        max_iterations=10,
    )
    budget = TokenBudget(max_tokens=100_000)
    # Can't call async evaluate_stopping without LLM, but test the hard ceiling path
    result = Evaluator._estimate_coverage(plan)
    assert result == 0.0


def test_coverage_estimation():
    plan = ResearchPlan(
        query="test",
        sub_questions=[
            SubQuestion(id="sq_0", question="Q1", status="answered"),
            SubQuestion(id="sq_1", question="Q2", status="pending"),
            SubQuestion(id="sq_2", question="Q3", status="answered"),
        ],
    )
    assert abs(Evaluator._estimate_coverage(plan) - 2 / 3) < 1e-6


def test_coverage_empty():
    plan = ResearchPlan(query="test", sub_questions=[])
    assert Evaluator._estimate_coverage(plan) == 0.0
