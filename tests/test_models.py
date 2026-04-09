from open_deep_research.models import (
    Finding,
    IterationMetrics,
    Report,
    ReportMetadata,
    ReportSection,
    ResearchPlan,
    SearchResult,
    SessionState,
    Source,
    StoppingEvaluation,
    SubQuestion,
    TokenBudget,
)


def test_token_budget_basics():
    b = TokenBudget(max_tokens=1000)
    assert b.remaining == 1000
    assert not b.is_exceeded
    assert not b.should_warn

    b.add(800)
    assert b.remaining == 200
    assert b.should_warn
    assert not b.is_exceeded

    b.add(200)
    assert b.remaining == 0
    assert b.is_exceeded


def test_token_budget_zero():
    b = TokenBudget(max_tokens=0)
    assert b.remaining == 0
    assert b.is_exceeded
    assert not b.should_warn


def test_search_result():
    r = SearchResult(url="https://example.com", title="Test", snippet="snip")
    assert r.url == "https://example.com"


def test_source_has_default_timestamp():
    s = Source(id=1, url="https://example.com", title="Test", snippet="snip")
    assert s.retrieved_at is not None


def test_finding_model():
    f = Finding(content="test", source_ids=[1, 2], confidence="high")
    assert f.confidence == "high"
    assert len(f.source_ids) == 2


def test_sub_question_defaults():
    sq = SubQuestion(id="sq_0", question="What?")
    assert sq.status == "pending"
    assert sq.findings == []


def test_research_plan():
    plan = ResearchPlan(
        query="test",
        sub_questions=[SubQuestion(id="sq_0", question="Q1")],
    )
    assert plan.iteration == 0
    assert plan.max_iterations == 10


def test_stopping_evaluation_defaults():
    ev = StoppingEvaluation(should_stop=False, reasoning="test")
    assert ev.coverage_score == 0.0
    assert not ev.saturation_detected
    assert ev.unanswered_questions == []


def test_session_state_defaults():
    plan = ResearchPlan(query="test", sub_questions=[])
    state = SessionState(session_id="abc123", plan=plan)
    assert state.status == "planning"
    assert state.sources == []
    assert state.findings == []
    assert state.report is None
    assert state.iteration_metrics == []


def test_iteration_metrics_model():
    m = IterationMetrics(iteration=0, new_findings_count=5, new_sources_count=3)
    assert m.dedup_removed_count == 0
    data = m.model_dump()
    assert data["iteration"] == 0
    assert data["new_findings_count"] == 5


def test_session_state_with_metrics_serializes():
    plan = ResearchPlan(query="test", sub_questions=[])
    state = SessionState(
        session_id="abc123",
        plan=plan,
        iteration_metrics=[
            IterationMetrics(iteration=0, new_findings_count=5, new_sources_count=3, dedup_removed_count=1),
        ],
    )
    json_str = state.model_dump_json()
    restored = SessionState.model_validate_json(json_str)
    assert len(restored.iteration_metrics) == 1
    assert restored.iteration_metrics[0].new_findings_count == 5


def test_report_model():
    report = Report(
        title="Test",
        executive_summary="Summary",
        sections=[ReportSection(title="S1", content="Content")],
    )
    assert len(report.sections) == 1
    assert report.contradictions == []
