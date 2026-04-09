import json
import tempfile
from pathlib import Path

from open_deep_research.models import ResearchPlan, SessionState, SubQuestion, TokenBudget
from open_deep_research.state.checkpoint import load_checkpoint, save_checkpoint
from open_deep_research.state.session import SessionManager


def _make_state(session_id: str = "test123") -> SessionState:
    plan = ResearchPlan(
        query="test query",
        sub_questions=[SubQuestion(id="sq_0", question="Q1")],
    )
    return SessionState(session_id=session_id, plan=plan)


def test_save_and_load_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        state = _make_state()
        path = save_checkpoint(state, Path(tmpdir))
        assert path.exists()

        loaded = load_checkpoint("test123", Path(tmpdir))
        assert loaded is not None
        assert loaded.session_id == "test123"
        assert loaded.plan.query == "test query"


def test_load_missing_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        loaded = load_checkpoint("nonexistent", Path(tmpdir))
        assert loaded is None


def test_session_manager_create():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SessionManager(tmpdir)
        plan = ResearchPlan(query="test", sub_questions=[])
        state = mgr.create_session(plan, 100_000)
        assert len(state.session_id) == 12
        assert state.budget.max_tokens == 100_000


def test_session_manager_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SessionManager(tmpdir)
        assert mgr.list_sessions() == []

        plan = ResearchPlan(query="test list", sub_questions=[])
        mgr.create_session(plan, 100_000)
        sessions = mgr.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["query"] == "test list"


def test_budget_check():
    from open_deep_research.state.budget import check_budget

    b = TokenBudget(max_tokens=1000, used_tokens=0)
    assert check_budget(b)

    b.used_tokens = 1000
    assert not check_budget(b)
