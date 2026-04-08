from __future__ import annotations

import uuid
from pathlib import Path

from open_deep_research.models import ResearchPlan, SessionState, TokenBudget
from open_deep_research.state.checkpoint import load_checkpoint, save_checkpoint


class SessionManager:
    def __init__(self, storage_dir: str = "~/.open-deep-research/sessions") -> None:
        self._storage_dir = Path(storage_dir).expanduser()

    def create_session(self, plan: ResearchPlan, budget_tokens: int = 500_000) -> SessionState:
        state = SessionState(
            session_id=uuid.uuid4().hex[:12],
            plan=plan,
            budget=TokenBudget(max_tokens=budget_tokens),
        )
        self.save(state)
        return state

    def save(self, state: SessionState) -> Path:
        return save_checkpoint(state, self._storage_dir)

    def load(self, session_id: str) -> SessionState | None:
        return load_checkpoint(session_id, self._storage_dir)

    def list_sessions(self) -> list[dict]:
        if not self._storage_dir.exists():
            return []

        sessions = []
        for d in self._storage_dir.iterdir():
            if not d.is_dir():
                continue
            state = load_checkpoint(d.name, self._storage_dir)
            if state:
                sessions.append({
                    "session_id": state.session_id,
                    "query": state.plan.query,
                    "status": state.status,
                    "created_at": state.created_at.isoformat(),
                    "sources": len(state.sources),
                    "findings": len(state.findings),
                })
        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)
