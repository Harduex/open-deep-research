from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from open_deep_research.models import SessionState


def save_checkpoint(state: SessionState, storage_dir: Path) -> Path:
    storage_dir = Path(storage_dir).expanduser()
    session_dir = storage_dir / state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    state.updated_at = datetime.now(timezone.utc)
    path = session_dir / "state.json"
    path.write_text(state.model_dump_json(indent=2))
    return path


def load_checkpoint(session_id: str, storage_dir: Path) -> SessionState | None:
    path = Path(storage_dir).expanduser() / session_id / "state.json"
    if not path.exists():
        return None
    return SessionState.model_validate_json(path.read_text())
