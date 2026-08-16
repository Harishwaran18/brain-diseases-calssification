"""Session-state manager for the NeuroCure platform.

Wraps the engine :class:`~brainframe.session.Session` in a Streamlit
``session_state`` singleton so that state persists across page navigations and
re-runs. Each long-running step is wrapped with a status container so the UI
shows progress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from brainframe.session import Session

# Repo root resolved from this file's location so the app works regardless of
# the current working directory (important for cloud deployments where the
# process may start from a different cwd).
REPO_ROOT = Path(__file__).resolve().parent.parent

SESSION_KEY = "neurocure_session"
WORK_DIR_KEY = "neurocure_work_dir"


def get_work_dir() -> Path:
    """Return (and lazily create) the per-user work directory."""
    if WORK_DIR_KEY not in st.session_state:
        work = REPO_ROOT / "data" / "outputs" / "neurocure"
        work.mkdir(parents=True, exist_ok=True)
        st.session_state[WORK_DIR_KEY] = work
    return st.session_state[WORK_DIR_KEY]


def get_session() -> Session | None:
    """Return the current engine Session, or ``None`` if not yet created."""
    return st.session_state.get(SESSION_KEY)


def init_session(**kwargs: Any) -> Session:
    """Create a fresh engine Session and store it in session_state."""
    work = get_work_dir()
    kwargs.setdefault("output_dir", work / f"subject_{len(list(work.iterdir())) + 1}")
    config_path = REPO_ROOT / "configs" / "default.yaml"
    sess = Session.from_config_path(str(config_path), **kwargs)
    st.session_state[SESSION_KEY] = sess
    return sess


def require_session() -> Session:
    """Return the current session or show a warning and stop execution."""
    sess = get_session()
    if sess is None:
        st.warning("No scan loaded yet. Go to the **Upload** page to load a brain MRI.")
        st.page_link("pages/1_Upload.py", label="→ Go to Upload", icon="📤")
        st.stop()
    return sess


def run_step(label: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a long step with a spinner + status, returning its result."""
    with st.status(label, expanded=True) as status:
        try:
            result = fn(*args, **kwargs)
            status.update(label=f"✅ {label}", state="complete")
            return result
        except Exception as e:  # noqa: BLE001
            status.update(label=f"❌ {label} failed", state="error")
            st.exception(e)
            raise


def session_summary() -> dict:
    """Return the current session's summary dict (or empty)."""
    sess = get_session()
    return sess.summary() if sess else {}
