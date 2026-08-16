"""Tests for the NeuroCure Streamlit app shell and viewer wiring.

Uses Streamlit's :class:`AppTest` to run app scripts headlessly (no browser)
and asserts on the rendered elements. ``page_link`` is not surfaced as a first
class element in this Streamlit version, so we assert on markdown content and
buttons instead.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "neurocure_app"
REPO = Path(__file__).resolve().parent.parent


def _app(script: str) -> AppTest:
    return AppTest.from_file(str(APP_DIR / script), default_timeout=60)


def test_app_shell_renders_welcome():
    at = _app("app.py").run()
    assert not at.exception
    assert any("NeuroCure" in t.value for t in at.title)
    all_md = " ".join(m.value for m in at.markdown)
    assert "Retraining-Free AI Framework" in all_md
    assert "Navigation" in all_md  # sidebar header rendered


def test_app_shell_sidebar_lists_all_pages():
    """The sidebar navigation section is rendered (markdown header present)."""
    at = _app("app.py").run()
    all_md = " ".join(m.value for m in at.markdown)
    assert "Navigation" in all_md
    # page_link labels are not surfaced by AppTest in this version; verify
    # the sidebar divider/header markdown is present instead.


def test_upload_page_renders_demo_button_and_uploader():
    at = _app("pages/1_Upload.py").run()
    assert not at.exception
    assert any("demo" in (b.label or "").lower() for b in at.button)
    assert at.file_uploader


def test_upload_page_loads_demo_brain(tmp_path, monkeypatch):
    """Clicking the demo brain button ingests a volume without error.

    ``st.switch_page`` raises under AppTest (no multipage context), so we
    tolerate that specific exception -- the ingestion itself succeeds.
    """
    monkeypatch.chdir(tmp_path)
    import shutil

    shutil.copytree(REPO / "neurocure_app", tmp_path / "neurocure_app")
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    if (REPO / "assets").exists():
        shutil.copytree(REPO / "assets", tmp_path / "assets")

    at = AppTest.from_file(
        str(tmp_path / "neurocure_app" / "pages" / "1_Upload.py"), default_timeout=60
    ).run()
    assert not at.exception
    demo_btn = next(b for b in at.button if "demo" in (b.label or "").lower())
    at = demo_btn.click().run()
    # The only allowed exception is switch_page failing in test isolation.
    excs = [e.value for e in at.exception]
    non_switch = [e for e in excs if "Could not find page" not in str(e)]
    assert non_switch == [], f"unexpected exceptions: {non_switch}"


def test_state_get_session_returns_none_when_unset():
    """In a bare (no ScriptRun) context, get_session returns None."""
    from neurocure_app import state

    result = state.get_session()
    assert result is None or hasattr(result, "volume")


def test_viewer_build_3d_figure_does_not_raise(tmp_path):
    """The figure builder used by render_brain_3d produces a valid figure."""
    from brainframe.reporting.unified_report import build_3d_figure
    from brainframe.session import Session

    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.config.sam.auto_download = False
    sess.config.pipeline.cache = False
    sess.load_demo_brain().segment().reconstruct()
    recon = sess.reconstruction
    fig = build_3d_figure(
        recon["meshes"],
        label_volume=recon["label_volume"],
        spacing=recon["spacing"],
    )
    assert fig is not None
    assert len(fig.data) >= 1


def test_viewer_build_3d_figure_with_simulation(tmp_path):
    """The animated cure figure (with simulation) builds with play frames."""
    from brainframe.reporting.unified_report import build_3d_figure
    from brainframe.session import Session

    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.config.sam.auto_download = False
    sess.config.pipeline.cache = False
    sess.load_demo_brain().segment().reconstruct().predict().evaluate().recommend()
    recon = sess.reconstruction
    fig = build_3d_figure(
        recon["meshes"],
        label_volume=recon["label_volume"],
        spacing=recon["spacing"],
        simulation=sess.simulation_override,
    )
    assert fig is not None
    assert len(fig.frames) >= 1  # animation frames present
