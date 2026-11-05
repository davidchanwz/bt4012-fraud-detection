"""Dashboard checks (app/). Headless, via Streamlit's AppTest; no browser or port needed.

- every tab renders under the main sidebar selections, with the real tables and with none
- the whole app renders and survives sidebar changes (catches duplicate widget keys across tabs)
- the app is read-only: no training / scoring / metric code is imported
"""
import ast
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import tests.app_check as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAIN = str(ROOT / "app" / "main.py")


@pytest.mark.parametrize("tab", C.TABS)
def test_tab_renders_with_real_tables(tab):
    assert C.check(tab, quick=True)


@pytest.mark.parametrize("tab", C.TABS)
def test_tab_renders_without_any_tables(tab, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_TABLE_DIR", str(tmp_path))
    monkeypatch.setenv("DASHBOARD_CLEAN_FILE", str(tmp_path / "none.parquet"))
    import importlib

    import app.data
    importlib.reload(app.data)
    try:
        assert C.check(tab, quick=True)
    finally:
        monkeypatch.undo()
        importlib.reload(app.data)


def test_whole_app_survives_sidebar_changes():
    at = AppTest.from_file(MAIN, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs][:2] == ["Review budget", "Scoreboard"]
    for label, value in [("Model", "rf"), ("Fraud subset", "normal_looking_strict"),
                         ("Model", "lr"), ("Fraud subset", "normal_looking")]:
        at.selectbox(key={"Model": "model", "Fraud subset": "subset"}[label]).select(value).run()
        assert not at.exception, (label, value, at.exception)


def test_app_never_imports_training_or_metric_code():
    banned = ("src.models.train", "src.models.run_main", "src.eval", "sklearn", "lightgbm")
    for path in (ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for n in names:
                assert not n.startswith(banned), f"{path.name} imports {n}"
