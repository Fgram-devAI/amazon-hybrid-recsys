"""Smoke import test for the Streamlit entrypoint.

Streamlit runs the target script with ``__name__ == "__main__"``, so plain import
must not call ``render()`` and must not raise.
"""
from __future__ import annotations

import runpy
from pathlib import Path


def test_streamlit_app_imports_without_rendering() -> None:
    import app.streamlit_app as module

    assert hasattr(module, "render")
    assert callable(module.render)


def test_streamlit_app_imports_when_executed_by_path() -> None:
    path = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    namespace = runpy.run_path(str(path), run_name="streamlit_app_smoke")

    assert callable(namespace["render"])


def test_streamlit_app_registers_recommendations_tab_with_method_labels() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    ).read_text(encoding="utf-8")
    assert '"Recommendations"' in source
    for label in (
        "Popularity",
        "SVD",
        "Item-KNN cosine",
        "Item-KNN pearson",
        "Item-KNN msd",
        "LightGCN checkpoint",
        "LLM hybrid — profile mode",
        "LLM hybrid — query mode",
    ):
        assert label in source, f"missing method label: {label!r}"
    # Caching wiring per §8: cache fitted models (resource), not data.
    assert "st.cache_resource" in source
    assert "st.cache_data" in source
    # Live LLM must be opt-in; default dry-run.
    assert "Call Groq" in source
