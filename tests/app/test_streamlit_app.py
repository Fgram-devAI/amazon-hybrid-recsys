"""Smoke import test for the Streamlit entrypoint.

Streamlit runs the target script with ``__name__ == "__main__"``, so plain import
must not call ``render()`` and must not raise.
"""
from __future__ import annotations


def test_streamlit_app_imports_without_rendering() -> None:
    import app.streamlit_app as module

    assert hasattr(module, "render")
    assert callable(module.render)
