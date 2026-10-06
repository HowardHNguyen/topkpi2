"""Smoke test: every page of the Streamlit app must render without an exception."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_all_pages_render():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"Page '{page}' raised: {at.exception}"
