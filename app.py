# app.py — TopKPI2 multipage app that runs .py pages if present, or renders .ipynb as HTML
import os, io, re, pathlib, importlib.util
from typing import Optional
import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="TopKPI2 – AI Growth & Retention Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Keep sidebar & chevron; hide only top-right & bottom-right chrome ----------
st.markdown("""
<style>
header { visibility: visible !important; }
div[data-testid="stDecoration"] { display: block !important; visibility: visible !important; }
div[data-testid="stSidebar"] { visibility: visible !important; display: block !important; }
div[data-testid="stToolbarActions"] { display: none !important; }           /* hide Fork/GitHub/⋮ */
div[data-testid="stToolbar"] { display: block !important; visibility: visible !important; }
.stAppBottomRightButtons, .stAppDeployButton { display: none !important; }   /* hide watermark */
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Imports for notebook rendering ----------
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from streamlit.components.v1 import html as st_html

# ---------- Mapping: Nav label -> file stem (without extension) ----------
PAGES = [
    ("KPIs Analysis",              "KPIs_Analysis"),
    ("Why People Churn",           "Why People Churn"),
    ("Why People Convert",         "Why People Convert"),
    ("Why People Engage",          "Why People Engage"),
    ("Time Series Analysis",       "TimeSeriesAnalysis"),
    ("Sentiment Analysis",         "SentimentAnalysis"),
    ("Predictive Analytics",       "Predictive_Analytics"),
    ("Product Recommendations",    "Product_Recommendations"),
    ("Customer Segmentation",      "Customer_Segmentation"),
]

# ---------- Sidebar Nav ----------
with st.sidebar:
    st.title("TopKPI2 📈")
    page_label = st.selectbox("Select a notebook:", [p[0] for p in PAGES], index=0)
    st.caption("Tip: if a same-named .py exists, it runs as an interactive Streamlit page; "
               "otherwise the .ipynb is executed and embedded below.")
    st.markdown("---")
    with st.expander("About this app", expanded=False):
        st.write(
            "A root-only Streamlit app that runs your analytics notebooks as pages, "
            "with zero subdirectories. Add a matching `.py` to upgrade any notebook "
            "to a fully interactive Streamlit page without changing navigation."
        )

# ---------- Helpers ----------
def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")

def find_source(stem: str) -> tuple[Optional[str], Optional[str]]:
    """Return (py_path, ipynb_path) if they exist in root."""
    py = f"{stem}.py"
    nb = f"{stem}.ipynb"
    return (py if os.path.exists(py) else None,
            nb if os.path.exists(nb) else None)

def run_python_module(py_path: str):
    """Import and run a .py page in-place (as if it were pages/Page.py)."""
    try:
        spec = importlib.util.spec_from_file_location("page_module", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # noqa: F401
    except Exception as e:
        st.error(f"Failed to run `{py_path}`:\n\n{e}")

@st.cache_data(show_spinner=True)
def render_notebook_to_html(nb_path: str, execute: bool = True) -> str:
    """Execute notebook and return HTML (cached)."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    if execute:
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3", allow_errors=True)
        ep.preprocess(nb, {'metadata': {'path': "."}})
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = False      # show code cells
    html_exporter.exclude_output_prompt = True
    body, _ = html_exporter.from_notebook_node(nb)
    return body

def show_page(stem: str, pretty_title: str):
    st.title(pretty_title)
    py_path, nb_path = find_source(stem)
    if py_path:
        st.info(f"Rendering **{py_path}** (interactive Streamlit).")
        run_python_module(py_path)
    elif nb_path:
        st.info(f"Rendering **{nb_path}** (executed then embedded).")
        with st.spinner("Running notebook…"):
            html = render_notebook_to_html(nb_path, execute=True)
        st_html(html, height=1200, scrolling=True)
    else:
        st.warning(f"Missing page files for **{pretty_title}**.\n"
                   f"Expected one of: `{stem}.py` or `{stem}.ipynb` in repo root.")

# ---------- Route & render ----------
stem = dict(PAGES)[page_label]
stem = slugify(stem) if not os.path.exists(f"{stem}.ipynb") and not os.path.exists(f"{stem}.py") else stem
show_page(stem, page_label)

# ---------- Optional footer ----------
st.markdown(
    "<hr style='border:1px solid #eee;'>"
    "<div style='text-align:center;color:gray'>© 2025 MaxAIS — TopKPI2</div>",
    unsafe_allow_html=True,
)
