import math
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Page & basic theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TopKPI2 – Advanced Marketing Growth & Retention Intelligence",
    page_icon="📈",
    layout="wide",
)

# Hide Streamlit chrome (keep the sidebar)
HIDE_STREAMLIT_STYLE = """
<style>
#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}
footer {visibility: hidden;}
button[kind="header"] {display: none;}
a[data-testid="stBaseLink"] {display:none;}
</style>
"""
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Constants & schema
# -----------------------------------------------------------------------------
REQUIRED_COLS: List[str] = [
    "Coverage",
    "Customer",
    "Customer Lifetime Value",
    "Education",
    "Effective To Date",
    "EmploymentStatus",
    "Gender",
    "Income",
    "Location Code",
    "Marital Status",
    "Monthly Premium Auto",
    "Months Since Last Claim",
    "Months Since Policy Inception",
    "Number of Open Complaints",
    "Number of Policies",
    "Policy",
    "Policy Type",
    "Renew Offer Type",
    "Response",
    "Sales Channel",
    "State",
    "Total Claim Amount",
    "Vehicle Class",
    "Vehicle Size",
    # New required fields:
    "Churn",
    "EngagementScore",
]

# Default cost assumptions by channel – override via sidebar later if desired
DEFAULT_CHANNEL_COST_MAP: Dict[str, float] = {
    "Web": 40.0,
    "Call Center": 70.0,
    "Branch": 90.0,
    "Agent": 120.0,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def yes_no_flag(series: pd.Series) -> pd.Series:
    """Convert Yes/No style strings to 1/0, robust to casing/whitespace."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(0)
        .astype(int)
    )


def compute_global_kpis(df: pd.DataFrame, channel_cost_map: Dict[str, float]) -> Dict[str, float]:
    """
    Compute high-level KPIs:
    - conversion rate
    - churn rate
    - CPA
    - CLV realized & average CLV
    - ROI
    """
    d = df.copy()

    # Flags
    d["is_convert"] = yes_no_flag(d["Response"])
    d["is_churn"] = yes_no_flag(d["Churn"])

    customers = d["Customer"].nunique()
    conversions = int(d["is_convert"].sum())
    churns = int(d["is_churn"].sum())

    conversion_rate = conversions / len(d) if len(d) else 0.0
    churn_rate = churns / len(d) if len(d) else 0.0

    # Cost per row based on Sales Channel
    d["acq_cost"] = d["Sales Channel"].map(channel_cost_map).fillna(0.0)
    total_spend = float(d["acq_cost"].sum())

    clv_col = "Customer Lifetime Value"
    clv_realized = float(d.loc[d["is_convert"] == 1, clv_col].sum())
    if conversions > 0:
        avg_clv = float(d.loc[d["is_convert"] == 1, clv_col].mean())
    else:
        avg_clv = float(d[clv_col].mean())

    cpa = total_spend / conversions if conversions > 0 else float("nan")
    revenue = clv_realized
    roi = (revenue - total_spend) / total_spend * 100.0 if total_spend > 0 else float("nan")

    return {
        "customers": customers,
        "conversions": conversions,
        "churns": churns,
        "conversion_rate": conversion_rate,
        "churn_rate": churn_rate,
        "total_spend": total_spend,
        "cpa": cpa,
        "clv_realized": clv_realized,
        "avg_clv": avg_clv,
        "roi": roi,
    }


def kpis_by_segment(
    df: pd.DataFrame,
    channel_cost_map: Dict[str, float],
    segment_col: str,
) -> pd.DataFrame:
    """
    Aggregate KPIs by a segment column (e.g., Sales Channel, Renew Offer Type).
    Returns a DataFrame with customers, conv_rate, churn_rate, CPA, CLV, ROI.
    """
    d = df.copy()
    d["is_convert"] = yes_no_flag(d["Response"])
    d["is_churn"] = yes_no_flag(d["Churn"])
    d["acq_cost"] = d["Sales Channel"].map(channel_cost_map).fillna(0.0)

    rows = []
    for seg_value, g in d.groupby(segment_col):
        customers = g["Customer"].nunique()
        conversions = int(g["is_convert"].sum())
        churns = int(g["is_churn"].sum())
        conv_rate = conversions / len(g) if len(g) else 0.0
        churn_rate = churns / len(g) if len(g) else 0.0
        spend = float(g["acq_cost"].sum())
        cpa = spend / conversions if conversions > 0 else float("nan")
        clv_real = float(
            g.loc[g["is_convert"] == 1, "Customer Lifetime Value"].sum()
        )
        roi = (clv_real - spend) / spend * 100.0 if spend > 0 else float("nan")

        rows.append(
            {
                segment_col: seg_value,
                "customers": customers,
                "conversion_rate": conv_rate,
                "churn_rate": churn_rate,
                "cpa": cpa,
                "clv_realized": clv_real,
                "roi": roi,
            }
        )

    return pd.DataFrame(rows).sort_values("conversion_rate", ascending=False)


@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data
def load_sample_csv(path: str = "data.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


# -----------------------------------------------------------------------------
# Sidebar – upload + navigation + schema
# -----------------------------------------------------------------------------
st.sidebar.title("TopKPI2 📈")
st.sidebar.caption("Advanced Marketing Growth & Retention Intelligence")

uploaded = st.sidebar.file_uploader(
    "Upload CSV (same schema as your notebooks)",
    type=["csv"],
)

# Channel cost overrides (optional)
with st.sidebar.expander("Cost per Acquisition (override)", expanded=False):
    web_cost = st.number_input("Web", value=DEFAULT_CHANNEL_COST_MAP["Web"], step=5.0)
    cc_cost = st.number_input("Call Center", value=DEFAULT_CHANNEL_COST_MAP["Call Center"], step=5.0)
    branch_cost = st.number_input("Branch", value=DEFAULT_CHANNEL_COST_MAP["Branch"], step=5.0)
    agent_cost = st.number_input("Agent", value=DEFAULT_CHANNEL_COST_MAP["Agent"], step=5.0)

CHANNEL_COST_MAP = {
    "Web": float(web_cost),
    "Call Center": float(cc_cost),
    "Branch": float(branch_cost),
    "Agent": float(agent_cost),
}

page = st.sidebar.radio(
    "Navigate",
    [
        "KPIs overview",
        "Why People Churn",
        "Why People Convert",
        "Why People Engage",
        "Time Series Analysis",
        "Sentiment Analysis",
        "Predictive Analytics",
        "Product Recommendations",
        "Customer Segmentation",
    ],
)

# Placeholder for df – load from upload or sample_data
df = None
source_label = ""

try:
    if uploaded is not None:
        df = load_csv(uploaded)
        source_label = f"Uploaded {uploaded.name}"
    else:
        df = load_sample_csv("data.csv")
        source_label = "Loaded sample_data.csv from repo root"
except Exception as e:
    st.error(f"Failed to load data: {e}")
    df = None

# Schema checklist
with st.sidebar.expander("Schema checklist", expanded=False):
    if df is None:
        st.info("Upload a CSV to run schema checks.")
    else:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        extra = [c for c in df.columns if c not in REQUIRED_COLS]

        if not missing:
            st.success("All required columns are present. ✅")
        else:
            st.error("Missing required columns: " + ", ".join(missing))

        if extra:
            st.caption("Extra columns (not used by core KPIs): " + ", ".join(extra))

# -----------------------------------------------------------------------------
# Main layout – title + data preview
# -----------------------------------------------------------------------------
st.markdown("## Advanced Marketing Growth & Retention Intelligence (TopKPI2)")
st.caption(
    "Conversion • Churn • CLV • CPA • ROI • Engagement — executive-ready analytics for growth and retention."
)

if df is None:
    st.warning("Upload a CSV (or include `data.csv` in the repo root) to begin.")
    st.stop()

# Light cleaning / coercion
df = coerce_numeric(
    df,
    [
        "Customer Lifetime Value",
        "Monthly Premium Auto",
        "Total Claim Amount",
        "Income",
        "EngagementScore",
    ],
)

st.success(f"{source_label} with {len(df):,} rows and {df.shape[1]} columns.")
st.markdown("### Data preview")
st.dataframe(df.head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# Page: KPIs overview
# -----------------------------------------------------------------------------
if page == "KPIs overview":
    st.markdown("## KPIs Overview – Growth & Profitability")

    kpi = compute_global_kpis(df, CHANNEL_COST_MAP)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{kpi['customers']:,}")
    c2.metric("Churn rate", f"{kpi['churn_rate'] * 100:0.1f}%")
    c3.metric("Conversion rate", f"{kpi['conversion_rate'] * 100:0.1f}%")
    c4.metric("Average CLV", f"${kpi['avg_clv']:,.0f}")

    c5, c6 = st.columns(2)
    c5.metric(
        "CPA (Cost per Acquisition)",
        f"${kpi['cpa']:,.0f}" if not math.isnan(kpi["cpa"]) else "n/a",
    )
    c6.metric(
        "ROI",
        f"{kpi['roi']:0.1f}%"
        if not math.isnan(kpi["roi"])
        else "n/a",
    )

    st.markdown("### How to read this section")
    st.markdown(
        """
- **Customers** – number of unique customers in the file.  
- **Churn rate** – share of customers labeled as churned (`Churn = Yes`).  
- **Conversion rate** – share of rows with a positive response (`Response = Yes`).  
- **Average CLV** – average *Customer Lifetime Value* for converted customers (falls back to overall mean if none converted).  
- **CPA** – marketing **cost per acquired customer**, using channel-level cost estimates in the sidebar.  
- **ROI** – return on investment: realized CLV from converted customers versus total acquisition spend.
        """
    )

    st.markdown("### KPIs by acquisition channel")
    seg_df = kpis_by_segment(df, CHANNEL_COST_MAP, "Sales Channel")
    st.dataframe(seg_df, use_container_width=True)

    if not seg_df.empty:
        # Conversion rate by channel
        fig = px.bar(
            seg_df,
            x="Sales Channel",
            y="conversion_rate",
            text=seg_df["conversion_rate"].mul(100).round(1).astype(str) + "%",
            labels={"conversion_rate": "Conversion rate"},
            title="Conversion rate by Sales Channel",
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        # ROI by channel
        fig2 = px.bar(
            seg_df,
            x="Sales Channel",
            y="roi",
            text=seg_df["roi"].round(1).astype(str) + "%",
            labels={"roi": "ROI (%)"},
            title="ROI by Sales Channel",
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Use this page for a high-level health check: are we acquiring profitable customers and where is ROI strongest?"
    )


# -----------------------------------------------------------------------------
# Page: Why People Churn
# -----------------------------------------------------------------------------
elif page == "Why People Churn":
    st.markdown("## Why People Churn")

    st.info(
        "This view shows where churn is highest so marketers can prioritize "
        "**save-campaigns**, retention journeys, and service interventions."
    )

    segment_options = [
        "Sales Channel",
        "Renew Offer Type",
        "Policy Type",
        "Coverage",
        "State",
        "Vehicle Size",
    ]
    segment_col = st.selectbox("View churn by segment", segment_options, index=0)

    seg_df = kpis_by_segment(df, CHANNEL_COST_MAP, segment_col)
    st.dataframe(seg_df, use_container_width=True)

    if not seg_df.empty:
        fig = px.bar(
            seg_df,
            x=segment_col,
            y="churn_rate",
            text=seg_df["churn_rate"].mul(100).round(1).astype(str) + "%",
            labels={"churn_rate": "Churn rate"},
            title=f"Churn rate by {segment_col}",
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
**How to interpret:**  
Segments with **high churn rate** but **meaningful customers or CLV** should be your top candidates for:
- Proactive outreach or save programs  
- Journey redesign (AJO, CRM, call center)  
- Pricing / benefit reviews  
"""
    )


# -----------------------------------------------------------------------------
# Page: Why People Convert
# -----------------------------------------------------------------------------
elif page == "Why People Convert":
    st.markdown("## Why People Convert")

    st.info(
        "This view highlights **what drives conversions** — which channels, offers, "
        "or segments are most effective at turning leads into customers."
    )

    segment_options = [
        "Sales Channel",
        "Renew Offer Type",
        "Policy Type",
        "Coverage",
        "State",
        "Vehicle Size",
    ]
    segment_col = st.selectbox("View conversion by", segment_options, index=0)

    seg_df = kpis_by_segment(df, CHANNEL_COST_MAP, segment_col)
    st.dataframe(seg_df, use_container_width=True)

    if not seg_df.empty:
        fig = px.bar(
            seg_df,
            x=segment_col,
            y="conversion_rate",
            text=seg_df["conversion_rate"].mul(100).round(1).astype(str) + "%",
            labels={"conversion_rate": "Conversion rate"},
            title=f"Conversion rate by {segment_col}",
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
**How to interpret:**  
- Segments with **high conversion rate** and **strong ROI** should be prioritized for scaling.  
- Segments with **low conversion** but **high CLV** may warrant targeted testing to unlock value.  
"""
    )


# -----------------------------------------------------------------------------
# Page: Why People Engage
# -----------------------------------------------------------------------------
elif page == "Why People Engage":
    st.markdown("## Why People Engage")

    st.info(
        "This section focuses on **EngagementScore** to show where customers are most active "
        "and which segments may need nurture or reactivation."
    )

    if "EngagementScore" not in df.columns:
        st.error("EngagementScore column is missing.")
    else:
        seg_col = st.selectbox(
            "View engagement by segment",
            ["Sales Channel", "State", "Renew Offer Type", "Policy Type", "Coverage"],
            index=0,
        )
        g = (
            df.groupby(seg_col)
            .agg(
                customers=("Customer", "nunique"),
                avg_engagement=("EngagementScore", "mean"),
                avg_clv=("Customer Lifetime Value", "mean"),
            )
            .reset_index()
            .sort_values("avg_engagement", ascending=False)
        )

        st.dataframe(g, use_container_width=True)

        if not g.empty:
            fig = px.bar(
                g,
                x=seg_col,
                y="avg_engagement",
                text=g["avg_engagement"].round(1),
                labels={"avg_engagement": "Average EngagementScore"},
                title=f"Average engagement by {seg_col}",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
**How to interpret:**  
- High engagement segments are prime for cross-sell, upsell, and advocacy programs.  
- Low engagement segments may need refreshed messaging, frequency, or channel mixes.  
"""
        )


# -----------------------------------------------------------------------------
# Remaining pages – simple placeholders (ready to be upgraded)
# -----------------------------------------------------------------------------
elif page == "Time Series Analysis":
    st.markdown("## Time Series Analysis")
    st.info(
        "Coming soon: trends over time in conversion, churn, CLV and engagement. "
        "This view will use `Effective To Date` to group by week/month."
    )

    if "Effective To Date" in df.columns:
        # Simple prototype: monthly conversion over time
        d = df.copy()
        d["Effective To Date"] = pd.to_datetime(d["Effective To Date"], errors="coerce")
        d = d.dropna(subset=["Effective To Date"])
        d["month"] = d["Effective To Date"].dt.to_period("M").dt.to_timestamp()
        d["is_convert"] = yes_no_flag(d["Response"])

        ts = (
            d.groupby("month")
            .agg(
                customers=("Customer", "nunique"),
                conversion_rate=("is_convert", "mean"),
            )
            .reset_index()
        )
        st.dataframe(ts, use_container_width=True)

        fig = px.line(
            ts,
            x="month",
            y="conversion_rate",
            markers=True,
            labels={"conversion_rate": "Conversion rate"},
            title="Conversion rate over time",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


elif page == "Sentiment Analysis":
    st.markdown("## Sentiment Analysis")
    st.info(
        "This section is reserved for NLP-based sentiment scoring on call notes, emails, "
        "or survey verbatims. It will expect a `SentimentScore` or text column in future versions."
    )


elif page == "Predictive Analytics":
    st.markdown("## Predictive Analytics")
    st.info(
        "This page will host uplift / propensity models for conversion and churn "
        "(e.g., Random Forest, XGBoost, and Stacking Generative AI). "
        "In the current version, the calibrated KPIs and by-segment views already "
        "provide strong guidance for targeting and experimentation."
    )


elif page == "Product Recommendations":
    st.markdown("## Product Recommendations")
    st.info(
        "This section will surface personalized product or coverage recommendations "
        "based on CLV, risk, and engagement patterns (e.g., association rules or "
        "matrix factorization on policy holdings)."
    )


elif page == "Customer Segmentation":
    st.markdown("## Customer Segmentation")
    st.info(
        "This view will cluster customers into actionable segments (e.g., k-means on "
        "CLV, engagement, channel preference, and demographics) and display profiles "
        "to inform strategy, creative, and measurement."
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    "<p style='text-align:center;color:gray;font-size:12px;margin-top:2rem;'>"
    "© 2025 Howard Nguyen, PhD – TopKPI2 • AI-powered growth, retention & profitability"
    "</p>",
    unsafe_allow_html=True,
)
