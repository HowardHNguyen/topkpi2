import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import streamlit as st


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TopKPI2 – Advanced Marketing Growth & Retention Intelligence",
    layout="wide",
    page_icon="📈",
)

# (Optional) You can re-add your CSS to hide Streamlit chrome here if you like.
# st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Constants & schema
# -----------------------------------------------------------------------------
MAIN_DEFAULT_CSV = "data.csv"

MAIN_REQUIRED_COLS: List[str] = [
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
    # New required fields for TopKPI2:
    "Churn",
    "EngagementScore",
]

RETAIL_REQUIRED_COLS: List[str] = [
    "InvoiceNo",
    "InvoiceDate",
    "StockCode",
    "Description",
    "Quantity",
    "UnitPrice",
    "CustomerID",
]


CHANNEL_COST_DEFAULTS = {
    "Web": 40.0,
    "Call Center": 70.0,
    "Branch": 90.0,
    "Agent": 120.0,
}


# -----------------------------------------------------------------------------
# Utility loaders
# -----------------------------------------------------------------------------
def load_main_marketing_data() -> Optional[pd.DataFrame]:
    """Load the marketing data (data.csv schema)."""

    up = st.sidebar.file_uploader(
        "Upload CSV (same schema as your notebooks)",
        type="csv",
        key="main_csv",
    )

    if up is not None:
        df = pd.read_csv(up)
        st.success("Uploaded CSV successfully.")
    elif os.path.exists(MAIN_DEFAULT_CSV):
        df = pd.read_csv(MAIN_DEFAULT_CSV)
        st.info(
            "Loaded data.csv from repo root. "
            "Upload your own CSV to replace it."
        )
    else:
        st.info("Upload marketing data (data.csv) to begin.")
        return None

    # Drop unnamed index columns if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    return df


def check_main_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in MAIN_REQUIRED_COLS if c not in df.columns]
    return len(missing) == 0, missing


def load_online_retail(upload_key: str = "retail_csv") -> Optional[pd.DataFrame]:
    """
    Load the Online Retail dataset used for Product Recommendations & Segmentation.
    We allow the user to upload once and reuse via session_state["retail_df"].
    """

    # Already in session?
    if "retail_df" in st.session_state:
        return st.session_state["retail_df"]

    up = st.file_uploader(
        "Upload Online Retail CSV (InvoiceNo, InvoiceDate, StockCode, "
        "Description, Quantity, UnitPrice, CustomerID)",
        type="csv",
        key=upload_key,
    )

    if up is None:
        st.info(
            "Upload the Online Retail dataset (e.g., online_retail or "
            "online_retail_II) to use this section."
        )
        return None

    # Try UTF-8 then fall back to ISO-8859-1 (common for this dataset)
    for enc in ("utf-8", "ISO-8859-1"):
        try:
            df = pd.read_csv(up, encoding=enc)
            break
        except UnicodeDecodeError:
            df = None

    if df is None:
        st.error("Could not decode CSV. Try re-saving as UTF-8 or ISO-8859-1.")
        return None

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    st.session_state["retail_df"] = df
    st.success("Online Retail data loaded and cached for this session.")
    return df


def check_retail_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in RETAIL_REQUIRED_COLS if c not in df.columns]
    return len(missing) == 0, missing


# -----------------------------------------------------------------------------
# KPI helpers (marketing dataset)
# -----------------------------------------------------------------------------
def prepare_main_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add convenience flags and typed columns."""
    df = df.copy()

    # conversion flag from Response
    df["conversion_flag"] = (
        df["Response"].astype(str).str.strip().str.lower().eq("yes")
    ).astype(int)

    # engaged flag from Response (per your definition)
    df["Engaged"] = df["conversion_flag"]

    # ensure Churn is numeric 0/1
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)

    # EngagementScore numeric
    df["EngagementScore"] = pd.to_numeric(
        df["EngagementScore"], errors="coerce"
    ).fillna(0.0)

    # Basic date parsing
    df["Effective To Date"] = pd.to_datetime(
        df["Effective To Date"], errors="coerce"
    )

    # Normalize CLV
    df["Customer Lifetime Value"] = pd.to_numeric(
        df["Customer Lifetime Value"], errors="coerce"
    ).fillna(0.0)

    return df


def compute_global_kpis(
    df: pd.DataFrame, channel_cost_map: Dict[str, float]
) -> Dict[str, float]:
    """Compute global customers, churn, conversion, CLV, CPA, ROI."""
    d = df.copy()

    n_customers = d["Customer"].nunique()
    churn_rate = d["Churn"].mean() if "Churn" in d.columns else 0.0
    conversion_rate = d["conversion_flag"].mean()

    # Cost per acquisition, ROI – using channel cost overrides
    d["channel_cost"] = d["Sales Channel"].map(channel_cost_map).fillna(0.0)

    total_cost = d["channel_cost"].sum()
    acquired = max(d["conversion_flag"].sum(), 1)  # avoid /0
    cpa = total_cost / acquired

    clv_realized = d.loc[d["conversion_flag"] == 1, "Customer Lifetime Value"].sum()
    roi = ((clv_realized - total_cost) / total_cost * 100.0) if total_cost > 0 else 0.0

    avg_clv = d["Customer Lifetime Value"].mean()

    return {
        "customers": n_customers,
        "churn_rate": churn_rate,
        "conversion_rate": conversion_rate,
        "avg_clv": avg_clv,
        "cpa": cpa,
        "roi": roi,
        "acquired": acquired,
        "total_cost": total_cost,
        "clv_realized": clv_realized,
    }


def channel_kpis(df: pd.DataFrame, channel_cost_map: Dict[str, float]) -> pd.DataFrame:
    d = df.copy()
    d["channel_cost"] = d["Sales Channel"].map(channel_cost_map).fillna(0.0)

    grp = (
        d.groupby("Sales Channel")
        .agg(
            customers=("Customer", "nunique"),
            leads=("Customer", "size"),
            converts=("conversion_flag", "sum"),
            churners=("Churn", "sum"),
            avg_clv=("Customer Lifetime Value", "mean"),
            engagement=("EngagementScore", "mean"),
            cost=("channel_cost", "sum"),
        )
        .reset_index()
    )

    grp["conversion_rate"] = grp["converts"] / grp["leads"].clip(lower=1)
    grp["churn_rate"] = grp["churners"] / grp["customers"].clip(lower=1)
    grp["cpa"] = grp["cost"] / grp["converts"].clip(lower=1)
    grp["roi"] = (
        grp["avg_clv"] * grp["converts"] - grp["cost"]
    ) / grp["cost"].replace(0, np.nan) * 100.0

    return grp


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("TopKPI2 📈")
st.sidebar.caption(
    "Advanced Marketing Growth & Retention Intelligence\n\n"
    "Conversion • Churn • CLV • CPA • ROI • Engagement"
)

st.sidebar.subheader("Cost per Acquisition (override)")
channel_cost_map = {}
for ch, default in CHANNEL_COST_DEFAULTS.items():
    channel_cost_map[ch] = st.sidebar.number_input(
        ch,
        min_value=0.0,
        max_value=10_000.0,
        value=float(default),
        step=1.0,
        key=f"cost_{ch.replace(' ', '_')}",
    )

st.sidebar.markdown("---")
view = st.sidebar.radio(
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

with st.sidebar.expander("Schema checklist", expanded=True):
    st.markdown("**Marketing data (data.csv)**")
    st.write("Required columns:")
    st.code(", ".join(MAIN_REQUIRED_COLS), language="text")

    st.markdown("**Online Retail (for Product Recs & Segmentation)**")
    st.write("Required columns:")
    st.code(", ".join(RETAIL_REQUIRED_COLS), language="text")


# -----------------------------------------------------------------------------
# Load marketing dataset once
# -----------------------------------------------------------------------------
main_df = load_main_marketing_data()
if main_df is not None:
    ok_main, missing_main = check_main_schema(main_df)
    if not ok_main:
        st.error(f"Missing required columns for marketing dataset: {missing_main}")
    else:
        st.success(
            f"Uploaded data.csv with {len(main_df):,} rows and "
            f"{len(main_df.columns)} columns."
        )
        st.dataframe(main_df.head(), use_container_width=True)
        main_df = prepare_main_df(main_df)


# -----------------------------------------------------------------------------
# View: KPIs overview
# -----------------------------------------------------------------------------
def page_kpis_overview(df: pd.DataFrame):
    st.subheader("KPIs Overview – Growth & Profitability")

    overall = compute_global_kpis(df, channel_cost_map)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{overall['customers']:,}")
    c2.metric("Churn rate", f"{overall['churn_rate']*100:,.2f}%")
    c3.metric("Conversion rate", f"{overall['conversion_rate']*100:,.2f}%")
    c4.metric("Average CLV", f"${overall['avg_clv']:,.0f}")

    c5, c6 = st.columns(2)
    c5.metric("Cost per Acquisition (CPA)", f"${overall['cpa']:,.0f}")
    c6.metric("ROI", f"{overall['roi']:,.2f}%")

    with st.expander("How to read this section", expanded=True):
        st.markdown(
            """
- **Customers** – distinct customers in the file.  
- **Churn rate** – share of customers flagged as churned (`Churn = 1`).  
- **Conversion rate** – share of rows with `Response = "Yes"`.  
- **Average CLV** – average `Customer Lifetime Value`.  
- **CPA** – total channel cost ÷ acquired customers.  
- **ROI** – (realized CLV – cost) ÷ cost.
            """
        )

    chan_df = channel_kpis(df, channel_cost_map)

    st.markdown("### KPIs by acquisition channel")
    st.dataframe(chan_df, use_container_width=True)

    fig_conv = px.bar(
        chan_df,
        x="Sales Channel",
        y="conversion_rate",
        title="Conversion rate by Sales Channel",
        text=chan_df["conversion_rate"].map(lambda x: f"{x*100:,.1f}%"),
    )
    fig_conv.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_conv, use_container_width=True)

    fig_roi = px.bar(
        chan_df,
        x="Sales Channel",
        y="roi",
        title="ROI by Sales Channel",
        text=chan_df["roi"].map(lambda x: f"{x:,.1f}%"),
    )
    st.plotly_chart(fig_roi, use_container_width=True)


# -----------------------------------------------------------------------------
# View: Why People Churn
# -----------------------------------------------------------------------------
def page_why_churn(df: pd.DataFrame):
    st.subheader("Why People Churn")

    st.write(
        "This view shows where churn is highest so marketers can prioritize "
        "save-campaigns and service interventions."
    )

    segment = st.selectbox(
        "View churn by segment",
        ["Sales Channel", "State", "Coverage", "Education", "Marital Status"],
    )

    grp = (
        df.groupby(segment)
        .agg(
            customers=("Customer", "nunique"),
            churners=("Churn", "sum"),
        )
        .reset_index()
    )
    grp["churn_rate"] = grp["churners"] / grp["customers"].clip(lower=1)

    st.dataframe(grp, use_container_width=True)

    fig = px.bar(
        grp,
        x=segment,
        y="churn_rate",
        title=f"Churn rate by {segment}",
        text=grp["churn_rate"].map(lambda x: f"{x*100:,.1f}%"),
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# View: Why People Convert
# -----------------------------------------------------------------------------
def page_why_convert(df: pd.DataFrame):
    st.subheader("Why People Convert")

    st.write(
        "This view highlights **what drives conversions** – which channels, "
        "offers, or segments are most effective at turning leads into customers."
    )

    segment = st.selectbox(
        "View conversion by",
        ["Sales Channel", "State", "Renew Offer Type", "Policy Type"],
    )

    grp = (
        df.groupby(segment)
        .agg(
            leads=("Customer", "size"),
            converts=("conversion_flag", "sum"),
        )
        .reset_index()
    )
    grp["conversion_rate"] = grp["converts"] / grp["leads"].clip(lower=1)

    st.dataframe(grp, use_container_width=True)

    fig = px.bar(
        grp,
        x=segment,
        y="conversion_rate",
        title=f"Conversion rate by {segment}",
        text=grp["conversion_rate"].map(lambda x: f"{x*100:,.1f}%"),
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# View: Why People Engage
# -----------------------------------------------------------------------------
def page_why_engage(df: pd.DataFrame):
    st.subheader("Why People Engage")

    st.info(
        "This section uses **Response** and **EngagementScore** to understand "
        "which segments are most engaged.\n\n"
        "- `Engaged = 1` when `Response = \"Yes\"`.\n"
        "- `EngagementScore` captures intensity (e.g., clicks, opens, time on site)."
    )

    segment = st.selectbox(
        "View engagement by",
        ["Sales Channel", "State", "Coverage", "Education", "Marital Status"],
    )

    grp = (
        df.groupby(segment)
        .agg(
            customers=("Customer", "nunique"),
            engaged=("Engaged", "sum"),
            avg_score=("EngagementScore", "mean"),
        )
        .reset_index()
    )
    grp["engaged_rate"] = grp["engaged"] / grp["customers"].clip(lower=1)

    st.dataframe(grp, use_container_width=True)

    fig = px.bar(
        grp,
        x=segment,
        y="engaged_rate",
        title=f"Engaged customers by {segment}",
        text=grp["engaged_rate"].map(lambda x: f"{x*100:,.1f}%"),
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(
        grp,
        x=segment,
        y="avg_score",
        title=f"Average EngagementScore by {segment}",
        text=grp["avg_score"].map(lambda x: f"{x:,.1f}"),
    )
    st.plotly_chart(fig2, use_container_width=True)


# -----------------------------------------------------------------------------
# View: Time Series Analysis
# -----------------------------------------------------------------------------
def page_time_series(df: pd.DataFrame):
    st.subheader("Time Series Analysis")

    st.write(
        "This view looks at conversions and churn over time, using "
        "`Effective To Date` as a proxy for campaign period."
    )

    ts = df.dropna(subset=["Effective To Date"]).copy()
    ts["date"] = ts["Effective To Date"].dt.to_period("M").dt.to_timestamp()

    grp = (
        ts.groupby("date")
        .agg(
            customers=("Customer", "size"),
            converts=("conversion_flag", "sum"),
            churners=("Churn", "sum"),
        )
        .reset_index()
    )
    grp["conversion_rate"] = grp["converts"] / grp["customers"].clip(lower=1)
    grp["churn_rate"] = grp["churners"] / grp["customers"].clip(lower=1)

    st.dataframe(grp, use_container_width=True)

    fig = px.line(
        grp,
        x="date",
        y=["conversion_rate", "churn_rate"],
        title="Conversion vs Churn over time",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# View: Sentiment Analysis (placeholder)
# -----------------------------------------------------------------------------
def page_sentiment(df: pd.DataFrame):
    st.subheader("Sentiment Analysis (placeholder)")

    st.info(
        "In this dataset we don’t yet have free-text feedback, call notes, or "
        "survey comments. In a real deployment, this section would:\n\n"
        "- Ingest NPS comments, call center notes, or email bodies.\n"
        "- Run NLP to derive sentiment scores and topics.\n"
        "- Tie those scores back to **Churn**, **EngagementScore**, and **CLV**."
    )


# -----------------------------------------------------------------------------
# View: Predictive Analytics (placeholder)
# -----------------------------------------------------------------------------
def page_predictive(df: pd.DataFrame):
    st.subheader("Predictive Analytics (placeholder)")

    st.info(
        "This section is where you’d plug in a production-grade model "
        "(e.g., gradient boosting, XGBoost, or your Stacking GenAI model) "
        "to score each customer for **churn risk**, **conversion propensity**, "
        "or **upsell likelihood**.\n\n"
        "The current app focuses on interpretable KPI slices, but the same "
        "schema can feed more advanced models."
    )


# -----------------------------------------------------------------------------
# View: Product Recommendations (Online Retail dataset)
# -----------------------------------------------------------------------------
def page_product_recs():
    st.subheader("Product Recommendations")

    st.write(
        "This view suggests **“people who bought X also bought Y”** using the "
        "Online Retail dataset. It looks at products that co-occur on the same "
        "invoices and surfaces the top companion items."
    )

    retail_df = load_online_retail(upload_key="retail_csv_for_recs")
    if retail_df is None:
        return

    ok, missing = check_retail_schema(retail_df)
    if not ok:
        st.error(f"Missing required columns for Online Retail dataset: {missing}")
        return

    # Basic cleaning
    df = retail_df.copy()
    df = df.dropna(subset=["InvoiceNo", "StockCode", "Description", "CustomerID"])
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df = df[df["Quantity"] > 0]

    st.markdown("#### Top Selling Products")
    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .reset_index()
    )
    st.dataframe(top_products, use_container_width=True)

    base_product = st.selectbox(
        "Choose a base product to recommend from:",
        options=top_products["Description"],
    )

    base_invoices = df.loc[df["Description"] == base_product, "InvoiceNo"].unique()
    co_df = df[df["InvoiceNo"].isin(base_invoices)]
    co_df = co_df[co_df["Description"] != base_product]

    if co_df.empty:
        st.warning("No co-purchases found for this product (in this sample).")
        return

    co_counts = (
        co_df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="co_purchase_quantity")
    )

    st.markdown(f"#### Frequently bought with **{base_product}**")
    st.dataframe(co_counts, use_container_width=True)

    fig = px.bar(
        co_counts,
        x="Description",
        y="co_purchase_quantity",
        title=f"Top companion products for {base_product}",
        text="co_purchase_quantity",
    )
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to use this section"):
        st.markdown(
            """
- Use this table to design **bundle offers** and **cross-sell campaigns**.  
- The logic is simple, transparent co-occurrence (no black-box model).  
- You can export the table and join it back to your product master for pricing and attribution.
            """
        )


# -----------------------------------------------------------------------------
# View: Customer Segmentation (Online Retail dataset)
# -----------------------------------------------------------------------------
def page_customer_segmentation():
    st.subheader("Customer Segmentation")

    st.write(
        "This view clusters customers into actionable segments using an "
        "**RFM approach** (Recency, Frequency, Monetary) based on the "
        "Online Retail dataset."
    )

    retail_df = load_online_retail(upload_key="retail_csv_for_seg")
    if retail_df is None:
        return

    ok, missing = check_retail_schema(retail_df)
    if not ok:
        st.error(f"Missing required columns for Online Retail dataset: {missing}")
        return

    df = retail_df.copy()
    df = df.dropna(subset=["InvoiceNo", "CustomerID", "InvoiceDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    df["Amount"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0) * pd.to_numeric(
        df["UnitPrice"], errors="coerce"
    ).fillna(0)
    df = df[df["Amount"] > 0]

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            frequency=("InvoiceNo", "nunique"),
            monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    st.markdown("#### RFM summary")
    st.dataframe(rfm.head(), use_container_width=True)

    # KMeans clustering
    features = rfm[["recency", "frequency", "monetary"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    k = st.slider("Number of segments (k-means clusters)", 3, 8, value=4, step=1)
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    rfm["segment"] = model.fit_predict(X)

    st.markdown("#### Segment profile (avg RFM by cluster)")
    seg_profile = (
        rfm.groupby("segment")
        .agg(
            customers=("CustomerID", "nunique"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
    )
    st.dataframe(seg_profile, use_container_width=True)

    fig = px.scatter_3d(
        rfm,
        x="recency",
        y="frequency",
        z="monetary",
        color="segment",
        title="RFM Segments (3D view)",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read this section", expanded=True):
        st.markdown(
            """
- **Recency** – days since the customer's last purchase.  
- **Frequency** – number of distinct invoices.  
- **Monetary** – total revenue from the customer.  

Typical patterns:  
- **High frequency & monetary, low recency** → VIP / loyal.  
- **Low frequency & monetary, high recency** → churn-risk or one-time buyers.  
- Use segments to tailor **loyalty**, **reactivation**, and **cross-sell** campaigns.
            """
        )


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if main_df is None and view not in ["Product Recommendations", "Customer Segmentation"]:
    st.stop()

if view == "KPIs overview" and main_df is not None:
    page_kpis_overview(main_df)
elif view == "Why People Churn" and main_df is not None:
    page_why_churn(main_df)
elif view == "Why People Convert" and main_df is not None:
    page_why_convert(main_df)
elif view == "Why People Engage" and main_df is not None:
    page_why_engage(main_df)
elif view == "Time Series Analysis" and main_df is not None:
    page_time_series(main_df)
elif view == "Sentiment Analysis" and main_df is not None:
    page_sentiment(main_df)
elif view == "Predictive Analytics" and main_df is not None:
    page_predictive(main_df)
elif view == "Product Recommendations":
    page_product_recs()
elif view == "Customer Segmentation":
    page_customer_segmentation()

st.markdown(
    "<br><center>© 2025 Howard Nguyen, PhD — TopKPI2 • AI-powered growth, "
    "retention & profitability</center>",
    unsafe_allow_html=True,
)
