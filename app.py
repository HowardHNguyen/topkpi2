# app.py – TopKPI2: AI Growth & Retention Intelligence
import os
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import streamlit as st

# ---------------------------------------------------------
# 0. PAGE CONFIG & CHROME CLEANUP (keep left sidebar)
# ---------------------------------------------------------
st.set_page_config(
    page_title="TopKPI2 – AI Growth & Retention Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep sidebar & chevron; hide only top-right & bottom-right Streamlit chrome
st.markdown("""
<style>
header { visibility: visible !important; }
div[data-testid="stDecoration"] { display: block !important; visibility: visible !important; }
div[data-testid="stSidebar"] { visibility: visible !important; display: block !important; }

/* Hide ONLY the top-right toolbar actions (Fork / GitHub / ⋮) */
div[data-testid="stToolbarActions"] { display: none !important; }
div[data-testid="stToolbar"] { display: block !important; visibility: visible !important; }

/* Hide bottom-right watermark/buttons */
.stAppBottomRightButtons, .stAppDeployButton { display: none !important; }

/* Optional: hide legacy text MainMenu (does not affect sidebar) */
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. SIDEBAR – NAV + DATA UPLOAD
# ---------------------------------------------------------
st.sidebar.title("TopKPI2 📈")
st.sidebar.caption("Advanced Marketing Growth & Retention Intelligence")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (same schema as your notebooks)", type=["csv"]
)

# high-level navigation
section = st.sidebar.radio(
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

# small schema helper so marketers know when data is “ready”
with st.sidebar.expander("Schema checklist", expanded=False):
    st.write(
        "This dashboard is designed for customer-level marketing data. "
        "Common columns that unlock more insights include:"
    )
    st.markdown(
        "- **Customer** or **Customer_ID**\n"
        "- **Response**, **Churn**, or **Converted**\n"
        "- **Channel** (e.g., Web / Call Center / Branch / Agent)\n"
        "- **Offer** / **Campaign**\n"
        "- **Customer Lifetime Value** or **CLV**\n"
        "- **Effective To Date** / **Date** for time trends\n"
        "- **Text feedback** (for sentiment)\n"
    )
    st.caption(
        "If a section is missing required columns, the app will show guidance "
        "instead of an error."
    )

# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

df = None
if uploaded_file:
    df = load_csv(uploaded_file)
    st.success(f"Loaded `{uploaded_file.name}` with {len(df):,} rows and {len(df.columns)} columns.")
else:
    st.info("Upload a CSV in the sidebar to activate all sections.")

# Always show a small preview if data is present
if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------------
# 3. HELPER: CHECK COLUMNS
# -------------------------------------------------------------------
def has_cols(d: pd.DataFrame, cols: List[str]) -> bool:
    return d is not None and all(c in d.columns for c in cols)

def first_present(d: pd.DataFrame, candidates: List[str]):
    """Return first column name that exists in df from candidates, else None."""
    if d is None:
        return None
    for c in candidates:
        if c in d.columns:
            return c
    return None

# -------------------------------------------------------------------
# 4. SECTION FUNCTIONS
# -------------------------------------------------------------------
def section_kpis_overview(df: pd.DataFrame):
    st.header("KPIs Overview – Growth & Profitability")

    if df is None:
        st.warning("Upload a CSV to compute KPIs.")
        return

    # Try to infer key columns
    churn_col = first_present(df, ["Churn", "churn", "Attrited", "Response"])
    conv_col  = first_present(df, ["conversion", "Converted", "Response"])
    clv_col   = first_present(df, ["Customer Lifetime Value", "CLV", "clv"])
    revenue_col = first_present(df, ["Revenue", "Total Claim Amount", "Premium", "Sales"])
    cost_col  = first_present(df, ["Cost", "AcquisitionCost", "cpa"])

    n_customers = df.shape[0]
    n_unique = df[first_present(df, ["Customer", "Customer_ID", "ID"])
                  ].nunique() if first_present(df, ["Customer", "Customer_ID", "ID"]) else n_customers

    cols = st.columns(4)
    with cols[0]:
        st.metric("Customers", f"{n_unique:,}")
    with cols[1]:
        if churn_col is not None:
            rate = df[churn_col].mean() if df[churn_col].dropna().isin([0,1]).all() else \
                df[churn_col].eq(1).mean()
            st.metric("Churn rate", f"{rate*100:0.1f}%")
        else:
            st.metric("Churn rate", "N/A")
    with cols[2]:
        if conv_col is not None:
            cr = df[conv_col].mean() if df[conv_col].dropna().isin([0,1]).all() else \
                df[conv_col].eq(1).mean()
            st.metric("Conversion rate", f"{cr*100:0.1f}%")
        else:
            st.metric("Conversion rate", "N/A")
    with cols[3]:
        if clv_col is not None:
            st.metric("Average CLV", f"${df[clv_col].mean():,.0f}")
        else:
            st.metric("Average CLV", "N/A")

    st.markdown("### How to read this section")
    st.markdown(
        """
        - **Customers** – number of unique customers in the file.  
        - **Churn rate** – share of customers labeled as churned.  
        - **Conversion rate** – share of rows with a positive response / conversion.  
        - **Average CLV** – typical lifetime value per customer (if CLV is present).  
        """
    )

    # Channel-based KPI table if we can find a channel column
    channel_col = first_present(df, ["Sales Channel", "Channel", "channel", "AcquisitionChannel"])
    if channel_col and conv_col:
        st.markdown("### KPIs by acquisition channel")
        by_ch = (
            df.groupby(channel_col)
            .agg(
                customers=("Customer", "nunique") if "Customer" in df.columns else (channel_col, "size"),
                conversion_rate=(conv_col, lambda x: np.mean(x==1) if set(x.dropna().unique()) <= {0,1} else np.nan),
            )
            .reset_index()
        )
        by_ch["conversion_rate"] = by_ch["conversion_rate"] * 100
        st.dataframe(by_ch, use_container_width=True)
        fig = px.bar(
            by_ch,
            x=channel_col,
            y="conversion_rate",
            text="conversion_rate",
            labels={"conversion_rate": "Conversion rate (%)"},
            title="Conversion rate by channel",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(yaxis_tickformat=".0f", uniformtext_minsize=8, uniformtext_mode="hide")
        st.plotly_chart(fig, use_container_width=True)


def section_why_churn(df: pd.DataFrame):
    st.header("Why People Churn")

    if df is None:
        st.warning("Upload a CSV to analyze churn drivers.")
        return

    churn_col = first_present(df, ["Churn", "churn", "Attrited", "Response"])
    seg_cols = [c for c in ["Sales Channel", "Channel", "State", "Region", "Segment", "Policy Type"]
                if c in df.columns]

    if churn_col is None:
        st.info("This section expects a binary **Churn** / **Response** column (0/1).")
        return

    st.markdown(
        "This view shows where churn is highest so marketers can prioritize **save-campaigns** "
        "and **service interventions**."
    )

    if seg_cols:
        seg_col = st.selectbox("View churn by segment", seg_cols)
        tmp = (
            df.groupby(seg_col)
            .agg(
                customers=(churn_col, "size"),
                churn_rate=(churn_col, lambda x: np.mean(x == 1)),
            )
            .reset_index()
        )
        tmp["churn_rate"] = tmp["churn_rate"] * 100
        st.dataframe(tmp, use_container_width=True)

        fig = px.bar(
            tmp.sort_values("churn_rate", ascending=False),
            x=seg_col,
            y="churn_rate",
            text="churn_rate",
            title=f"Churn rate by {seg_col}",
            labels={"churn_rate": "Churn rate (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add columns like **Sales Channel**, **Segment**, or **Policy Type** to segment churn.")


def section_why_convert(df: pd.DataFrame):
    st.header("Why People Convert")

    if df is None:
        st.warning("Upload a CSV to analyze conversion drivers.")
        return

    conv_col = first_present(df, ["conversion", "Converted", "Response"])
    if conv_col is None:
        st.info("This section expects a binary **conversion/Converted/Response** column.")
        return

    st.markdown(
        "This view highlights **what drives conversions** – which channels, offers, or segments "
        "are most effective at turning leads into customers."
    )

    seg_cols = [c for c in ["Sales Channel", "Channel", "Renew Offer Type", "Offer", "Campaign"]
                if c in df.columns]
    if not seg_cols:
        st.info("Add columns like **Sales Channel**, **Renew Offer Type**, or **Campaign** to break down conversion.")
        return

    seg_col = st.selectbox("View conversion by", seg_cols)
    tmp = (
        df.groupby(seg_col)
        .agg(
            leads=(conv_col, "size"),
            converts=(conv_col, lambda x: (x == 1).sum()),
        )
        .reset_index()
    )
    tmp["conversion_rate"] = (tmp["converts"] / tmp["leads"]) * 100
    st.dataframe(tmp, use_container_width=True)

    fig = px.bar(
        tmp.sort_values("conversion_rate", ascending=False),
        x=seg_col,
        y="conversion_rate",
        text="conversion_rate",
        title=f"Conversion rate by {seg_col}",
        labels={"conversion_rate": "Conversion rate (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def section_why_engage(df: pd.DataFrame):
    st.header("Why People Engage")

    if df is None:
        st.warning("Upload a CSV to analyze engagement.")
        return

    engage_col = first_present(df, ["EngagementScore", "Engagement", "Clicks", "Opens"])
    channel_col = first_present(df, ["Sales Channel", "Channel", "Touchpoint"])

    if engage_col is None:
        st.info(
            "This section expects some engagement proxy such as **EngagementScore**, "
            "**Clicks**, or **Opens**."
        )
        return

    st.markdown(
        "Use this section to understand **where customers are most engaged** – which "
        "channels or tactics keep them active."
    )

    if channel_col:
        tmp = (
            df.groupby(channel_col)
            .agg(avg_engagement=(engage_col, "mean"))
            .reset_index()
        )
        st.dataframe(tmp, use_container_width=True)
        fig = px.bar(
            tmp.sort_values("avg_engagement", ascending=False),
            x=channel_col,
            y="avg_engagement",
            text="avg_engagement",
            title=f"Average engagement by {channel_col}",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add a **Channel** column to break down engagement by touchpoint.")


def section_timeseries(df: pd.DataFrame):
    st.header("Time Series Analysis")

    if df is None:
        st.warning("Upload a CSV to analyze time trends.")
        return

    date_col = first_present(df, ["Effective To Date", "Date", "date", "CampaignDate"])
    conv_col = first_present(df, ["conversion", "Converted", "Response"])
    churn_col = first_present(df, ["Churn", "churn", "Attrited"])

    if date_col is None:
        st.info("This section expects a date column (e.g., **Effective To Date** or **Date**).")
        return

    st.markdown(
        "This view shows how **conversion, churn, or volume** trends over time, "
        "useful for spotting seasonality and campaign impact."
    )

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d["month"] = d[date_col].dt.to_period("M").dt.to_timestamp()

    agg = d.groupby("month").size().rename("n").reset_index()
    figs = []

    if conv_col:
        conv = (
            d.groupby("month")[conv_col]
            .apply(lambda x: np.mean(x == 1))
            .rename("conversion_rate")
            .reset_index()
        )
        conv["conversion_rate"] *= 100
        figs.append(("conversion_rate", conv))

    if churn_col:
        churn = (
            d.groupby("month")[churn_col]
            .apply(lambda x: np.mean(x == 1))
            .rename("churn_rate")
            .reset_index()
        )
        churn["churn_rate"] *= 100
        figs.append(("churn_rate", churn))

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(agg, x="month", y="n", markers=True, title="Volume over time")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if figs:
            for name, fdf in figs:
                fig = px.line(
                    fdf,
                    x="month",
                    y=name,
                    markers=True,
                    title=name.replace("_", " ").title(),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add **conversion** or **Churn** columns to see rate trends as well.")


def section_sentiment(df: pd.DataFrame):
    st.header("Sentiment Analysis")

    if df is None:
        st.warning("Upload a CSV with customer feedback / text columns.")
        return

    text_col = first_present(df, ["Feedback", "Comments", "Review", "Text"])
    sent_col = first_present(df, ["sentiment_score", "Sentiment", "Polarity"])

    if text_col is None and sent_col is None:
        st.info(
            "This section expects either:\n"
            "- a pre-computed **sentiment_score** column, or\n"
            "- a **Feedback/Comments/Review** column that you score offline.\n\n"
            "The idea is to bring in sentiment already computed in your notebooks, "
            "and then visualize it here."
        )
        return

    if sent_col is None:
        st.info(
            "You have text feedback but no sentiment scores yet. "
            "Score sentiment in your notebook, add a `sentiment_score` column, "
            "and reload the CSV."
        )
        return

    st.markdown(
        "This view summarizes **how customers feel** in their own words, "
        "and which segments are most positive or negative."
    )

    st.histogram = st.plotly_chart(
        px.histogram(
            df,
            x=sent_col,
            nbins=30,
            title="Sentiment score distribution",
        ),
        use_container_width=True,
    )

    seg_col = first_present(df, ["Sales Channel", "Channel", "Segment", "State"])
    if seg_col:
        tmp = (
            df.groupby(seg_col)[sent_col]
            .mean()
            .rename("avg_sentiment")
            .reset_index()
        )
        fig = px.bar(
            tmp.sort_values("avg_sentiment"),
            x="avg_sentiment",
            y=seg_col,
            orientation="h",
            title="Average sentiment by segment",
        )
        st.plotly_chart(fig, use_container_width=True)


def section_predictive(df: pd.DataFrame):
    st.header("Predictive Analytics & Propensity")

    if df is None:
        st.warning("Upload a CSV to view predictive scores.")
        return

    prop_col = first_present(df, ["propensity", "score", "predicted_prob", "p_convert"])
    conv_col = first_present(df, ["conversion", "Converted", "Response"])

    if prop_col is None:
        st.info(
            "This section expects a propensity / model score column such as "
            "**propensity**, **score**, or **predicted_prob** exported from your notebook."
        )
        return

    st.markdown(
        "Use this section to see how well your **model distinguishes converters** "
        "from non-converters and to choose a practical threshold."
    )

    st.plotly_chart(
        px.histogram(df, x=prop_col, nbins=30, title="Propensity score distribution"),
        use_container_width=True,
    )

    if conv_col is not None:
        # simple lift by decile
        d = df[[prop_col, conv_col]].dropna().copy()
        d["decile"] = pd.qcut(d[prop_col], 10, labels=False, duplicates="drop") + 1
        lift = (
            d.groupby("decile")[conv_col]
            .mean()
            .rename("conversion_rate")
            .reset_index()
        )
        lift["lift"] = lift["conversion_rate"] / lift["conversion_rate"].mean()

        st.subheader("Lift by decile")
        st.dataframe(lift, use_container_width=True)
        fig = px.line(
            lift,
            x="decile",
            y="lift",
            markers=True,
            title="Model lift curve",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add a **conversion/Converted** column to calculate lift vs. actual outcomes.")


def section_product_reco(df: pd.DataFrame):
    st.header("Product Recommendations (Cross-sell signals)")

    if df is None:
        st.warning("Upload a CSV with product / policy columns.")
        return

    prod_col = first_present(df, ["Product", "Policy Type", "Policy"])
    if prod_col is None:
        st.info(
            "This section expects a product-like column such as **Product**, "
            "**Policy Type**, or **Policy**."
        )
        return

    st.markdown(
        "This view surfaces **which products tend to be bought together** "
        "to guide cross-sell and bundle design."
    )

    top_products = (
        df[prod_col].value_counts()
        .reset_index()
        .rename(columns={prod_col: "count", "index": prod_col})
    )
    st.subheader("Top products")
    st.dataframe(top_products, use_container_width=True)

    fig = px.bar(
        top_products,
        x=prod_col,
        y="count",
        title="Most common products",
        text="count",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def section_segmentation(df: pd.DataFrame):
    st.header("Customer Segmentation (Clustering)")

    if df is None:
        st.warning("Upload a CSV to run clustering.")
        return

    # Try a simple numeric-only clustering
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num_df.shape[1] < 2:
        st.info(
            "This section needs at least two numeric columns (e.g., CLV, tenure, "
            "number of products) to run clustering."
        )
        return

    st.markdown(
        "This view creates **data-driven customer segments** based on numeric "
        "features such as value, tenure, and usage."
    )

    n_clusters = st.slider("Number of clusters (k)", 3, 8, 4)
    scaler = (num_df - num_df.mean()) / num_df.std(ddof=0)
    scaler = scaler.fillna(0)

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(scaler)

    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(scaler)
    seg_df = pd.DataFrame(emb, columns=["pc1", "pc2"])
    seg_df["cluster"] = labels.astype(str)

    fig = px.scatter(
        seg_df,
        x="pc1",
        y="pc2",
        color="cluster",
        title="Customer segments (PCA projection)",
        opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster sizes")
    st.dataframe(seg_df["cluster"].value_counts().rename("count"), use_container_width=True)


# -------------------------------------------------------------------
# 5. ROUTE TO SECTION
# -------------------------------------------------------------------
if section == "KPIs overview":
    section_kpis_overview(df)
elif section == "Why People Churn":
    section_why_churn(df)
elif section == "Why People Convert":
    section_why_convert(df)
elif section == "Why People Engage":
    section_why_engage(df)
elif section == "Time Series Analysis":
    section_timeseries(df)
elif section == "Sentiment Analysis":
    section_sentiment(df)
elif section == "Predictive Analytics":
    section_predictive(df)
elif section == "Product Recommendations":
    section_product_reco(df)
elif section == "Customer Segmentation":
    section_segmentation(df)

st.markdown(
    "<hr style='border:1px solid #eee;'>"
    "<div style='text-align:center;color:gray;font-size:13px;'>"
    "© 2025 Howard Nguyen, PhD – TopKPI2 • AI-powered growth, retention & profitability"
    "</div>",
    unsafe_allow_html=True,
)
