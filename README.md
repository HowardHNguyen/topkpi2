# TopKPI2

Advanced Marketing Growth & Retention Intelligence: a Streamlit dashboard plus
analysis notebooks (KPIs, churn, conversion, engagement, time series, sentiment,
predictive analytics, recommendations, segmentation).

## Run the app

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app loads `data.csv` (auto-insurance customer data) by default, or any CSV
uploaded in the sidebar. Segmentation and Recommendations need the UCI Online
Retail dataset. `Churn` and `EngagementScore` are optional columns; churn views
are unavailable without `Churn`.

## Notebooks

```bash
pip install -r requirements-notebooks.txt
```

Some notebooks read files that are not in this repo (`Tweets.csv`,
`churn-data.csv`, `engage-data.csv`, `data/convert-data.csv`); add them locally to run those.
