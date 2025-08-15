import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

st.set_page_config(page_title="AI Ethics Dashboard", page_icon="🤖📚", layout="wide")


@st.cache_data
def load_csv(file_like):
    return pd.read_csv(file_like)


def normalize_cols(df):
    cols = [
        "id",
        "title",
        "year",
        "country",
        "sector",
        "harm_type",
        "risk_severity",
        "stakeholders",
        "summary",
        "source",
        "organization",
        "what_went_wrong",
        "prevention",
        "topic",
    ]
    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()
    if "year" in df:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ["sector", "harm_type", "risk_severity", "country"]:
        if col in df:
            df[col] = df[col].astype(str).str.strip()
    if "stakeholders" in df:
        df["stakeholders"] = df["stakeholders"].fillna("").astype(str)
    return df


st.title("AI Ethics Dashboard")
st.caption(
    "Filter and visualize AI incident patterns for quick risk awareness and discussion."
)

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader(
        "Upload incidents CSV",
        type=["csv"],
        help="Headers must match the README schema.",
    )
    if uploaded is None:
        df = load_csv("data/incidents.csv")
        st.info("Using sample dataset. Upload a CSV to replace it for this session.")
    else:
        df = load_csv(uploaded)
        st.success("Custom dataset loaded.")

    df = normalize_cols(df)

    st.divider()
    st.header("Filters")
    years = (
        sorted([int(y) for y in df["year"].dropna().unique()]) if "year" in df else []
    )
    year_sel = st.multiselect("Year", options=years, default=years)
    sector_sel = st.multiselect(
        "Sector",
        options=sorted(df.get("sector", pd.Series(dtype=str)).dropna().unique()),
        default=None,
    )
    harm_sel = st.multiselect(
        "Harm type",
        options=sorted(df.get("harm_type", pd.Series(dtype=str)).dropna().unique()),
        default=None,
    )
    risk_sel = st.multiselect(
        "Risk severity", options=["Low", "Medium", "High", "Critical"], default=None
    )
    country_sel = st.multiselect(
        "Country",
        options=sorted(df.get("country", pd.Series(dtype=str)).dropna().unique()),
        default=None,
    )

mask = pd.Series([True] * len(df))
if "year" in df and year_sel:
    mask &= df["year"].isin(year_sel)
if sector_sel:
    mask &= df["sector"].isin(sector_sel)
if harm_sel:
    mask &= df["harm_type"].isin(harm_sel)
if risk_sel:
    mask &= df["risk_severity"].isin(risk_sel)
if country_sel:
    mask &= df["country"].isin(country_sel)

fdf = df[mask].copy()

left, mid, right = st.columns(3)
with left:
    st.metric("Incidents", len(fdf))
with mid:
    top_sector = (
        fdf["sector"].mode().iat[0] if "sector" in fdf and not fdf.empty else "—"
    )
    st.metric("Most common sector", top_sector)
with right:
    top_harm = (
        fdf["harm_type"].mode().iat[0] if "harm_type" in fdf and not fdf.empty else "—"
    )
    st.metric("Top harm type", top_harm)

st.divider()

colA, colB = st.columns(2)
with colA:
    if "harm_type" in fdf and not fdf.empty:
        harm_counts = fdf["harm_type"].value_counts().reset_index()
        harm_counts.columns = ["harm_type", "count"]
        fig = px.bar(
            harm_counts, x="harm_type", y="count", title="Incidents by harm type"
        )
        st.plotly_chart(fig, use_container_width=True)
with colB:
    if "sector" in fdf and not fdf.empty:
        sec_counts = fdf["sector"].value_counts().reset_index()
        sec_counts.columns = ["sector", "count"]
        fig2 = px.bar(sec_counts, x="sector", y="count", title="Incidents by sector")
        st.plotly_chart(fig2, use_container_width=True)

if "organization" in fdf and not fdf["organization"].fillna("").eq("").all():
    st.subheader("Incidents by organization")
    org_counts = (
        fdf["organization"].replace("", pd.NA).dropna().value_counts().reset_index()
    )
    org_counts.columns = ["organization", "count"]
    fig3 = px.bar(
        org_counts, x="organization", y="count", title="Incidents by organization"
    )
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Incident details")
st.dataframe(fdf, use_container_width=True)

st.subheader("Readable cards")
for _, row in fdf.iterrows():
    with st.expander(
        f"{row.get('title','(untitled)')} — {row.get('year','')}", expanded=False
    ):
        st.write(
            f"**Sector:** {row.get('sector','')}  |  **Harm:** {row.get('harm_type','')}  |  **Risk:** {row.get('risk_severity','')}"
        )
        st.write(
            f"**Country:** {row.get('country','')}  |  **Stakeholders:** {row.get('stakeholders','')}"
        )
        st.write(row.get("summary", ""))
        if row.get("what_went_wrong"):
            st.write(f"**What went wrong:** {row.get('what_went_wrong', '')}")
        if row.get("prevention"):
            st.write(f"**How to prevent:** {row.get('prevention', '')}")
        if str(row.get("source", "")).startswith("http"):
            st.markdown(f"[Source link]({row['source']})")

st.divider()
st.download_button(
    label="Download filtered CSV",
    data=fdf.to_csv(index=False).encode("utf-8"),
    file_name="filtered_incidents.csv",
    mime="text/csv",
)

st.subheader("Add a quick incident (temporary)")
with st.form("add_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        title = st.text_input("Title")
        year = st.number_input(
            "Year", min_value=1900, max_value=2100, value=2025, step=1
        )
    with c2:
        sector = st.text_input("Sector")
        harm_type = st.text_input("Harm type")
    with c3:
        risk = st.selectbox("Risk severity", ["Low", "Medium", "High", "Critical"])
    stakeholders = st.text_input("Stakeholders (semicolon separated)")
    country = st.text_input("Country")
    summary = st.text_area("Summary", height=120)
    source = st.text_input("Source URL")
    submitted = st.form_submit_button("Add to view")
    if submitted:
        new = {
            "id": f"tmp-{pd.Timestamp.now().value}",
            "title": title,
            "year": int(year),
            "country": country,
            "sector": sector,
            "harm_type": harm_type,
            "risk_severity": risk,
            "stakeholders": stakeholders,
            "summary": summary,
            "source": source,
        }
        st.session_state.setdefault("new_rows", [])
        st.session_state["new_rows"].append(new)
        st.success("Added to current session. Use the download button to export.")
        fdf.loc[len(fdf)] = new
