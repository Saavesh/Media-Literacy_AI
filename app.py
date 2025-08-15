import re
import requests
import tldextract
import pandas as pd
import plotly.express as px
import streamlit as st
from bs4 import BeautifulSoup
from readability import Document
import textstat
from textblob import TextBlob
from typing import Optional

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="AI Ethics Dashboard", page_icon="🤖📰", layout="wide")

# -----------------------
# Welcome & Overview
# -----------------------
st.markdown("# Media Literacy Lab")
st.caption(
    "A bridge between real AI incidents and how news covers them. Analyze a news article to spot bias/framing cues, "
    "then compare with real-world AI incidents for context."
)
st.info(
    "**How this dashboard works:**\n"
    "1) **Analyze news coverage** – Paste a URL below to see language and framing signals.\n"
    "2) **Explore real AI incidents** – Scroll down to browse a sample dataset of real AI-related events.\n"
    "3) **Compare** – After an analysis, you'll see **similar incidents** suggested from the sample database."
)
st.markdown("---")


# ===============
# Style helper for compact charts
# ===============
def _compact(fig, title, h=260):
    fig.update_layout(title=title, height=h, margin=dict(l=10, r=10, t=45, b=10))
    fig.update_xaxes(tickangle=-25)
    return fig


# ===============
# URL Analyzer – fetch & extract
# ===============
@st.cache_data(show_spinner=False, ttl=600)
def fetch_and_extract(url: str):
    headers = {"User-Agent": "Mozilla/5.0 (educational noncommercial tool)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    html = r.text

    doc = Document(html)
    clean_html = doc.summary(html_partial=True)
    title = (doc.short_title() or "").strip()

    soup = BeautifulSoup(clean_html, "lxml")
    paragraphs = [
        p.get_text(" ", strip=True) for p in soup.find_all(["p", "li", "blockquote"])
    ]
    text = "\n".join([p for p in paragraphs if p])

    full = BeautifulSoup(html, "lxml")
    links = [a.get("href") for a in full.find_all("a", href=True)]
    return {"title": title, "text": text.strip(), "links": links, "html_len": len(html)}


# ===============
# (Later) External API Spot
# ===============
# add an API (such as news classifier/toxicity/stance detection) here later.
# Keep the function signature; just merge its outputs into `report`.
#
# def call_external_ai_api(text: str) -> dict:
#     """
#     Placeholder for a future API.
#     Return dict like: {"toxicity": 0.12, "stance": "pro", "hallucination_risk": 0.3, ...}
#     """
#     implement
#     return {}
# ===============


# ========
# Vocabulary Patterns
# ===============
HEDGE_WORDS = (
    r"\b(may|might|could|possibly|reportedly|allegedly|suggests?|appears?|claims?)\b"
)
CLICKBAIT = r"\b(shocking|won't believe|this is why|you need to|exposed|revealed|the truth about)\b"
LOADED_WORDS = r"\b(disaster|outrage|corrupt|fraud|heroic|evil|traitor|hoax)\b"


# ===============
# Text analysis (quality, cues, linking)
#  ===============
def analyze_text(text: str, links: list, url: str):
    words = re.findall(r"\w+", (text or "").lower())
    word_count = len(words)
    sentence_count = max(
        1, (text or "").count(".") + (text or "").count("!") + (text or "").count("?")
    )

    fk = textstat.flesch_kincaid_grade(text) if text else 0.0
    fog = textstat.gunning_fog(text) if text else 0.0

    blob = TextBlob((text or "")[:20000])
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)

    hedges = len(re.findall(HEDGE_WORDS, text or "", flags=re.I))
    clickbait_hits = len(re.findall(CLICKBAIT, text or "", flags=re.I))
    loaded_hits = len(re.findall(LOADED_WORDS, text or "", flags=re.I))
    quotes = len(re.findall(r"“[^”]+”|\"[^\"]+\"", text or ""))

    article_dom = tldextract.extract(url).registered_domain if url else ""
    out_domains = []
    for href in links or []:
        try:
            d = tldextract.extract(href).registered_domain
            if d and d != article_dom:
                out_domains.append(d)
        except Exception:
            pass
    unique_sources = sorted(set(out_domains))

    # For later: external API later, merge its metrics here into the dict returned
    # api_metrics = call_external_ai_api(text)  # <- but later
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "flesch_kincaid_grade": round(fk, 1) if fk else 0.0,
        "gunning_fog": round(fog, 1) if fog else 0.0,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "hedge_count": hedges,
        "clickbait_terms": clickbait_hits,
        "loaded_terms": loaded_hits,
        "quote_count": quotes,
        "unique_source_count": len(unique_sources),
        "unique_sources": unique_sources[:12],
        # ** when API is done  include: ** | **api_metrics**
    }


# ===============
# Keyword tagging for “Similar incidents”
# ===============
SECTOR_KEYWORDS = {
    "Healthcare": [
        "hospital",
        "clinic",
        "medical",
        "patient",
        "diagnosis",
        "biomedical",
        "ehr",
    ],
    "Finance": [
        "bank",
        "loan",
        "credit",
        "trading",
        "fraud detection",
        "fintech",
        "insurance",
    ],
    "Government": [
        "police",
        "court",
        "public sector",
        "welfare",
        "agency",
        "surveillance",
    ],
    "Education": [
        "school",
        "university",
        "students",
        "grading",
        "admissions",
        "campus",
    ],
    "Employment": ["hiring", "recruiting", "hr", "resume", "screening", "workplace"],
    "Social media": ["platform", "content moderation", "viral", "influencer", "feed"],
    "Transportation": [
        "autonomous",
        "driverless",
        "vehicle",
        "traffic",
        "routing",
        "ride-hailing",
    ],
    "Retail": ["ecommerce", "recommendation", "shopping", "customer", "ads"],
}

HARM_KEYWORDS = {
    "Bias/Discrimination": [
        "bias",
        "discrimination",
        "fairness",
        "inequitable",
        "racial",
        "gender",
    ],
    "Privacy/Security": [
        "privacy",
        "breach",
        "leak",
        "surveillance",
        "tracking",
        "doxx",
    ],
    "Safety/Reliability": [
        "harm",
        "unsafe",
        "accident",
        "failure",
        "malfunction",
        "bug",
    ],
    "Misinformation/Manipulation": [
        "misinformation",
        "disinformation",
        "fake",
        "deepfake",
        "propaganda",
    ],
    "Fraud/Deception": ["fraud", "scam", "deception", "impersonation", "phishing"],
}


def extract_tags(text: str):
    t = (text or "").lower()
    sector_hits, harm_hits = [], []
    for sector, kws in SECTOR_KEYWORDS.items():
        if any(kw in t for kw in kws):
            sector_hits.append(sector)
    for harm, kws in HARM_KEYWORDS.items():
        if any(kw in t for kw in kws):
            harm_hits.append(harm)
    return {"sectors": sorted(set(sector_hits)), "harms": sorted(set(harm_hits))}


def rank_similar_incidents(df: pd.DataFrame, text: str, top_n: int = 5):
    if df is None or df.empty:
        return df

    tags = extract_tags(text)
    t = (text or "").lower()

    def score_row(row):
        s = 0
        blob = f"{row.get('title','')} {row.get('summary','')} {row.get('what_went_wrong','')}".lower()
        for _, kws in SECTOR_KEYWORDS.items():
            s += sum(1 for kw in kws if kw in t and kw in blob)
        for _, kws in HARM_KEYWORDS.items():
            s += sum(1 for kw in kws if kw in t and kw in blob)
        if tags["sectors"] and str(row.get("sector", "")) in tags["sectors"]:
            s += 2
        if tags["harms"] and str(row.get("harm_type", "")) in tags["harms"]:
            s += 2
        return s

    scored = df.copy()
    scored["_sim_score"] = scored.apply(score_row, axis=1)
    scored = scored.sort_values("_sim_score", ascending=False)
    scored = scored[scored["_sim_score"] > 0]
    return scored.head(top_n).drop(columns=["_sim_score"], errors="ignore")


# ===============
# Render: article & stats & checklist
# ===============
def render_article_full(
    title: str, text: str, report: dict, df_for_compare: Optional[pd.DataFrame] = None
):
    st.subheader(title or "Untitled article")
    left, right = st.columns([2, 1])

    with left:
        st.markdown("### Article text (first ~2000 characters)")
        if text:
            preview = text[:2000] + ("..." if len(text) > 2000 else "")
            st.write(preview)
        else:
            st.write("_No text extracted_")

    with right:
        st.markdown("### Quick stats")
        st.metric("Words", report["word_count"])
        st.metric("Sentences", report["sentence_count"])
        st.metric("Reading grade (FK)", report["flesch_kincaid_grade"])
        st.metric("Gunning Fog", report["gunning_fog"])
        st.metric("Sentiment polarity", report["polarity"])
        st.metric("Subjectivity", report["subjectivity"])
        st.metric("Quotes in text", report["quote_count"])
        st.metric("Hedging terms", report["hedge_count"])
        st.metric("Clickbait cues", report["clickbait_terms"])
        st.metric("Loaded terms", report["loaded_terms"])
        st.metric("External sources", report["unique_source_count"])
        if report["unique_sources"]:
            st.caption("Sources referenced:")
            st.write(", ".join(report["unique_sources"]))

    st.markdown("---")
    st.markdown("### Media literacy checklist (auto-filled hints)")

    hints = []
    if report["subjectivity"] > 0.55:
        hints.append(
            "The language in this article is quite subjective. Pay attention to where it may be presenting opinions rather than objective facts."
        )
    if report["hedge_count"] > 5:
        hints.append(
            "This article uses many hedging words. These can indicate uncertainty or a lack of firm evidence."
        )
    if report["clickbait_terms"] > 0:
        hints.append(
            "Some phrases appear designed to grab attention. Consider whether the content matches the dramatic tone."
        )
    if report["loaded_terms"] > 3:
        hints.append(
            "The language includes emotionally charged words. This can influence how readers feel about the topic."
        )
    if report["unique_source_count"] < 2:
        hints.append(
            "Only a few outside sources are cited. Look for additional perspectives to get a fuller picture."
        )
    if report["flesch_kincaid_grade"] > 14:
        hints.append(
            "The reading level is quite high. Complex language can make claims harder to evaluate."
        )

    if not hints:
        st.success(
            "No major concerns detected in these basic checks, but it’s still important to read critically."
        )
    else:
        for h in hints:
            st.warning("• " + h)

    # Similar incidents from sample dataset
    if df_for_compare is not None and not df_for_compare.empty and text:
        st.markdown("---")
        st.markdown("### Similar real incidents (from the sample AI incident database)")
        matches = rank_similar_incidents(df_for_compare, text, top_n=5)
        if matches is not None and not matches.empty:
            for _, row in matches.iterrows():
                with st.expander(
                    f"{row.get('title','(untitled)')} – {row.get('year','')}",
                    expanded=False,
                ):
                    st.write(
                        f"**Sector:** {row.get('sector','')}  |  **Harm:** {row.get('harm_type','')}  |  **Risk:** {row.get('risk_severity','')}"
                    )
                    st.write(row.get("summary", ""))
                    if str(row.get("source", "")).startswith("http"):
                        st.markdown(f"[Source link]({row['source']})")
        else:
            st.caption("_No close matches found in the sample dataset._")

    st.markdown("---")


# ===============
# Dataset utilities (load/normalize)
#  ===============
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


# Try to load bundled sample so “Similar incidents” works before any upload
try:
    df_sample = normalize_cols(load_csv("data/incidents.csv"))
except Exception:
    df_sample = pd.DataFrame()


# ===============
# URL Input (primary)
#  ===============
st.markdown(
    """
<style>
div[data-testid="stTextInput"] input { font-size: 1.05rem; padding: 0.9rem 1rem; }
div[data-testid="stFormSubmitButton"] button { padding: 0.6rem 1.1rem; font-size: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.subheader("URL Analyzer")
st.caption(
    "Paste any news article URL to scan for potential bias and misinformation signals."
)

with st.form("url_form"):
    url = st.text_input(
        "Paste a news article URL",
        placeholder="https://example.com/news/story",
        key="url_input",
    ).strip()
    submitted = st.form_submit_button("Analyze")

st.info(
    "The URL Analyzer evaluates the single article you submit. It does **not** change the incident dataset below."
)

if submitted and url:
    if not (url.startswith("http://") or url.startswith("https://")):
        st.warning("Please enter a valid http(s) URL.")
        st.stop()
    with st.spinner("Fetching and analyzing..."):
        try:
            data = fetch_and_extract(url)
            if not data["text"].strip():
                st.warning(
                    "I couldn’t extract readable text from that page. Try another link or paste the article text below."
                )
                st.stop()
            report = analyze_text(data["text"], data["links"], url)
        except Exception as e:
            st.error(f"Could not fetch or analyze that URL: {e}")
            st.stop()

    # Use df_sample for matching so it works even before any upload
    render_article_full(
        data["title"] or "Untitled article",
        data["text"],
        report,
        df_for_compare=df_sample,
    )


# ===============
# AI Incident Database (Sample)
# ===============
st.header("AI Incident Database (Sample)")
st.caption(
    "Browse real-world AI-related incidents to compare patterns with the article you analyzed above."
)

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader(
        "Upload incidents CSV",
        type=["csv"],
        help="Headers must match the README schema.",
    )
    if uploaded is None:
        df = df_sample.copy()
        st.info("Using bundled sample file: `data/incidents.csv`")
    else:
        df = normalize_cols(load_csv(uploaded))
        st.success(f"Custom dataset loaded: {uploaded.name}")

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

dataset_source_msg = (
    f"Source: your uploaded file **{uploaded.name}**."
    if "uploaded" in locals() and uploaded is not None
    else "Source: bundled sample file **data/incidents.csv**."
)
st.info(
    "This section is a reference set of real incidents to give context while learning media literacy. "
    "**It’s separate from the URL Analyzer** — analyzing a URL does not modify this dataset.  "
    f"_{dataset_source_msg}_"
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
        fdf["sector"].mode().iat[0] if "sector" in fdf and not fdf.empty else "-"
    )
    st.metric("Most common sector", top_sector)
with right:
    top_harm = (
        fdf["harm_type"].mode().iat[0] if "harm_type" in fdf and not fdf.empty else "-"
    )
    st.metric("Top harm type", top_harm)

st.divider()

st.subheader("Incident details")
st.dataframe(fdf, use_container_width=True)

st.subheader("Readable cards")
for _, row in fdf.iterrows():
    with st.expander(
        f"{row.get('title','(untitled)')} - {row.get('year','')}", expanded=False
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


# ===============
#  Charts (might take out at the end)
#  ===============
with st.expander("Explore charts (optional)", expanded=False):
    colA, colB = st.columns(2)

    with colA:
        if "harm_type" in fdf and not fdf.empty:
            harm_counts = fdf["harm_type"].value_counts().reset_index()
            harm_counts.columns = ["harm_type", "count"]
            fig = px.bar(harm_counts, x="harm_type", y="count")
            st.plotly_chart(
                _compact(fig, "Incidents by harm type"), use_container_width=True
            )

    with colB:
        if "sector" in fdf and not fdf.empty:
            sec_counts = fdf["sector"].value_counts().reset_index()
            sec_counts.columns = ["sector", "count"]
            fig2 = px.bar(sec_counts, x="sector", y="count")
            st.plotly_chart(
                _compact(fig2, "Incidents by sector"), use_container_width=True
            )

    if "organization" in fdf and not fdf["organization"].fillna("").eq("").all():
        org_counts = (
            fdf["organization"].replace("", pd.NA).dropna().value_counts().reset_index()
        )
        org_counts.columns = ["organization", "count"]
        fig3 = px.bar(org_counts, x="organization", y="count")
        st.plotly_chart(
            _compact(fig3, "Incidents by organization"), use_container_width=True
        )
