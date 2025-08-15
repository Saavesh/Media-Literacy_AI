import re
import requests
import tldextract
import streamlit as st
from newsapi import NewsApiClient
from bs4 import BeautifulSoup
from readability import Document
import textstat
from textblob import TextBlob


# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Media Bias & AI Incident Dashboard", page_icon="🤖📰", layout="wide"
)
st.markdown(
    """
<style>
/* Make primary buttons stand out more */
.stButton > button[kind="primary"] { padding: 0.7rem 1rem; font-weight: 600; }
/* Clean up link spacing */
.block-container a { text-decoration: none; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------
# Title & Overview
# -----------------------
st.title("Media Bias & AI Incident Dashboard")
st.caption(
    "A bridge between real AI incidents and how news covers them. Analyze a news article to spot bias and framing cues, "
    "then compare headlines from live news sources for context."
)
st.info(
    "**How this dashboard works:**\n"
    "1) **Analyze news coverage**. Paste a URL below to see language and framing signals.\n"
    "2) **Browse live headlines**. Use the search or the readable cards to explore current stories.\n"
    "3) **Compare**. After an analysis, use the signals to think critically about patterns across stories."
)
st.markdown("---")


# -----------------------
# URL Analysis Queue helper
# -----------------------
def _queue_for_analysis(url: str):
    st.session_state["url_to_analyze"] = url


# -----------------------
# News API Key - Load and Initialize
# -----------------------
NEWSAPI_KEY = st.secrets["api_keys"]["newsapi"]
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)


# ======================
# Core helpers & analysis
# ======================
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


# ---------- helpers for evidence snippets ----------
def _snippets(
    text: str, pattern: str, *, pad_chars: int = 70, max_items: int = 12, flags=re.I
):
    """Return short context snippets around regex matches for display."""
    if not text:
        return []
    seen, out = set(), []
    for m in re.finditer(pattern, text, flags):
        s = max(0, m.start() - pad_chars)
        e = min(len(text), m.end() + pad_chars)
        snippet = re.sub(r"\s+", " ", text[s:e].strip())
        if snippet not in seen:
            seen.add(snippet)
            out.append(snippet)
        if len(out) >= max_items:
            break
    return out


def _unique_domains(links, article_url):
    base = (
        tldextract.extract(article_url).top_domain_under_public_suffix
        if article_url
        else ""
    )
    if not base:
        return []
    out, seen = [], set()
    for href in links or []:
        try:
            d = tldextract.extract(href).top_domain_under_public_suffix
            if d and d != base and d not in seen:
                seen.add(d)
                out.append(d)
        except Exception:
            pass
    return out


# ---------- vocabulary patterns ----------
HEDGE_WORDS = (
    r"\b(may|might|could|possibly|reportedly|allegedly|suggests?|appears?|claims?)\b"
)
CLICKBAIT = r"\b(shocking|won'?t believe|this is why|you need to|exposed|revealed|the truth about)\b"
LOADED_WORDS = r"\b(disaster|outrage|corrupt|fraud|heroic|evil|traitor|hoax)\b"


def analyze_text(text: str, links: list, url: str):
    # basic counts
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

    # evidence-based signals
    hedge_snips = _snippets(text or "", HEDGE_WORDS)
    clickbait_snips = _snippets(text or "", CLICKBAIT)
    loaded_snips = _snippets(text or "", LOADED_WORDS)
    quotes = len(re.findall(r"“[^”]+”|\"[^\"]+\"", text or ""))

    # title/lead sentence checks
    first_line = (text or "").strip().split("\n", 1)[0]
    title_clickbait_snips = _snippets(first_line, CLICKBAIT)

    # style markers
    excess_punct_matches = list(re.finditer(r"[!?]{2,}", text or ""))
    excess_punct = len(excess_punct_matches)
    punct_snips = _snippets(text or "", r"[!?]{2,}", pad_chars=35)

    caps_words = [
        w
        for w in re.findall(r"\b[A-Z]{4,}\b", text or "")
        if w not in {"COVID", "USA", "NPR", "BBC"}
    ]
    caps_examples = sorted(set(caps_words))[:10]

    # passive voice and sourcing
    passive_snips = _snippets(
        text or "", r"\b(?:was|were|is|are|been|being)\s+\w+ed\b(?:\s+by\b)?"
    )
    according_snips = _snippets(text or "", r"\baccording to\b")

    # outgoing domains
    unique_sources = _unique_domains(links, url)

    return {
        # basics
        "word_count": word_count,
        "sentence_count": sentence_count,
        "flesch_kincaid_grade": round(fk, 1) if fk else 0.0,
        "gunning_fog": round(fog, 1) if fog else 0.0,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "quote_count": quotes,
        # counts
        "hedge_count": len(hedge_snips),
        "clickbait_terms": len(clickbait_snips),
        "loaded_terms": len(loaded_snips),
        "title_clickbait": len(title_clickbait_snips),
        "excess_punct": excess_punct,
        "all_caps_words": len(caps_words),
        "passive_hits": len(passive_snips),
        "according_to": len(according_snips),
        # evidence
        "hedge_examples": hedge_snips[:6],
        "clickbait_examples": clickbait_snips[:6] or title_clickbait_snips[:3],
        "loaded_examples": loaded_snips[:6],
        "punct_examples": punct_snips[:6],
        "caps_examples": caps_examples,
        "passive_examples": passive_snips[:6],
        "according_examples": according_snips[:6],
        # linking
        "unique_source_count": len(unique_sources),
        "unique_sources": unique_sources[:12],
    }


# ---------- light tagging for the "readable cards" ----------
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


def render_article_full(title: str, text: str, report: dict):
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
        st.metric("Words", report.get("word_count", 0))
        st.metric("Sentences", report.get("sentence_count", 0))
        st.metric("Reading grade (FK)", report.get("flesch_kincaid_grade", 0.0))
        st.metric("Gunning Fog", report.get("gunning_fog", 0.0))
        st.metric("Sentiment polarity", report.get("polarity", 0.0))
        st.metric("Subjectivity", report.get("subjectivity", 0.0))
        st.metric("Quotes in text", report.get("quote_count", 0))
        st.metric("Hedging terms", report.get("hedge_count", 0))
        st.metric("Clickbait cues", report.get("clickbait_terms", 0))
        st.metric("Loaded terms", report.get("loaded_terms", 0))
        st.metric("External sources", report.get("unique_source_count", 0))
        if report.get("unique_sources"):
            st.caption("Sources referenced:")
            st.write(", ".join(report["unique_sources"]))

    st.markdown("---")
    st.markdown("### Media literacy checklist (auto-filled hints)")

    # Subjectivity
    if report.get("subjectivity", 0) >= 0.5:
        st.warning(
            "• The tone is subjective. Check whether opinions are clearly labeled."
        )

    # Hedging
    if report.get("hedge_count", 0) >= 4:
        st.warning("• Many hedging terms suggest uncertainty.")
        if report.get("hedge_examples"):
            st.caption("Examples:")
            for s in report["hedge_examples"]:
                st.write(f"– {s}")

    # Clickbait
    if report.get("title_clickbait", 0) > 0 or report.get("clickbait_terms", 0) > 0:
        st.warning("• Headline or body uses attention grabbing phrases.")
        if report.get("clickbait_examples"):
            st.caption("Examples:")
            for s in report["clickbait_examples"]:
                st.write(f"– {s}")

    # Loaded language / emphasis
    if (
        report.get("loaded_terms", 0) >= 3
        or report.get("excess_punct", 0) > 0
        or report.get("all_caps_words", 0) >= 3
    ):
        st.warning(
            "• Emotionally loaded wording or emphasis markers can steer interpretation."
        )
        if report.get("loaded_examples"):
            st.caption("Examples:")
            for s in report["loaded_examples"]:
                st.write(f"– {s}")
        if report.get("punct_examples"):
            st.caption("Repeated punctuation:")
            for s in report["punct_examples"]:
                st.write(f"– {s}")
        if report.get("caps_examples"):
            st.caption("All-caps words:")
            st.write(", ".join(report["caps_examples"]))

    # Passive voice
    if report.get("passive_hits", 0) >= 5:
        st.warning(
            "• Frequent passive voice can obscure responsibility. Look for named actors."
        )
        if report.get("passive_examples"):
            st.caption("Examples:")
            for s in report["passive_examples"]:
                st.write(f"– {s}")

    # Sourcing
    if report.get("according_to", 0) < 1 and report.get("unique_source_count", 0) < 2:
        st.warning(
            "• Few attributions or outside sources. Seek independent corroboration."
        )
        if report.get("according_examples"):
            st.caption("Mentions of 'according to':")
            for s in report["according_examples"]:
                st.write(f"– {s}")

    # Reading level
    if report.get("flesch_kincaid_grade", 0) >= 14:
        st.warning(
            "• Very high reading level. Dense language can make claims harder to check quickly."
        )

    st.markdown("---")


# ======================
# Interactive News Search
# ======================
st.subheader("Search news and analyze")

q = st.text_input("Keyword(s)", value="artificial intelligence")
mode = st.radio("Source", ["Top headlines", "Everything"], horizontal=True)

if mode == "Top headlines":
    c1, c2 = st.columns([1, 2])
    with c1:
        country = st.selectbox("Country", ["us", "gb", "ca", "au"], index=0)
    with c2:
        cats = st.multiselect(
            "Categories",
            ["technology", "science", "business", "health", "entertainment", "sports"],
            default=["technology", "science"],
        )

    if st.button("Search headlines"):
        shown = set()
        if not cats:
            data = newsapi.get_top_headlines(q=q, country=country, page_size=8)
            arts = data.get("articles", []) or []
            if arts:
                st.markdown(f"**Top headlines for** `{q}`")
                for i, a in enumerate(arts):
                    title = a.get("title") or "(no title)"
                    url = a.get("url") or "#"
                    source = (a.get("source") or {}).get("name") or "Unknown"
                    if title in shown:
                        continue
                    shown.add(title)
                    with st.container():
                        st.markdown(f"**[{title}]({url})**")
                        st.caption(source)
                        st.button(
                            "Analyze this article",
                            key=f"analyze_headlines_{i}",
                            on_click=_queue_for_analysis,
                            args=(url,),
                            type="primary",
                            use_container_width=True,
                        )
            else:
                st.write("No results.")
        else:
            for cat in cats:
                data = newsapi.get_top_headlines(
                    q=q, country=country, category=cat, page_size=5
                )
                arts = data.get("articles", []) or []
                if arts:
                    st.markdown(f"**{cat.capitalize()}**")
                    for i, a in enumerate(arts):
                        title = a.get("title") or "(no title)"
                        url = a.get("url") or "#"
                        source = (a.get("source") or {}).get("name") or "Unknown"
                        if title in shown:
                            continue
                        shown.add(title)
                        with st.container():
                            st.markdown(f"**[{title}]({url})**")
                            st.caption(source)
                            st.button(
                                "Analyze this article",
                                key=f"analyze_{cat}_{i}",
                                on_click=_queue_for_analysis,
                                args=(url,),
                            )
else:
    c1, c2 = st.columns([1, 1])
    with c1:
        lang = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
    with c2:
        sort_by = st.selectbox(
            "Sort by", ["relevancy", "popularity", "publishedAt"], index=0
        )

    if st.button("Search articles"):
        data = newsapi.get_everything(q=q, language=lang, sort_by=sort_by, page_size=10)
        arts = data.get("articles", []) or []
        if arts:
            st.markdown(f"**Articles for** `{q}`")
            seen = set()
            for i, a in enumerate(arts):
                title = a.get("title") or "(no title)"
                url = a.get("url") or "#"
                source = (a.get("source") or {}).get("name") or "Unknown"
                if title in seen:
                    continue
                seen.add(title)
                with st.container():
                    st.markdown(f"**[{title}]({url})**")
                    st.caption(source)
                    st.button(
                        "Analyze this article",
                        key=f"analyze_everything_{i}",
                        on_click=_queue_for_analysis,
                        args=(url,),
                    )
        else:
            st.write("No results.")


### make button bigger


# -----------------------
# If a URL was queued from the search, analyze it now
# -----------------------
queued = st.session_state.pop("url_to_analyze", None)
if queued:
    st.markdown("---")
    st.subheader("Selected article analysis")
    st.caption(f"Source: {queued}")

    with st.spinner("Fetching and analyzing the selected article..."):
        try:
            data = fetch_and_extract(queued)
            if not data["text"].strip():
                st.warning(
                    "I could not extract readable text from that page. Try another link."
                )
            else:
                report = analyze_text(data["text"], data["links"], queued)
                render_article_full(
                    data["title"] or "Untitled article",
                    data["text"],
                    report,
                )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                st.error(
                    "This site blocks automated extraction. Try a different article (Reuters, AP, BBC, NPR, etc.)."
                )
            else:
                st.error(f"Could not fetch or analyze that URL: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Could not fetch or analyze that URL: {e}")
            st.stop()


# -----------------------
# Readable cards (live from NewsAPI)
# -----------------------
def _risk_from_harm(harm_label: str) -> str:
    if "Misinformation" in harm_label:
        return "Medium"
    if "Privacy" in harm_label or "Fraud" in harm_label:
        return "High"
    if "Safety" in harm_label:
        return "High"
    if "Bias" in harm_label:
        return "Medium"
    return "-"


def _prevention_tip(harm_label: str) -> str:
    if "Misinformation" in harm_label:
        return "Cross-check multiple outlets, link to primary sources, and disclose uncertainty clearly."
    if "Privacy" in harm_label:
        return "Minimize data collection, get consent, and avoid sharing PII unless necessary."
    if "Bias" in harm_label:
        return "Audit datasets, document limitations, and include counter examples in evaluation."
    if "Safety" in harm_label:
        return "Test failure cases, document known hazards, and provide user warnings."
    if "Fraud" in harm_label:
        return "Use verification steps, stronger identity checks, and rate limit risky actions."
    return ""


st.subheader("Top AI stories with quick bias scan")
st.caption(
    "Live feed from NewsAPI. Click any story to open it or run the analyzer for a deeper scan."
)

with st.spinner("Loading top AI stories…"):
    feed = newsapi.get_everything(
        q="artificial intelligence", language="en", sort_by="publishedAt", page_size=5
    )
    articles = feed.get("articles", []) or []

if not articles:
    st.caption("No recent AI stories found.")
else:
    for i, a in enumerate(articles):
        title = a.get("title") or "(untitled)"
        url = a.get("url") or "#"
        source_name = (a.get("source") or {}).get("name") or "Unknown source"
        published = a.get("publishedAt") or ""
        year = published[:4] if published else ""
        desc = a.get("description") or ""

        blob = f"{title} {desc}".strip().lower()
        tags = extract_tags(blob)
        sector = ", ".join(tags["sectors"]) if tags["sectors"] else "News/Media"
        harm = (
            ", ".join(tags["harms"]) if tags["harms"] else "Misinformation/Manipulation"
        )
        risk = _risk_from_harm(harm)
        tip = _prevention_tip(harm)

        header = f"{title}" + (f" ({year})" if year else "")
        with st.expander(header, expanded=False):
            st.write(f"**Sector:** {sector}  |  **Harm:** {harm}  |  **Risk:** {risk}")
            st.write(f"**Source:** {source_name}")
            if desc:
                st.write(desc)
            if tip:
                st.write(f"**How to prevent:** {tip}")
            if url != "#":
                st.markdown(f"[Source link]({url})")
            st.button(
                "Analyze this article",
                key=f"analyze_top_ai_{i}",
                on_click=_queue_for_analysis,
                args=(url,),
            )


# -----------------------
# Manual URL Analyzer (form)
# -----------------------
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
    "The URL Analyzer evaluates the single article you submit. It does not change any other content on this page."
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
                    "I could not extract readable text from that page. Try another link or paste the article text below."
                )
                st.stop()
            report = analyze_text(data["text"], data["links"], url)
        except Exception as e:
            st.error(f"Could not fetch or analyze that URL: {e}")
            st.stop()

    render_article_full(
        data["title"] or "Untitled article",
        data["text"],
        report,
    )
