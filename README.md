# Media Bias & AI Incident Dashboard

A Streamlit-powered tool for exploring how AI-related incidents are covered in the news, spot signas of bias, and see how headlines shift across different news sources — all in one place

## Overview

The dashboard is designed to help users critically evaluate AI-related news by:
- **Scanning articles** for framing cues such as hedging terms, loaded language, and sourcing quality.
- **Tagging sectors and risk types** (bias, misinformation, privacy risks) based on keyword scanning.
- **Comparing live headlines** from multiple outlets to understand how the same topic is framed across sources.

This is a research and educational tool. It does **not** store or republish full articles.

---

## How It Works

1. **Paste an article URL** into the analyzer to:
   - Extract readable text.
   - Flag  bias-related patterns and emotional framing.
   - Show quick stats (word count, sentiment, reading level).
   - Highlight relevant examples in question.

2. **Browse live AI headlines** using the NewsAPI feed:
   - Filter by keywords, country, or category.
   - Quickly scan summaries for bias and potential harms.
   - Launch an in-depth article analysis with one click.

3. **Sector & harm tags**:
   - Automated tagging based on content keywords.
   - Maps stories to risk types (e.g., misinformation, bias, fraud).

---

## Run Locally

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Start the app
streamlit run app.py
```

---

## Future Plans

Planned enhancements include:
- **Additional API Sources** (GDELT, Media Bias/Fact Check, Google Fact Check Tools).
- **Bias detection in multiple languages** for global news sources.
- **Customizable bias word lists** so users can adapt detection to specific contexts.
- **Automated scans** with stored analysis summaries.
- **Exportable reports** (PDF/CSV) for research and classroom use.

---

## API Usage & Licensing

This project uses the **Developer (free)** tier of [NewsAPI.org](https://newsapi.org/) for research and educational purposes only.

**Key usage points:**
- All content is retrieved via NewsAPI search endpoints.
- Only article titles, descriptions, and links are displayed.
- No full text, images, or proprietary content are stored or redistributed.
- Only one API key is used, and requests stay within the free-tier quota.
- Usage complies fully with [NewsAPI’s terms of service](https://newsapi.org/terms).
