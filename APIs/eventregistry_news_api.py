"""
EVENT-REGISTRY NEWS API
───────────────────────────────────────────────────────────────────────────
NewsAPI.ai (EventRegistry) provides real-time access to news articles from
100,000+ publishers worldwide. Articles include full body text, publication
date, source outlet, geographic origin, and a pre-computed sentiment score
(–1.0 to +1.0) from EventRegistry's internal NLP pipeline.

This module fetches Iran conflict articles from UK and Nigerian sources,
labels each article by sentiment, and returns a clean DataFrame for NLP.

Sentiment classes:  0 = negative (<-0.15)  |  1 = neutral  |  2 = positive (>0.15)

Get API key : https://www.newsapi.ai/  → My Profile → API Key (free: 2,000 req/day)
Docs        : https://www.newsapi.ai/documentation?tab=introduction
Install     : pip install eventregistry pandas

How to run:
    * Run the `get_news()` function

Params:
    keywords        : list of search terms (OR logic)
    countries       : list of country names to filter by source location
    max_per_country : max articles per country (default 500)
    min_body_words  : skip articles shorter than this (default 100)
    output_path     : optional .csv path to save results
    api_key         : your EventRegistry API key

Usage:
    df = get_news(
        keywords=["Iran war", "Iran nuclear", "Iran sanctions", "Iran conflict"],
        countries=["United Kingdom", "Nigeria"],
        max_per_country=500,
        output_path="iran_news_dataset.csv",
        api_key="YOUR_API_KEY"
    )
    print(df.head())
───────────────────────────────────────────────────────────────────────────
"""

import time
import pandas as pd
from typing import List, Optional
from eventregistry import EventRegistry, QueryArticlesIter, QueryItems, ReturnInfo, ArticleInfoFlags


def _label(score):
    if score is None: return None, "unknown"
    if score < -0.15: return 0, "negative"
    if score >  0.15: return 2, "positive"
    return 1, "neutral"


def get_news(
    keywords: List[str],
    countries: List[str],
    max_per_country: int = 500,
    min_body_words: int = 100,
    output_path: Optional[str] = None,
    api_key: str = "YOUR_API_KEY",
) -> pd.DataFrame:

    er   = EventRegistry(apiKey=api_key, allowUseOfArchive=False)
    rows = []

    for country in countries:
        uri = er.getLocationUri(country)
        q   = QueryArticlesIter(
            keywords=QueryItems.OR(keywords),
            sourceLocationUri=uri,
            lang="eng",
            dataType=["news", "blog"]
        )
        for art in q.execQuery(er, sortBy="date", maxItems=max_per_country,
                               returnInfo=ReturnInfo(articleInfo=ArticleInfoFlags(
                                   sentiment=True, categories=True, bodyLen=-1))):
            body = (art.get("body") or "").strip()
            if len(body.split()) < min_body_words: continue
            score      = art.get("sentiment")
            cls, label = _label(score)
            if cls is None: continue
            cats = art.get("categories", [])
            rows.append({
                "title"           : (art.get("title") or "").strip(),
                "body"            : body,
                "word_count"      : len(body.split()),
                "date"            : art.get("date", ""),
                "source"          : art.get("source", {}).get("title", ""),
                "url"             : art.get("url", ""),
                "country"         : country,
                "sentiment_score" : round(score, 4),
                "sentiment_label" : label,
                "sentiment_class" : cls,
                "category"        : cats[0]["label"] if cats else "uncategorised",
            })
            time.sleep(0.05)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved → {output_path}  ({len(df)} articles)")

    return df


if __name__ == "__main__":
    df = get_news(
        keywords=["Iran war", "Iran conflict", "Iran nuclear", "Iran sanctions", "Iran missile"],
        countries=["United Kingdom", "Nigeria"],
        max_per_country=500,
        output_path="iran_news_dataset.csv",
        api_key="YOUR_API_KEY",
    )
    print(df[["date", "country", "sentiment_label", "sentiment_score", "source", "title"]].head(10))