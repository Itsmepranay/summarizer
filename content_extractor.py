"""
Article content extraction.

Downloads the raw HTML for a news article URL and extracts the main
readable text using trafilatura. This is the only place in the system
that downloads raw article pages (per design principle #5).
"""

from __future__ import annotations

import logging
from typing import Optional

import requests
import trafilatura

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "PortfolioIntelligenceBot/1.0"
)

REQUEST_TIMEOUT_SECONDS = 15
MIN_CONTENT_LENGTH = 100  # below this, treat extraction as a failure


def extract_article_content(url: str) -> Optional[str]:
    """
    Download `url` and extract the main readable article text.

    Returns the extracted text, or None if the page could not be fetched
    or no meaningful content could be extracted (e.g. paywalled, JS-only
    pages, 404s, etc.). Callers should treat None as "skip this article".
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to download article: %s", url)
        return None

    html = response.text
    if not html:
        logger.warning("Empty response body for article: %s", url)
        return None

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )

    if not text or len(text.strip()) < MIN_CONTENT_LENGTH:
        logger.warning("Extracted content too short/empty for article: %s", url)
        return None

    return text.strip()
