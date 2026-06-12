"""
Article Processor - Lambda (Module 3)

Triggered by SQS messages produced by the news scraper (Module 2).

For each message:
  1. Parse {title, url, published_at, source}.
  2. Compute url_hash = SHA256(url).
  3. If url_hash already exists in news_articles -> skip (dedup, no LLM call).
  4. Otherwise:
       - Download the article and extract readable text -> raw_content.
       - Send raw_content to the LLM (LangChain + NVIDIA NIM) for structured
         JSON analysis (summary, sentiment, importance, tags, type).
       - Store everything in news_articles.

Uses SQS partial batch failure reporting (ReportBatchItemFailures must be
enabled on the event source mapping) so that only failed records are
retried/redriven, while successfully processed records are removed from the
queue.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import db
from content_extractor import extract_article_content
from llm_processor import analyze_article
from models import NewsItem

logging.basicConfig(level="INFO")
logger = logging.getLogger("article-processor")


def _parse_published_at(value: str | None) -> datetime | None:
    """Parse the ISO-8601 published_at string produced by the scraper."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning("Could not parse published_at value: %r", value)
        return None


def process_record(record: dict[str, Any]) -> None:
    """Process a single SQS record. Raises on any failure (-> retry)."""
    body = json.loads(record["body"])
    item = NewsItem(**body)

    url_hash = db.url_to_hash(item.url)

    # --- Deduplication ----------------------------------------------------
    if db.article_exists(url_hash):
        logger.info("Duplicate article, skipping (no LLM call): %s", item.url)
        return

    logger.info("New article, processing: %s", item.url)

    # --- Download + extract -------------------------------------------------
    raw_content = extract_article_content(item.url)
    if not raw_content:
        # Nothing useful to analyze (paywall, JS-only page, dead link, etc.)
        # We deliberately do NOT raise here, so this message is treated as
        # successfully processed and removed from the queue rather than
        # retried indefinitely.
        logger.warning("No extractable content, skipping article: %s", item.url)
        return

    # --- LLM structured analysis --------------------------------------------
    analysis = analyze_article(item.title, raw_content)

    # --- Persist -------------------------------------------------------------
    db.insert_article(
        url_hash=url_hash,
        url=item.url,
        title=item.title,
        source=item.source,
        published_at=_parse_published_at(item.published_at),
        raw_content=raw_content,
        summary=analysis.summary,
        sentiment_score=analysis.sentiment,
        importance_score=analysis.importance,
        article_type=analysis.article_type,
        company_tags=analysis.company_tags,
        industry_tags=analysis.industry_tags,
    )

    logger.info(
        "Stored article '%s' | sentiment=%.2f importance=%.2f type=%s "
        "company_tags=%s industry_tags=%s",
        item.title,
        analysis.sentiment,
        analysis.importance,
        analysis.article_type,
        analysis.company_tags,
        analysis.industry_tags,
    )


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    SQS-triggered handler.

    Returns `batchItemFailures` listing any records that raised an
    exception, so the Lambda SQS event source mapping (with
    ReportBatchItemFailures enabled) only retries those specific messages.
    """
    batch_item_failures: list[dict[str, str]] = []

    for record in event.get("Records", []):
        message_id = record["messageId"]
        try:
            process_record(record)
        except Exception:
            logger.exception("Failed to process SQS message %s", message_id)
            batch_item_failures.append({"itemIdentifier": message_id})

    return {"batchItemFailures": batch_item_failures}
