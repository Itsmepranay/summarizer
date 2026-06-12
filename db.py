"""
PostgreSQL (Amazon RDS) access layer for the Article Processor.

Uses a module-level connection that is reused across warm Lambda
invocations to avoid the overhead of opening a new TCP/TLS connection on
every SQS message.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Optional

import psycopg2

logger = logging.getLogger(__name__)

DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

# Reused across warm Lambda invocations within the same execution environment.
_connection = None


def get_connection():
    """Return a live psycopg2 connection, opening a new one if needed."""
    global _connection

    if _connection is not None and _connection.closed == 0:
        return _connection

    logger.info("Opening new DB connection to %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
    _connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=5,
    )
    # Each Lambda invocation processes one article; autocommit keeps things
    # simple and avoids holding open transactions across invocations.
    _connection.autocommit = True
    return _connection


def url_to_hash(url: str) -> str:
    """SHA-256 hash of the article URL - the deduplication key."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def article_exists(url_hash: str) -> bool:
    """Return True if an article with this url_hash is already stored."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM news_articles WHERE url_hash = %s", (url_hash,))
        return cur.fetchone() is not None


def insert_article(
    *,
    url_hash: str,
    url: str,
    title: str,
    source: Optional[str],
    published_at: Optional[datetime],
    raw_content: str,
    summary: str,
    sentiment_score: float,
    importance_score: float,
    article_type: str,
    company_tags: list[str],
    industry_tags: list[str],
) -> None:
    """Insert a fully-processed article into news_articles.

    Uses ON CONFLICT DO NOTHING on url_hash as a final safety net against
    races between concurrent Lambda invocations processing the same URL.
    """
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO news_articles (
                url_hash, url, title, source, published_at, raw_content,
                summary, sentiment_score, importance_score, article_type,
                company_tags, industry_tags
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s::jsonb, %s::jsonb
            )
            ON CONFLICT (url_hash) DO NOTHING
            """,
            (
                url_hash,
                url,
                title,
                source,
                published_at,
                raw_content,
                summary,
                sentiment_score,
                importance_score,
                article_type,
                json.dumps(company_tags),
                json.dumps(industry_tags),
            ),
        )
