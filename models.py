"""
Shared data models for the Article Processor.

- NewsItem: shape of the lightweight message produced by the news scraper
  and consumed from SQS.
- ArticleAnalysis: strict schema we ask the LLM (via NVIDIA NIM) to return.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    """Schema of the lightweight message pushed to SQS by the scraper."""

    title: str
    url: str
    published_at: Optional[str] = None
    source: Optional[str] = None


ArticleType = Literal[
    "earnings",
    "partnership",
    "merger",
    "acquisition",
    "government_policy",
    "macro",
    "lawsuit",
    "management_change",
    "dividend",
    "analyst_rating",
    "product_launch",
    "general",
]


class ArticleAnalysis(BaseModel):
    """Structured output requested from the LLM for each article."""

    summary: str = Field(
        description=(
            "Concise (2-4 sentence) investor-focused summary of the article. "
            "Factual only, no speculation."
        )
    )
    sentiment: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "Sentiment score from -1.0 (very negative) to 1.0 (very positive), "
            "0.0 = neutral, from the perspective of the companies/industries "
            "mentioned."
        ),
    )
    importance: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How impactful this article is for investors: 0.0 (irrelevant/noise) "
            "to 1.0 (major market-moving news)."
        ),
    )
    company_tags: List[str] = Field(
        default_factory=list,
        description=(
            "NSE ticker symbols of companies EXPLICITLY and directly mentioned "
            "in the article, e.g. ['TCS', 'INFY']. Do not infer companies that "
            "are not directly named."
        ),
    )
    industry_tags: List[str] = Field(
        default_factory=list,
        description=(
            "Broad industries/sectors relevant to the article, "
            "e.g. ['Information Technology', 'Banking']."
        ),
    )
    article_type: ArticleType = Field(
        description="Single best-fitting category for this article."
    )
