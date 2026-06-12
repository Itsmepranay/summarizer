"""
LLM-based article analysis using LangChain + NVIDIA NIM.

Sends raw article content to an NVIDIA NIM-hosted chat model and gets back
a strictly-typed `ArticleAnalysis` object via LangChain's structured output
support. The provider is intentionally abstracted behind LangChain so it can
be swapped (OpenAI, Bedrock, Azure, etc.) by changing only this module.
"""

from __future__ import annotations

import logging
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from models import ArticleAnalysis

logger = logging.getLogger(__name__)

# NVIDIA NIM model to use. Any NIM chat model that supports tool/function
# calling can be used here (required for structured output).
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-70b-instruct")

# NVIDIA_API_KEY is read automatically by ChatNVIDIA from the environment,
# but we read it explicitly here so we fail fast with a clear error if it's
# missing, rather than failing deep inside the LangChain call.
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

# Truncate article content sent to the LLM to keep prompt size & cost
# bounded. ~8000 chars is comfortably within context limits for typical
# news articles while covering the full body of almost all of them.
MAX_CONTENT_CHARS = int(os.environ.get("MAX_CONTENT_CHARS", "8000"))

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial analyst assistant for an Indian stock market "
     "investor intelligence platform. You read news articles and extract "
     "structured information that helps retail investors understand how "
     "the article might affect specific listed companies.\n\n"
     "Guidelines:\n"
     "- summary: 2-4 sentences, factual and investor-focused. No speculation "
     "or investment advice.\n"
     "- sentiment: -1.0 (very negative) to 1.0 (very positive), 0.0 = "
     "neutral, judged from the perspective of the companies/industries "
     "mentioned.\n"
     "- importance: 0.0 (irrelevant/noise) to 1.0 (major market-moving "
     "news).\n"
     "- company_tags: ONLY companies that are EXPLICITLY named or clearly "
     "and unambiguously identifiable in the article. Use standard NSE "
     "ticker symbols (e.g. TCS, INFY, RELIANCE, HDFCBANK, ICICIBANK). Do "
     "NOT guess or infer companies that are not directly mentioned. If no "
     "specific company is mentioned, return an empty list.\n"
     "- industry_tags: broad sectors relevant to the article "
     "(e.g. 'Information Technology', 'Banking', 'Pharmaceuticals', "
     "'Automobile', 'Energy').\n"
     "- article_type: choose the single best-fitting category."),
    ("human", "Title: {title}\n\nArticle content:\n{content}"),
])


def _get_llm() -> ChatNVIDIA:
    return ChatNVIDIA(
        model=NIM_MODEL,
        api_key=NVIDIA_API_KEY,
        temperature=0,
    )


def analyze_article(title: str, raw_content: str) -> ArticleAnalysis:
    """
    Run the article through the LLM and return a structured ArticleAnalysis.

    Raises if the LLM call fails or returns output that doesn't validate
    against `ArticleAnalysis` - callers (the Lambda handler) treat this as a
    retryable failure for that SQS record.
    """
    llm = _get_llm()
    structured_llm = llm.with_structured_output(ArticleAnalysis)
    chain = _PROMPT | structured_llm

    truncated_content = raw_content[:MAX_CONTENT_CHARS]

    logger.info("Calling NIM model '%s' for structured analysis", NIM_MODEL)
    result = chain.invoke({"title": title, "content": truncated_content})

    if not isinstance(result, ArticleAnalysis):
        # Defensive: with_structured_output should already enforce this,
        # but guard against provider/version quirks.
        result = ArticleAnalysis.model_validate(result)

    return result
