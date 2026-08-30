"""
LLM module — isolated, replaceable text generation.

Provider: Google Gemini (gemini-2.0-flash)
  - Free-tier friendly for learning projects
  - Strong instruction-following for grounded Q&A
  - Official SDK: google-genai

Why isolate the LLM?
  Retrieval and prompting should not depend on which API vendor you use.
  Swap providers by implementing LLMClient in this module only.

Why can the LLM still hallucinate even with RAG?
  The model is a probabilistic text generator. It may ignore instructions,
  blend training knowledge with context, or misread chunks. Retrieval quality
  AND prompt rules both matter.

Input:  user question + retrieved chunks (via answer_with_context)
Output: generated answer text
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from prompt_builder import build_rag_prompt

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Pinned to an explicit version, not an alias like "gemini-flash-latest".
# Reason: a moving alias would silently change answers between runs, which makes
# it impossible to tell whether a retrieval change helped or the model changed.
#
# gemini-2.5-flash was retired by Google for new API keys (404 NOT_FOUND on
# generateContent, even though it still appears in models.list()).
#
# gemini-3.6-flash works but its free tier allows only 20 requests PER DAY
# (quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier), which a single
# test session exhausts. Free-tier quota is per-model, so 3.5-flash has its own
# separate budget and is the practical choice for this project.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Free-tier quotas are PER MODEL and PER DAY (20 requests/day/model observed).
# A single test session exhausts one model. When a daily-quota error comes
# back, the client moves to the next model in this list and records which one
# actually answered (GeminiClient.active_model), so results stay attributable.
# Per-MINUTE limits are not handled here — callers pace themselves for those.
MODEL_FALLBACKS = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def _get_api_key() -> str:
    """
    Read Gemini API key from environment.

    Set GEMINI_API_KEY in your .env file. Never hardcode secrets in source code.
    """
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY not found. Copy .env.example to .env and set your API key."
        )
    return key


class LLMError(Exception):
    """Raised when the LLM API call fails."""


def is_daily_quota_error(message: str) -> bool:
    """
    A per-DAY quota error means this model is done until tomorrow — retrying
    is pointless, switching models is not. A per-MINUTE error clears in
    seconds and should be retried after a pause, not escaped by switching.
    Google names the quota in the error body ("...PerDayPerProjectPerModel...").
    """
    m = message.lower()
    return "perday" in m or "per_day" in m or "per day" in m


class LLMClient(ABC):
    """Abstract interface — implement this to swap LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a completed prompt to the LLM and return the text response."""

    def answer_with_context(
        self,
        question: str,
        context: str,
        low_confidence: bool = False,
        depth: str = "normal",
        wants_example: bool = False,
        conversation: str | None = None,
    ) -> str:
        """
        Build a grounded prompt from question + context, then generate an answer.

        Input:  question, a formatted evidence block, low-confidence flag,
                requested depth, example flag, optional recent conversation text
        Output: answer string

        The evidence block is produced by context_builder.py. This layer stays
        deliberately ignorant of chunks, pages and similarity scores — it only
        forwards text to a provider, which is what makes the provider swappable.
        """
        prompt = build_rag_prompt(
            question, context, low_confidence=low_confidence,
            depth=depth, wants_example=wants_example, conversation=conversation,
        )
        return self.generate(prompt)


class GeminiClient(LLMClient):
    """Google Gemini API implementation using the official google-genai SDK."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ):
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            ) from exc

        self.model = model
        self.active_model = model
        self._api_key = api_key or _get_api_key()
        self._client = genai.Client(api_key=self._api_key)
        self._types = types
        self._errors = errors
        # Models still worth trying after the active one runs out for the day.
        self._fallbacks = [m for m in MODEL_FALLBACKS if m != model]

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.active_model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    temperature=0.2,
                ),
            )
            text = response.text
            if not text:
                raise LLMError("The LLM returned an empty response.")
            return text.strip()
        except self._errors.ClientError as exc:
            message = str(exc).lower()
            if "api key" in message or "401" in message or "403" in message:
                logger.exception("Gemini API client error")
                raise LLMError(
                    "Invalid API key. Check GEMINI_API_KEY in your .env file."
                ) from exc
            if "404" in message or "not_found" in message:
                logger.exception("Gemini API client error")
                raise LLMError(
                    f"The model '{self.active_model}' is not available to this API key. "
                    "Change DEFAULT_MODEL in src/llm.py to a model listed by models.list()."
                ) from exc
            if "429" in message or "quota" in message or "rate" in message:
                if is_daily_quota_error(message) and self._fallbacks:
                    exhausted, self.active_model = self.active_model, self._fallbacks.pop(0)
                    logger.warning(
                        "Daily quota exhausted for %s; switching to %s",
                        exhausted, self.active_model,
                    )
                    print(f"\n=== LLM === daily quota exhausted for {exhausted}; "
                          f"switching to {self.active_model}")
                    return self.generate(prompt)
                logger.exception("Gemini API client error")
                raise LLMError(
                    "The Gemini API rate limit was exceeded. Please wait and try again."
                ) from exc
            logger.exception("Gemini API client error")
            raise LLMError(
                "The Gemini API returned an error. See terminal logs for details."
            ) from exc
        except self._errors.ServerError as exc:
            logger.exception("Gemini API server error")
            raise LLMError(
                "The Gemini API is temporarily unavailable. Please try again later."
            ) from exc
        except LLMError:
            raise
        except Exception as exc:
            logger.exception("Unexpected Gemini LLM error")
            raise LLMError(
                "An unexpected error occurred while calling the LLM. "
                "See terminal logs for details."
            ) from exc


def create_llm(provider: str = "gemini") -> LLMClient:
    """Factory function to create an LLM client."""
    if provider == "gemini":
        return GeminiClient()
    raise ValueError(f"Unknown LLM provider: {provider}")
