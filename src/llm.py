"""
LLM module — isolated, replaceable text generation.

Provider selected for this project: OpenAI gpt-4o-mini
  - Inexpensive and suitable for learning projects
  - Strong instruction-following for grounded Q&A
  - Easy API; swap providers by implementing LLMClient

Why isolate the LLM?
  Retrieval and prompting should not depend on OpenAI vs another vendor.
  Swap providers by changing one module.

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

DEFAULT_MODEL = "gpt-4o-mini"


def _get_api_key() -> str:
    """
    Read API key from environment.

    Supports LLM_API_KEY (preferred) and OPENAI_API_KEY (legacy fallback).
    Never hardcode secrets in source code.
    """
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "LLM_API_KEY not found. Copy .env.example to .env and set your API key."
        )
    return key


class LLMError(Exception):
    """Raised when the LLM API call fails."""


class LLMClient(ABC):
    """Abstract interface — implement this to swap LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a completed prompt to the LLM and return the text response."""

    def answer_with_context(
        self,
        question: str,
        chunks: list[dict],
        low_confidence: bool = False,
    ) -> str:
        """
        Build a grounded prompt from question + chunks, then generate an answer.

        Input:  question, retrieved chunks, optional low-confidence flag
        Output: answer string
        """
        prompt = build_rag_prompt(question, chunks, low_confidence=low_confidence)
        return self.generate(prompt)


class OpenAIClient(LLMClient):
    """OpenAI API implementation using gpt-4o-mini by default."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ):
        try:
            from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from exc

        self.model = model
        self._api_key = api_key or _get_api_key()
        self._client = OpenAI(api_key=self._api_key)
        self._APIConnectionError = APIConnectionError
        self._APIStatusError = APIStatusError
        self._RateLimitError = RateLimitError

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMError("The LLM returned an empty response.")
            return content.strip()
        except self._RateLimitError as exc:
            logger.exception("OpenAI rate limit exceeded")
            raise LLMError(
                "The LLM API rate limit was exceeded. Please wait and try again."
            ) from exc
        except self._APIConnectionError as exc:
            logger.exception("OpenAI connection error")
            raise LLMError(
                "Could not connect to the LLM API. Check your internet connection."
            ) from exc
        except self._APIStatusError as exc:
            logger.exception("OpenAI API status error: %s", exc.status_code)
            if exc.status_code == 401:
                raise LLMError(
                    "Invalid API key. Check LLM_API_KEY in your .env file."
                ) from exc
            raise LLMError(
                f"The LLM API returned an error (status {exc.status_code}). "
                "See terminal logs for details."
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected LLM error")
            raise LLMError(
                "An unexpected error occurred while calling the LLM. "
                "See terminal logs for details."
            ) from exc


def create_llm(provider: str = "openai") -> LLMClient:
    """Factory function to create an LLM client."""
    if provider == "openai":
        return OpenAIClient()
    raise ValueError(f"Unknown LLM provider: {provider}")
