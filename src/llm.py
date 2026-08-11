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

DEFAULT_MODEL = "gemini-2.0-flash"


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
        self._api_key = api_key or _get_api_key()
        self._client = genai.Client(api_key=self._api_key)
        self._types = types
        self._errors = errors

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.model,
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
            logger.exception("Gemini API client error")
            message = str(exc).lower()
            if "api key" in message or "401" in message or "403" in message:
                raise LLMError(
                    "Invalid API key. Check GEMINI_API_KEY in your .env file."
                ) from exc
            if "429" in message or "quota" in message or "rate" in message:
                raise LLMError(
                    "The Gemini API rate limit was exceeded. Please wait and try again."
                ) from exc
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
