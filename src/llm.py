"""
LLM module — isolated, replaceable text generation.

Why isolate the LLM?
  The rest of the RAG pipeline (retrieval, prompting) should not depend on
  OpenAI vs Anthropic vs a local model. Swap providers by changing one module.

Why can the LLM still hallucinate even with RAG?
  The model is a probabilistic text generator. It may:
  - Ignore instructions
  - Blend training knowledge with context
  - Misread or over-interpret retrieved chunks
  That is why retrieval quality AND prompt rules both matter.

Input:  a completed prompt string (built by prompt_builder)
Output: generated answer text
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class LLMClient(ABC):
    """Abstract interface — implement this to swap LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the text response."""


class OpenAIClient(LLMClient):
    """
    OpenAI API implementation.

    Requires OPENAI_API_KEY in the environment (or .env file).
    Never hardcode API keys in source code.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from exc

        self.model = model
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in your environment or in a .env file."
            )

        self._client = OpenAI(api_key=key)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Lower = less creative = fewer hallucinations
        )
        return response.choices[0].message.content.strip()


def create_llm(provider: str = "openai") -> LLMClient:
    """
    Factory function to create an LLM client.

    Change the provider string here when you add new backends.
    """
    if provider == "openai":
        return OpenAIClient()
    raise ValueError(f"Unknown LLM provider: {provider}")
