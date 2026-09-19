"""
LLM module — isolated, replaceable text generation.

Two providers, and the CALLER chooses which one answers:

    anthropic   claude-opus-5 via the official anthropic SDK
    gemini      gemini-3.5-flash-lite via google-genai

Why isolate the LLM?
  Retrieval and prompting do not depend on which API vendor you use. Swap or
  add a provider by implementing LLMClient in this module only — nothing in
  rag.py, the retriever or the prompt builder knows a provider name.

Why there is NO provider fallback here:
  The user selects the provider per request, so substituting a different one
  would answer a question they did not ask and label the answer with a model
  that did not produce it. create_llm() therefore builds exactly what was asked
  for, or raises. (embeddings.py DOES fall back, for reasons specific to vector
  spaces — see create_llm's docstring for the contrast.)

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

# Walk up for .env rather than assuming a fixed depth — see embeddings.py.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / ".env").is_file():
        load_dotenv(_candidate / ".env")
        break

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

# --- Anthropic (primary) ---------------------------------------------------
# Pinned for the same reason the Gemini model is: a moving alias would change
# answers between runs and make it impossible to tell whether a retrieval change
# helped or the model did.
ANTHROPIC_MODEL = "claude-opus-5"
# A ceiling, not a target — answers here are short. Sized to stay well inside
# the SDK's default HTTP timeout on a non-streaming request.
ANTHROPIC_MAX_TOKENS = 16000


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

    name = "gemini"

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


class AnthropicClient(LLMClient):
    """
    Anthropic Claude. PRIMARY provider.

    Why primary: generation here is grounded extraction under strict rules —
    cite the passage a fact came from, refuse when the evidence does not
    support an answer. Instruction-following is the whole job. It also removes
    the free-tier cliff that makes Gemini switch models mid-session (see
    MODEL_FALLBACKS above), which is what made answers unattributable.

    Two differences from the Gemini client, both required rather than stylistic:

      no temperature   Claude Opus 5 rejects temperature/top_p/top_k with a 400.
                       Determinism is instead a property of the prompt, which
                       already pins the answer to the evidence block.

      stop_reason      A response can come back with stop_reason "refusal" and
                       no text. Checked before reading content, so a refusal
                       surfaces as a clear LLMError rather than an IndexError.

    Anthropic does NOT offer an embedding model, so this changes generation
    only. Embeddings stay on Hugging Face with the Gemini fallback.
    """

    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_MODEL, api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            ) from exc

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Copy .env.example to .env and set your API key."
            )
        self.model = model
        self.active_model = model
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=key)

    def generate(self, prompt: str) -> str:
        a = self._anthropic
        try:
            response = self._client.messages.create(
                model=self.active_model,
                max_tokens=ANTHROPIC_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
        # Most specific first: the 4xx classes below all subclass APIStatusError,
        # and APITimeoutError subclasses APIConnectionError.
        except a.AuthenticationError as exc:
            logger.exception("Anthropic authentication error")
            raise LLMError(
                "Invalid API key. Check ANTHROPIC_API_KEY in your .env file."
            ) from exc
        except a.NotFoundError as exc:
            logger.exception("Anthropic model not found")
            raise LLMError(
                f"The model '{self.active_model}' is not available to this API key."
            ) from exc
        except a.RateLimitError as exc:
            logger.exception("Anthropic rate limit")
            raise LLMError(
                "The Anthropic API rate limit was exceeded. Please wait and try again."
            ) from exc
        except a.APIStatusError as exc:
            logger.exception("Anthropic API error")
            raise LLMError(
                "The Anthropic API returned an error. See terminal logs for details."
            ) from exc
        except a.APIConnectionError as exc:
            logger.exception("Anthropic connection error")
            raise LLMError(
                "Could not reach the Anthropic API. Please try again later."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - mirrors GeminiClient's last resort
            logger.exception("Unexpected Anthropic LLM error")
            raise LLMError(
                "An unexpected error occurred while calling the LLM. "
                "See terminal logs for details."
            ) from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMError("The model declined to answer this request.")

        # content is a list of blocks; only the text ones carry the answer.
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise LLMError("The LLM returned an empty response.")
        return text


# The providers a caller may choose from. The API validates against this list,
# so an unknown name is rejected before any client is constructed.
LLM_PROVIDERS = ("anthropic", "gemini")

# The provider used when a request does not name one.
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

_LLM_BUILDERS = {"anthropic": AnthropicClient, "gemini": GeminiClient}


def create_llm(provider: str = DEFAULT_LLM_PROVIDER) -> LLMClient:
    """
    Build one named LLM client. Raises if it cannot be constructed.

    THERE IS DELIBERATELY NO PROVIDER FALLBACK HERE.

    The user chooses which model answers their question, so answering with a
    different one would be answering a question they did not ask — and the
    answer would carry a provider label that did not produce it, which is the
    same class of dishonesty as a citation pointing at the wrong passage.
    A missing key surfaces as an error naming the provider instead.

    Note the contrast with embeddings.py, which DOES fall back: there the
    fallback happens at INDEX time and is recorded on the document, because two
    embedding models produce incomparable coordinate systems and a query must
    use the one that indexed the document. Two LLMs both read the same evidence
    block, so the constraint is the user's intent rather than correctness.
    """
    try:
        builder = _LLM_BUILDERS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Expected one of {sorted(_LLM_BUILDERS)}."
        ) from None
    return builder()
