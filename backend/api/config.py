"""
Environment-based configuration for the API layer.

Nothing here is a secret. GEMINI_API_KEY lives in the project-root .env and is
read by src/llm.py; this module only adds the settings the API boundary itself
needs (CORS, upload limits, database URL).
"""

import os
from pathlib import Path

# Load the project-root .env BEFORE any getenv below runs.
#
# This module is the first thing main.py imports, and it reads its settings at
# import time. The .env was previously only loaded further down the import
# chain (src/embeddings.py), by which point Settings() had already been
# constructed — so every variable here silently fell back to its default. That
# is how a configured DATABASE_URL could be ignored and the API quietly serve
# from in-memory storage.
#
# Optional: a deployed function is handed its environment by the platform and
# has no .env file to read.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings:
    def __init__(self) -> None:
        # Comma-separated list of frontend origins allowed to call this API.
        # Defaults cover local Next.js dev; set explicitly in production.
        self.allowed_origins = _split_csv(
            os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
        )
        # Vercel caps a serverless function's REQUEST BODY at 4.5MB and returns
        # 413 FUNCTION_PAYLOAD_TOO_LARGE above it — a platform limit no
        # application setting can raise. Defaulting to just under it means an
        # oversized upload gets our own clear message instead of an opaque
        # platform error. Raise it only where the platform allows (a container
        # host); to accept larger files on Vercel the upload has to go to blob
        # storage directly from the browser, with the function fetching it.
        self.max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))
        # Used when a chat request does not name a provider. There is no
        # fallback: an unavailable provider is an error, not a substitution.
        self.llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
        # Unset -> in-memory storage (local dev and tests). Set -> Postgres.
        self.database_url = os.getenv("DATABASE_URL") or None

        # Shared secret for the API. Unset means the API is open, which is the
        # local development default; main.py refuses to start in that state on
        # a deployment. It is never sent to a browser — see security.py.
        self.api_key = os.getenv("API_KEY") or None

        # Per-client caps over a rolling window, counted in the database so the
        # limit is shared by every instance. 0 disables a limit.
        self.rate_limit_window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))
        # Sized against the free tiers this runs on: Gemini allows 20
        # generations/day per model, Hugging Face has no daily embedding cap.
        # So questions are the tighter of the two.
        self.rate_limit_questions = int(os.getenv("RATE_LIMIT_QUESTIONS", "20"))
        self.rate_limit_uploads = int(os.getenv("RATE_LIMIT_UPLOADS", "10"))


settings = Settings()
