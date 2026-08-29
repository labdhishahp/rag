"""
Environment-based configuration for the API layer.

Nothing here is a secret. GEMINI_API_KEY lives in the project-root .env and is
read by src/llm.py; this module only adds the settings the API boundary itself
needs (CORS, upload limits, database URL).
"""

import os


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
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini")
        # Unset -> in-memory storage (local dev and tests). Set -> Postgres.
        self.database_url = os.getenv("DATABASE_URL") or None


settings = Settings()
