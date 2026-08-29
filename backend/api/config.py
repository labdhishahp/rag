"""
Environment-based configuration for the API layer.

Nothing here is a secret. GEMINI_API_KEY lives in the project-root .env and is
read by src/llm.py exactly as it was for the Streamlit app — this module only
adds the settings the API boundary itself needs (CORS, upload limits, session
lifetime).
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
        self.max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
        self.session_ttl_seconds = int(os.getenv("SESSION_TTL_SECONDS", str(6 * 3600)))
        self.max_sessions = int(os.getenv("MAX_SESSIONS", "200"))
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini")


settings = Settings()
