"""
Who may call this API, and how often.

------------------------------------------------------------------------------
AUTHENTICATION
------------------------------------------------------------------------------
A single shared key in an X-API-Key header. Not user accounts: there are no
users to model here, and the thing being protected is a pair of API quotas, not
per-person data.

The key must never reach a browser. Anything a browser holds is readable by
whoever opens devtools, so a key shipped to the client would authenticate the
whole internet. The Next.js frontend therefore calls its own route handler,
which runs server-side and forwards the request with the key attached
(frontend/app/api/[...path]/route.ts). The browser never sees it.

/health is deliberately unauthenticated so uptime monitoring works.

------------------------------------------------------------------------------
RATE LIMITING
------------------------------------------------------------------------------
Counters live in Postgres, not in the process. A serverless deployment runs
many short-lived instances, so an in-process counter would allow
(number of instances x the limit) and quietly fail at exactly the moment load
is high enough to matter. The database is already there and is the only thing
all instances share.

Two limits, because the two expensive operations fail differently:

    uploads   embed every chunk of a document. One large PDF can spend a
              meaningful share of a daily embedding quota.
    questions one embedding plus one generation. Generation is the scarcer
              quota (20/day per model on Gemini's free tier).

Requests are identified by the caller's IP, which the proxy forwards as
X-Client-Id. Trusting a header is safe here precisely because the API key is
required first: only our own frontend can reach this code at all.
"""

from fastapi import Depends, Header, HTTPException, Request

from .config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Reject anything without the shared key.

    When no key is configured the API is open — that is the local development
    default, and main.py refuses to start in that state on a deployment.
    """
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


def client_id(request: Request) -> str:
    """
    Who to count this request against.

    X-Client-Id is set by our own proxy to the browser's address. Falling back
    to the socket address keeps direct callers (curl, tests) rate limited too,
    rather than unlimited.
    """
    forwarded = request.headers.get("x-client-id") or request.headers.get("x-forwarded-for")
    if forwarded:
        # x-forwarded-for is a chain; the original client is the first entry.
        return forwarded.split(",")[0].strip()[:100]
    return request.client.host if request.client else "unknown"


async def limit_uploads(request: Request) -> None:
    await _enforce(request, "upload", settings.rate_limit_uploads)


async def limit_questions(request: Request) -> None:
    await _enforce(request, "question", settings.rate_limit_questions)


async def _enforce(request: Request, kind: str, limit: int) -> None:
    if limit <= 0:  # 0 disables the limit
        return
    storage = request.app.state.rag_state.storage
    who = client_id(request)
    try:
        used = storage.record_and_count(who, kind, settings.rate_limit_window_seconds)
    except Exception:  # noqa: BLE001
        # A rate limiter that breaks must not take the API down with it. Failing
        # open is the right call for a quota guard on a single-tenant app; the
        # quotas themselves are still enforced upstream by the providers.
        return
    if used > limit:
        window_minutes = max(1, settings.rate_limit_window_seconds // 60)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit reached: {limit} {kind}s per {window_minutes} minutes. "
                "Please wait and try again."
            ),
            headers={"Retry-After": str(settings.rate_limit_window_seconds)},
        )


# Convenience bundles for the routers.
AUTH = Depends(require_api_key)
UPLOAD_LIMIT = Depends(limit_uploads)
QUESTION_LIMIT = Depends(limit_questions)
