import os
import importlib
from dataclasses import dataclass

from fastapi import Header, HTTPException


@dataclass
class CurrentUser:
    user_id: str


def _decode_jwt_user_id(token: str) -> str | None:
    """Decode JWT and extract user id. Returns None if decoding fails."""
    jwt_secret = os.getenv("AUTH_JWT_SECRET", "").strip()
    jwt_algorithm = os.getenv("AUTH_JWT_ALGORITHM", "HS256").strip() or "HS256"

    if not jwt_secret:
        return None

    try:
        jwt = importlib.import_module("jwt")

        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=[jwt_algorithm],
            options={"verify_aud": False},
        )
    except Exception:
        return None

    user_id = payload.get("user_id") or payload.get("sub") or payload.get("id")
    if user_id is None:
        return None
    user_id = str(user_id).strip()
    return user_id or None


def get_current_user(
    authorization: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_authenticated_userid: str | None = Header(default=None),
    x_authenticated_user: str | None = Header(default=None),
) -> CurrentUser:
    """
    Resolve the current user from auth sources in this order:
    1) Trusted forwarded headers from upstream auth backend
    2) Bearer JWT (if AUTH_JWT_SECRET is configured)
    """

    for candidate in (x_user_id, x_authenticated_userid, x_authenticated_user):
        if candidate and candidate.strip():
            return CurrentUser(user_id=candidate.strip())

    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        user_id = _decode_jwt_user_id(token)
        if user_id:
            return CurrentUser(user_id=user_id)

    raise HTTPException(
        status_code=401,
        detail=(
            "Unauthorized: user identity not found. Provide a Bearer token "
            "(with AUTH_JWT_SECRET configured) or forward x-user-id header "
            "from your auth backend."
        ),
    )
