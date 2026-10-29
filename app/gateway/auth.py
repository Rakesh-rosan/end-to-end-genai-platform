"""Authentication (who are you?) and role-based authorization (what may you do?).

Demo: username/password -> HS256 JWT. Production: delegate login to an identity
provider (Cognito, Auth0, Entra ID, Keycloak) over OIDC and only *verify* its
RS256 tokens here against the provider's JWKS.
"""

import hmac
import time
from dataclasses import dataclass, field

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings

# Never store plaintext passwords outside a demo. Use argon2/bcrypt hashes.
DEMO_USERS = {
    "alice": {"password": "alice123", "customer_id": "c-alice", "roles": ["shopper"]},
    "bob": {"password": "bob123", "customer_id": "c-bob", "roles": ["shopper"]},
    "agent": {"password": "agent123", "customer_id": None, "roles": ["agent"]},
}


@dataclass
class User:
    sub: str
    customer_id: str | None
    roles: list[str] = field(default_factory=list)


def authenticate(username: str, password: str) -> User | None:
    record = DEMO_USERS.get(username)
    if record is None or not hmac.compare_digest(record["password"], password):
        return None
    return User(sub=username, customer_id=record["customer_id"], roles=record["roles"])


def issue_token(user: User, settings: Settings) -> str:
    now = int(time.time())
    claims = {
        "sub": user.sub,
        "cid": user.customer_id,
        "roles": user.roles,
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": now + settings.jwt_ttl_minutes * 60,
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str, settings: Settings) -> User:
    claims = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"], issuer=settings.jwt_issuer)
    return User(sub=claims["sub"], customer_id=claims.get("cid"), roles=claims.get("roles", []))


_bearer = HTTPBearer(auto_error=False)


async def current_user(
    request: Request, creds: HTTPAuthorizationCredentials | None = Depends(_bearer)
) -> User:
    if creds is None:
        raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    try:
        return decode_token(creds.credentials, request.app.state.settings)
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"})


def require_role(role: str):
    async def dependency(user: User = Depends(current_user)) -> User:
        if role not in user.roles:
            raise HTTPException(403, f"requires role '{role}'")
        return user

    return dependency
