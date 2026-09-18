"""API-key issuance/verification and JWT exchange.

* Keys look like ``ng_<key_id>_<secret>``. Only a SHA-256 of the full key is stored, so a Redis
  dump never leaks usable credentials. The ``key_id`` prefix is public and safe to log.
* Clients may authenticate with ``X-API-Key: <key>``, ``Authorization: Bearer <key>`` (so the
  stock OpenAI SDK works unchanged), or ``Authorization: Bearer <jwt>`` obtained from
  ``POST /v1/auth/token``.
* JWTs are short-lived, but still checked against the key record so revoking a key revokes its
  outstanding tokens immediately.
"""

import hashlib
import hmac
import secrets
import time
from datetime import UTC, datetime

import jwt
from pydantic import BaseModel, Field
from redis.asyncio import Redis

KEY_PREFIX = "ng_"
KEY_HASHES = "apikey_hashes"


class ApiKeyRecord(BaseModel):
    key_id: str
    tenant_id: str
    name: str
    created_at: datetime
    rate_limit_capacity: int | None = Field(default=None, gt=0)
    rate_limit_refill_per_sec: float | None = Field(default=None, gt=0)


class Principal(BaseModel):
    key_id: str
    tenant_id: str
    rate_limit_capacity: int | None = None
    rate_limit_refill_per_sec: float | None = None


class InvalidCredentials(Exception):
    pass


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _record_key(key_id: str) -> str:
    return f"apikey:{key_id}"


def _hash_index_key(key_hash: str) -> str:
    return f"apikey_hash:{key_hash}"


def _tenant_index_key(tenant_id: str) -> str:
    return f"tenant_keys:{tenant_id}"


class ApiKeyStore:
    def __init__(self, redis: Redis):
        self._redis = redis

    async def create(
        self,
        tenant_id: str,
        name: str,
        rate_limit_capacity: int | None = None,
        rate_limit_refill_per_sec: float | None = None,
    ) -> tuple[str, ApiKeyRecord]:
        """Create a key. The raw key is returned exactly once and never stored."""
        key_id = secrets.token_hex(6)
        raw_key = f"{KEY_PREFIX}{key_id}_{secrets.token_urlsafe(32)}"
        record = ApiKeyRecord(
            key_id=key_id,
            tenant_id=tenant_id,
            name=name,
            created_at=datetime.now(UTC),
            rate_limit_capacity=rate_limit_capacity,
            rate_limit_refill_per_sec=rate_limit_refill_per_sec,
        )
        key_hash = hash_key(raw_key)
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(_record_key(key_id), record.model_dump_json())
        pipe.set(_hash_index_key(key_hash), key_id)
        pipe.hset(KEY_HASHES, key_id, key_hash)  # back-reference so revoke can drop the index
        pipe.sadd(_tenant_index_key(tenant_id), key_id)
        await pipe.execute()
        return raw_key, record

    async def get(self, key_id: str) -> ApiKeyRecord | None:
        raw = await self._redis.get(_record_key(key_id))
        return ApiKeyRecord.model_validate_json(raw) if raw else None

    async def verify(self, raw_key: str) -> ApiKeyRecord:
        if not raw_key.startswith(KEY_PREFIX):
            raise InvalidCredentials("malformed API key")
        key_id = await self._redis.get(_hash_index_key(hash_key(raw_key)))
        if key_id is None:
            raise InvalidCredentials("unknown or revoked API key")
        record = await self.get(key_id)
        if record is None:
            raise InvalidCredentials("unknown or revoked API key")
        return record

    async def list_for_tenant(self, tenant_id: str) -> list[ApiKeyRecord]:
        key_ids = sorted(await self._redis.smembers(_tenant_index_key(tenant_id)))
        records = [await self.get(k) for k in key_ids]
        return [r for r in records if r is not None]

    async def revoke(self, key_id: str) -> bool:
        record = await self.get(key_id)
        if record is None:
            return False
        key_hash = await self._redis.hget(KEY_HASHES, key_id)
        pipe = self._redis.pipeline(transaction=True)
        pipe.delete(_record_key(key_id))
        if key_hash:
            pipe.delete(_hash_index_key(key_hash))
        pipe.hdel(KEY_HASHES, key_id)
        pipe.srem(_tenant_index_key(record.tenant_id), key_id)
        await pipe.execute()
        return True


class TokenService:
    def __init__(self, secret: str, algorithm: str, ttl_seconds: int):
        self._secret = secret
        self._algorithm = algorithm
        self.ttl_seconds = ttl_seconds

    def issue(self, record: ApiKeyRecord) -> str:
        now = int(time.time())
        claims = {
            "sub": record.key_id,
            "tenant": record.tenant_id,
            "iat": now,
            "exp": now + self.ttl_seconds,
            "iss": "nexusgate",
        }
        return jwt.encode(claims, self._secret, algorithm=self._algorithm)

    def decode(self, token: str) -> dict:
        try:
            return jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                issuer="nexusgate",
                options={"require": ["sub", "exp", "iss"]},
            )
        except jwt.ExpiredSignatureError as e:
            raise InvalidCredentials("token expired") from e
        except jwt.PyJWTError as e:
            raise InvalidCredentials("invalid token") from e


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def principal_from(record: ApiKeyRecord) -> Principal:
    return Principal(
        key_id=record.key_id,
        tenant_id=record.tenant_id,
        rate_limit_capacity=record.rate_limit_capacity,
        rate_limit_refill_per_sec=record.rate_limit_refill_per_sec,
    )
