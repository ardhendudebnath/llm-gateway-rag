import time

import jwt
import pytest

from app.core.security import ApiKeyStore, InvalidCredentials, TokenService, hash_key

SECRET = "unit-test-secret-that-is-at-least-32-bytes"


@pytest.fixture
def store(redis_pair):
    return ApiKeyStore(redis_pair[0])


async def test_created_key_verifies_and_only_hash_is_stored(store, redis_pair):
    raw_key, record = await store.create("acme", "ci")
    assert raw_key.startswith(f"ng_{record.key_id}_")
    assert (await store.verify(raw_key)).tenant_id == "acme"

    r = redis_pair[0]
    for key in await r.keys("*"):
        assert raw_key not in key
        kind = await r.type(key)
        values = {
            "string": lambda k: [r.get(k)],
            "hash": lambda k: [r.hvals(k)],
            "set": lambda k: [r.smembers(k)],
        }[kind](key)
        for v in values:
            assert raw_key not in str(await v)
    assert await r.get(f"apikey_hash:{hash_key(raw_key)}") == record.key_id


@pytest.mark.parametrize("bad", ["", "sk-openai-style", "ng_deadbeef_wrong"])
async def test_unknown_or_malformed_keys_are_rejected(store, bad):
    await store.create("acme", "ci")
    with pytest.raises(InvalidCredentials):
        await store.verify(bad)


async def test_revoke_invalidates_key(store):
    raw_key, record = await store.create("acme", "ci")
    assert await store.revoke(record.key_id) is True
    with pytest.raises(InvalidCredentials):
        await store.verify(raw_key)
    assert await store.list_for_tenant("acme") == []
    assert await store.revoke(record.key_id) is False


async def test_list_for_tenant(store):
    await store.create("acme", "one")
    await store.create("acme", "two")
    await store.create("globex", "three")
    assert sorted(r.name for r in await store.list_for_tenant("acme")) == ["one", "two"]


async def test_jwt_roundtrip(store):
    _, record = await store.create("acme", "ci")
    tokens = TokenService(SECRET, "HS256", 60)
    claims = tokens.decode(tokens.issue(record))
    assert claims["sub"] == record.key_id
    assert claims["tenant"] == "acme"


async def test_expired_jwt_rejected():
    tokens = TokenService(SECRET, "HS256", 60)
    expired = jwt.encode(
        {"sub": "k", "iss": "nexusgate", "exp": int(time.time()) - 10}, SECRET, algorithm="HS256"
    )
    with pytest.raises(InvalidCredentials, match="expired"):
        tokens.decode(expired)


@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        jwt.encode(
            {"sub": "k", "iss": "nexusgate", "exp": 9999999999}, "wrong-secret-32-bytes-long-xxxxx"
        ),
        jwt.encode({"sub": "k", "iss": "someone-else", "exp": 9999999999}, SECRET),
    ],
)
def test_forged_or_foreign_jwt_rejected(token):
    with pytest.raises(InvalidCredentials):
        TokenService(SECRET, "HS256", 60).decode(token)
