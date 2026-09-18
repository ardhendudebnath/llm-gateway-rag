"""Distributed token-bucket rate limiter.

The refill-and-take is a single Lua script, so it is atomic across any number of API replicas
sharing one Redis — no read-modify-write race between concurrent requests for the same key.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis

# KEYS[1]=bucket  ARGV: capacity, refill_per_sec, now, cost
# Floats are returned as strings: Redis truncates Lua numbers to integers in replies.
_TOKEN_BUCKET_LUA = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  ts = now
end

tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = (cost - tokens) / rate
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / rate) + 60)
return {allowed, tostring(tokens), tostring(retry_after)}
"""


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: int
    limit: int


class TokenBucketLimiter:
    def __init__(self, redis: Redis, clock: Callable[[], float] = time.time):
        self._redis = redis
        self._clock = clock
        self._script = redis.register_script(_TOKEN_BUCKET_LUA)

    async def hit(
        self, key: str, *, capacity: int, refill_per_sec: float, cost: int = 1
    ) -> RateLimitDecision:
        allowed, tokens, retry_after = await self._script(
            keys=[f"ratelimit:{key}"], args=[capacity, refill_per_sec, self._clock(), cost]
        )
        return RateLimitDecision(
            allowed=bool(int(allowed)),
            remaining=math.floor(float(tokens)),
            retry_after_seconds=math.ceil(float(retry_after)),
            limit=capacity,
        )
