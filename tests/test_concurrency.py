"""Tests for the bounded-concurrency async helper."""

import asyncio
import threading

from testagent.engine.concurrency import gather_with_concurrency


async def test_gather_preserves_order() -> None:
    """Results must come back in input order regardless of completion order."""

    async def make(x: int) -> int:
        return x * 2

    result = await gather_with_concurrency(2, make(1), make(2), make(3))
    assert result == [2, 4, 6]


async def test_gather_unbounded_when_none() -> None:
    """``n=None`` runs every coroutine concurrently."""
    counter = {"max": 0, "cur": 0}
    lock = threading.Lock()

    async def work(i: int) -> int:
        with lock:
            counter["cur"] += 1
            counter["max"] = max(counter["max"], counter["cur"])
        await asyncio.sleep(0.02)
        with lock:
            counter["cur"] -= 1
        return i

    result = await gather_with_concurrency(None, *(work(i) for i in range(5)))
    assert result == [0, 1, 2, 3, 4]
    # All 5 ran at once when unbounded.
    assert counter["max"] == 5


async def test_gather_bounded_by_semaphore() -> None:
    """``n=2`` caps concurrent execution at 2 even with many coroutines."""
    counter = {"max": 0, "cur": 0}
    lock = threading.Lock()

    async def work(i: int) -> int:
        with lock:
            counter["cur"] += 1
            counter["max"] = max(counter["max"], counter["cur"])
        await asyncio.sleep(0.05)
        with lock:
            counter["cur"] -= 1
        return i

    result = await gather_with_concurrency(2, *(work(i) for i in range(6)))
    assert result == [0, 1, 2, 3, 4, 5]
    assert counter["max"] == 2


async def test_gather_empty() -> None:
    """No coroutines yields an empty list (no error)."""
    assert await gather_with_concurrency(3) == []
