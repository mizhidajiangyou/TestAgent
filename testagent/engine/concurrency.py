"""Bounded-concurrency async helpers.

This mirrors LangChain's ``gather_with_concurrency`` primitive: when an
explicit limit ``n`` is given, a :class:`asyncio.Semaphore` gates execution so
at most ``n`` coroutines run at once; otherwise all coroutines run
concurrently via :func:`asyncio.gather`. Results are returned in the input
order, so callers can rely on positional correspondence with their inputs.
"""

import asyncio
from collections.abc import Awaitable


async def gather_with_concurrency[T](n: int | None, *coros: Awaitable[T]) -> list[T]:
    """Run ``coros`` with at most ``n`` running concurrently, preserving order.

    Args:
        n: Maximum number of coroutines to run at once. ``None`` or a value
            less than 1 means unbounded (equivalent to ``asyncio.gather``).
        *coros: The awaitables to run.

    Returns:
        A list of results in the same order as ``coros``.
    """
    if not coros:
        return []

    if n is None or n < 1:
        return list(await asyncio.gather(*coros))

    sem = asyncio.Semaphore(n)

    async def _gate(coro: Awaitable[T]) -> T:
        async with sem:
            return await coro

    return list(await asyncio.gather(*(_gate(c) for c in coros)))
