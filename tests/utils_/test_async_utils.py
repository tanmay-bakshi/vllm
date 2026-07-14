# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
from collections.abc import AsyncIterator

import pytest

from vllm.utils.async_utils import ManagedAsyncIterator, merge_async_iterators


class _CloseRecorder:
    """Record deterministic asynchronous closure for lifecycle tests."""

    close_count: int
    close_error: BaseException | None

    def __init__(self, close_error: BaseException | None = None) -> None:
        """Create a closure recorder.

        :param close_error: Optional error raised after recording closure.
        """
        self.close_count = 0
        self.close_error = close_error

    async def aclose(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


async def _mock_async_iterator(idx: int):
    try:
        while True:
            yield f"item from iterator {idx}"
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        print(f"iterator {idx} cancelled")


@pytest.mark.asyncio
async def test_merge_async_iterators():
    iterators = [_mock_async_iterator(i) for i in range(3)]
    merged_iterator = merge_async_iterators(*iterators)

    async def stream_output(generator: AsyncIterator[tuple[int, str]]):
        async for idx, output in generator:
            print(f"idx: {idx}, output: {output}")

    task = asyncio.create_task(stream_output(merged_iterator))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for iterator in iterators:
        try:
            await asyncio.wait_for(anext(iterator), 1)
        except StopAsyncIteration:
            # All iterators should be cancelled and print this message.
            print("Iterator was cancelled normally")
        except (Exception, asyncio.CancelledError) as e:
            raise AssertionError() from e


@pytest.mark.asyncio
async def test_merge_async_iterators_closes_every_iterator_on_early_exit():
    closed = [asyncio.Event() for _ in range(3)]

    async def iterator(index: int) -> AsyncIterator[int]:
        try:
            yield index
            await asyncio.Event().wait()
        finally:
            closed[index].set()

    iterators = [iterator(index) for index in range(3)]
    merged_iterator = merge_async_iterators(*iterators)
    async with contextlib.aclosing(merged_iterator):
        async for _ in merged_iterator:
            break

    assert all(event.is_set() for event in closed)


@pytest.mark.asyncio
async def test_managed_async_iterator_closes_unstarted_owners_once() -> None:
    owner = _CloseRecorder()

    async def transform() -> AsyncIterator[str]:
        yield "unused"

    iterator = ManagedAsyncIterator(transform(), (owner,))

    await iterator.aclose()
    await iterator.aclose()

    assert owner.close_count == 1


@pytest.mark.asyncio
async def test_managed_async_iterator_preserves_active_iteration_error() -> None:
    owner = _CloseRecorder(RuntimeError("close failed"))

    async def transform() -> AsyncIterator[str]:
        raise ValueError("iteration failed")
        yield "unreachable"

    iterator = ManagedAsyncIterator(transform(), (owner,))

    with pytest.raises(ValueError, match="iteration failed"):
        await anext(iterator)

    assert owner.close_count == 1
