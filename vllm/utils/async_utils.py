# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Contains helpers related to asynchronous code.

This is similar in concept to the `asyncio` module.
"""

import asyncio
import contextlib
import traceback
from asyncio import FIRST_COMPLETED, AbstractEventLoop, Future, Task
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from typing_extensions import ParamSpec

from vllm.logger import init_logger

logger = init_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class AsyncCloseable(Protocol):
    """Asynchronous resource supporting deterministic closure."""

    async def aclose(self) -> None:
        """Close the resource."""


class CloseableAsyncIterator(Protocol[T_co]):
    """Asynchronous iterator supporting deterministic closure."""

    def __aiter__(self) -> AsyncIterator[T_co]: ...

    async def __anext__(self) -> T_co: ...

    async def aclose(self) -> None: ...


class ManagedAsyncIterator(Generic[T_co]):
    """Asynchronous iterator that owns its complete resource lifecycle."""

    _iterator: CloseableAsyncIterator[T_co]
    _owners: tuple[AsyncCloseable, ...]
    _closed: bool

    def __init__(
        self,
        iterator: CloseableAsyncIterator[T_co],
        owners: tuple[AsyncCloseable, ...],
    ) -> None:
        """Create a managed transformation over eagerly acquired resources.

        :param iterator: Iterator that transforms the owned resources.
        :param owners: Resources acquired before the iterator is first polled.
        """
        self._iterator = iterator
        self._owners = owners
        self._closed = False

    def __aiter__(self) -> "ManagedAsyncIterator[T_co]":
        return self

    async def __anext__(self) -> T_co:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException:
            await self._close_preserving_active_exception()
            raise

    async def aclose(self) -> None:
        """Close the transformation and every eagerly acquired resource."""
        if self._closed:
            return
        self._closed = True

        first_error: BaseException | None = None
        try:
            await self._iterator.aclose()
        except BaseException as error:
            first_error = error
            logger.error(
                "Failed to close a managed asynchronous iterator\n%s",
                traceback.format_exc(),
            )

        for owner in self._owners:
            try:
                await owner.aclose()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                logger.error(
                    "Failed to close a managed asynchronous resource\n%s",
                    traceback.format_exc(),
                )

        if first_error is not None:
            raise first_error

    async def _close_preserving_active_exception(self) -> None:
        try:
            await self.aclose()
        except BaseException:
            logger.error(
                "Failed to close a managed asynchronous iterator while handling "
                "another error\n%s",
                traceback.format_exc(),
            )


def cancel_task_threadsafe(task: Task):
    if task and not task.done():
        run_in_loop(task.get_loop(), task.cancel)


def make_async(
    func: Callable[P, T],
    executor: Executor | None = None,
) -> Callable[P, Awaitable[T]]:
    """
    Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Future[T]:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=executor, func=p_func)

    return _async_wrapper


def make_async_with_semaphore(
    func: Callable[P, T],
    executor: ThreadPoolExecutor,
) -> Callable[P, Awaitable[T]]:
    """
    Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.

    The function is wrapped in a semaphore to limit the number of
    concurrent executions making it easier to cancel tasks before they start.
    """

    semaphore = asyncio.Semaphore(executor._max_workers)

    async def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        async with semaphore:
            return await loop.run_in_executor(executor, p_func)

    return _async_wrapper


def run_in_loop(loop: AbstractEventLoop, function: Callable, *args):
    if in_loop(loop):
        function(*args)
    elif not loop.is_closed():
        loop.call_soon_threadsafe(function, *args)


def in_loop(event_loop: AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() == event_loop
    except RuntimeError:
        return False


# A hack to pass mypy
if TYPE_CHECKING:

    def anext(it: CloseableAsyncIterator[T]):
        return it.__anext__()


async def merge_async_iterators(
    *iterators: CloseableAsyncIterator[T],
) -> AsyncGenerator[tuple[int, T], None]:
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.
    """
    if len(iterators) == 1:
        iterator = iterators[0]
        try:
            async for item in iterator:
                yield 0, item
        finally:
            await iterator.aclose()
        return

    loop = asyncio.get_running_loop()

    awaits = {loop.create_task(anext(it)): (i, it) for i, it in enumerate(iterators)}
    try:
        while len(awaits) > 0:
            done, _ = await asyncio.wait(awaits.keys(), return_when=FIRST_COMPLETED)
            for d in done:
                pair = awaits.pop(d)
                try:
                    item = await d
                    i, it = pair
                    awaits[loop.create_task(anext(it))] = pair
                    yield i, item
                except StopAsyncIteration:
                    pass
    finally:
        pending_tasks = list(awaits)
        for task in pending_tasks:
            task.cancel()
        if len(pending_tasks) > 0:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        for iterator in iterators:
            with contextlib.suppress(BaseException):
                await iterator.aclose()


async def collect_from_async_generator(
    iterator: CloseableAsyncIterator[T],
) -> list[T]:
    """Collect all items from an async generator into a list."""
    items: list[T] = []
    try:
        async for item in iterator:
            items.append(item)
    finally:
        await iterator.aclose()
    return items
