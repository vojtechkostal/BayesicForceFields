"""Progress reporting for long-running loops."""

import time
from typing import Iterable, Iterator, TypeVar

from .logs import Logger

T = TypeVar("T")


def format_time(seconds: float) -> str:
    """Format seconds as hours, minutes, and seconds."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    if minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    return f"{int(seconds)}s"


def iter_progress(
    iterable: Iterable[T],
    *,
    total: int,
    logger: Logger,
    label: str,
    log_every: int = 100,
) -> Iterator[T]:
    """Yield items while reporting progress.

    On a terminal, a progress line is overwritten after every item. Every
    ``log_every`` items, and at the end, a line is also written to the log
    file, which is the only progress a batch job's console shows.
    """
    if total < 0:
        raise ValueError("'total' must be non-negative.")
    if log_every < 1:
        raise ValueError("'log_every' must be at least 1.")
    if total == 0:
        return

    start_time = time.time()
    for i, item in enumerate(iterable, start=1):
        yield item
        if i == total:
            continue
        logged = i % log_every == 0
        if not (logged or logger.interactive):
            continue
        elapsed_time = time.time() - start_time
        eta = (elapsed_time / i) * (total - i)
        logger.progress_status(
            f"{label}: {i}/{total} | "
            f"{format_time(elapsed_time)} < {format_time(eta)}",
            i,
            total,
            overwrite=True,
            write_file=logged,
        )

    elapsed_time = time.time() - start_time
    logger.done(label, detail=f"{total}/{total} in {format_time(elapsed_time)}")
