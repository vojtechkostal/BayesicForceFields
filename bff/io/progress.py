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
    stride: int = 1,
) -> Iterator[T]:
    """Yield items while logging pytest-style progress updates."""
    if stride < 1:
        raise ValueError("'stride' must be a positive integer.")
    if total < 0:
        raise ValueError("'total' must be non-negative.")
    if total == 0:
        return

    start_time = time.time()

    for i, item in enumerate(iterable, start=1):
        yield item

        if (i % stride != 0) and (i != total):
            continue

        elapsed_time = time.time() - start_time
        eta = 0.0 if i == 0 else (elapsed_time / i) * max(total - i, 0)
        detail = (
            f"{label}: {i}/{total} | "
            f"{format_time(elapsed_time)} < {format_time(eta)}"
        )
        logger.progress_status(detail, i, total, level=0)

    elapsed_time = time.time() - start_time
    logger.result_summary(total, "completed", elapsed_time)
