import time
from typing import Any, Iterable, Iterator, TypeVar

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
    logger: Any,
    label: str,
    stride: int = 1,
) -> Iterator[T]:
    """Yield items while logging coarse progress updates."""
    if stride < 1:
        raise ValueError("'stride' must be a positive integer.")
    if total < 0:
        raise ValueError("'total' must be non-negative.")
    if total == 0:
        return

    start_time = time.time()
    pad = len(str(total))
    template = (
        f"{label}: " + "{i:>{pad}}/{total} "
        "({percent:3.0f}%) | {elapsed} < {eta}"
    )

    logger.info(
        template.format(
            i=0,
            pad=pad,
            total=total,
            percent=0,
            elapsed="0s",
            eta="0s",
        ),
        level=1,
        overwrite=True,
    )

    for i, item in enumerate(iterable, start=1):
        yield item

        if (i % stride != 0) and (i != total):
            continue

        elapsed_time = time.time() - start_time
        eta = 0.0 if i == 0 else (elapsed_time / i) * max(total - i, 0)
        logger.info(
            template.format(
                i=i,
                pad=pad,
                total=total,
                percent=100 * i / total if total else 100,
                elapsed=format_time(elapsed_time),
                eta=format_time(eta),
            ),
            level=1,
            overwrite=True,
        )
