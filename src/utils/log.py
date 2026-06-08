"""Logging and timing utilities for experiment scripts."""

import time
from collections.abc import Generator
from contextlib import contextmanager


def log(message: str = "") -> None:
    """Print with immediate flush to ensure output is visible during long runs."""
    print(message, flush=True)


def log_header(title: str) -> None:
    """Print a clearly delimited section header."""
    log()
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)


def log_step(step: int, total: int, message: str) -> None:
    """Print a numbered progress step."""
    log(f"[{step}/{total}] {message}")


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string like '1h 4m 32s'.

    Returns a short string suitable for log output.
    """
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


@contextmanager
def log_timer(label: str) -> Generator[None, None, None]:
    """Context manager that logs the wall-clock duration of a code block.

    Logs a line like: ``    [TIMER] generation: 1h 4m 32s``
    """
    start = time.monotonic()
    yield
    log(f"    [TIMER] {label}: {_fmt_duration(time.monotonic() - start)}")
