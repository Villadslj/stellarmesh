"""Progress logging helpers for long-running operations."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from threading import Event, Thread
from time import monotonic
from typing import Iterator, Optional


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds as a compact human-readable duration."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{seconds:.1f}s"


def default_progress_interval() -> float:
    """Return the heartbeat interval configured by the environment."""
    return float(os.environ.get("STELLARMESH_PROGRESS_INTERVAL", "60"))


@contextmanager
def progress_heartbeat(
    logger: logging.Logger,
    operation: str,
    interval: Optional[float] = None,
) -> Iterator[None]:
    """Log elapsed-time heartbeats while a blocking operation is running."""
    if interval is None:
        interval = default_progress_interval()
    if interval <= 0 or not logger.isEnabledFor(logging.INFO):
        yield
        return

    start = monotonic()
    finished = Event()

    def report() -> None:
        while not finished.wait(interval):
            logger.info(
                "%s is still running (elapsed %s).",
                operation,
                _format_duration(monotonic() - start),
            )

    logger.info("%s started.", operation)
    thread = Thread(target=report, name="stellarmesh-progress", daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        logger.error(
            "%s failed after %s.",
            operation,
            _format_duration(monotonic() - start),
        )
        raise
    else:
        logger.info(
            "%s completed in %s.",
            operation,
            _format_duration(monotonic() - start),
        )
    finally:
        finished.set()
        thread.join()


def log_progress(
    logger: logging.Logger,
    operation: str,
    completed: int,
    total: int,
) -> None:
    """Log the first item, each 10% boundary, and completion."""
    if total <= 0 or not logger.isEnabledFor(logging.INFO):
        return
    previous_decile = ((completed - 1) * 10) // total
    current_decile = (completed * 10) // total
    if completed == 1 or completed == total or current_decile > previous_decile:
        logger.info(
            "%s: %d/%d (%.0f%%).",
            operation,
            completed,
            total,
            100 * completed / total,
        )
