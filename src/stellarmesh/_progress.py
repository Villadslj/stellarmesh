"""Progress logging helpers for long-running operations."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from contextlib import contextmanager
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
    value = os.environ.get("STELLARMESH_PROGRESS_INTERVAL", "60")
    try:
        return float(value)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid STELLARMESH_PROGRESS_INTERVAL=%r; using 60 seconds.",
            value,
        )
        return 60.0


@contextmanager
def progress_heartbeat(
    logger: logging.Logger,
    operation: str,
    interval: Optional[float] = None,
    *,
    independent: bool = False,
) -> Iterator[None]:
    """Log elapsed-time heartbeats while a blocking operation is running."""
    if interval is None:
        interval = default_progress_interval()
    if interval <= 0 or not logger.isEnabledFor(logging.INFO):
        yield
        return

    start = monotonic()
    finished = Event()
    process = None

    def report() -> None:
        while not finished.wait(interval):
            logger.info(
                "%s is still running (elapsed %s).",
                operation,
                _format_duration(monotonic() - start),
            )

    logger.info("%s started.", operation)
    if independent:
        script = (
            "import sys,time\n"
            "operation=sys.argv[1]; interval=float(sys.argv[2]); "
            "start=time.monotonic()\n"
            "while True:\n"
            " time.sleep(interval)\n"
            " elapsed=int(time.monotonic()-start); "
            "h,r=divmod(elapsed,3600); m,s=divmod(r,60)\n"
            " duration=(f'{h}h {m:02d}m {s:02d}s' if h else "
            "(f'{m}m {s:02d}s' if m else f'{elapsed}s'))\n"
            " print(f'Stellarmesh: {operation} is still running "
            "(elapsed {duration}).',file=sys.stderr,flush=True)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", script, operation, str(interval)]
        )
        thread = None
    else:
        thread = Thread(target=report, name="stellarmesh-progress", daemon=True)
        thread.start()
    try:
        yield
    except BaseException:
        logger.exception(
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
        if process is not None:
            process.terminate()
            process.wait()
        elif thread is not None:
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
    if completed in (1, total) or current_decile > previous_decile:
        logger.info(
            "%s: %d/%d (%.0f%%).",
            operation,
            completed,
            total,
            100 * completed / total,
        )
