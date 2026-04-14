"""Centralised logging configuration for the census package.

Call ``configure_logging()`` exactly once, at the entrypoint of each
executable context (``__main__`` blocks, CLI scripts, the FastAPI startup
event).  Never call it at module import time — doing so adds loguru handlers
as a side effect whenever any module is imported, which:

    - pollutes test output with unexpected log lines,
    - duplicates handlers when the same module is imported multiple times,
    - makes the package harder to embed as a library.

Modules that only *emit* log messages should simply do::

    from loguru import logger

and let the entrypoint decide whether and where to route those messages.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# ── Format strings ────────────────────────────────────────────────────────────

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level:<8} | "
    "{name}:{function}:{line} — "
    "{message}"
)


def configure_logging(
    log_dir: Path,
    filename_stem: str,
    *,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> None:
    """Configure loguru for a specific entrypoint.

    Removes *all* existing handlers (including loguru's default stderr sink)
    and installs two fresh ones:

    - **Console** — writes to stderr at *console_level* with colour.
    - **File** — writes to ``log_dir/<filename_stem>_<date>.log`` at
      *file_level*; rotated daily, retained for 7 days.

    Calling this function more than once replaces the previous handlers, so
    it is safe to call in tests that need a different configuration.

    Args:
        log_dir:       Directory where log files will be written.
                       Created automatically if it does not exist.
        filename_stem: Prefix for the log filename, e.g. ``"data"`` produces
                       ``data_2026-04-14.log``.
        console_level: Minimum level for the stderr sink. Default ``"INFO"``.
        file_level:    Minimum level for the file sink. Default ``"DEBUG"``.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()  # drop all existing handlers (including loguru's default)

    logger.add(
        sys.stderr,
        level=console_level,
        format=_CONSOLE_FORMAT,
        colorize=True,
    )
    logger.add(
        log_dir / f"{filename_stem}_{{time:YYYY-MM-DD}}.log",
        level=file_level,
        format=_FILE_FORMAT,
        rotation="1 day",
        retention="7 days",
        encoding="utf-8",
    )

    logger.debug(
        "configure_logging | stem={} | log_dir={} | console={} | file={}",
        filename_stem,
        log_dir,
        console_level,
        file_level,
    )
