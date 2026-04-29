"""Structured logging for VibeFinder AI.

Sets up a root "vibefinder" logger with:
  - JSON-formatted rotating file handler  (logs/vibefinder.log)
  - Human-readable console handler        (stdout)

Call setup_logging() once at startup; subsequent calls are no-ops.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from datetime import datetime, timezone

_SKIP_FIELDS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }
)


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _SKIP_FIELDS:
                continue
            try:
                json.dumps(value)
                data[key] = value
            except (TypeError, ValueError):
                data[key] = str(value)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(log_dir: str = "logs") -> None:
    """Configure the root 'vibefinder' logger. Safe to call multiple times."""
    root = logging.getLogger("vibefinder")
    if root.handlers:
        return  # already configured (handles Streamlit hot-reloads)

    os.makedirs(log_dir, exist_ok=True)
    root.setLevel(logging.DEBUG)

    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "vibefinder.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JSONFormatter())

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )

    root.addHandler(fh)
    root.addHandler(ch)
