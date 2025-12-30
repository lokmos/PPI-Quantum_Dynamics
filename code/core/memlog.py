"""
Lightweight memory logger for PyFlow.

Writes JSON Lines records to a file path specified by env var PYFLOW_MEMLOG_FILE.
This is designed to be extremely low-overhead when disabled (default).

Usage:
  - Set PYFLOW_MEMLOG=1 to enable (main scripts can set PYFLOW_MEMLOG_FILE automatically)
  - Or set PYFLOW_MEMLOG_FILE=/path/to/file.jsonl to force a log file.
  - Optional: PYFLOW_MEMLOG_FLUSH=1 to flush each write.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

_fh = None
_path = None
_flush = False


def _enabled() -> bool:
    if os.environ.get("PYFLOW_MEMLOG_FILE"):
        return True
    v = os.environ.get("PYFLOW_MEMLOG", "0")
    return v in ("1", "true", "True", "yes", "YES", "on", "ON")


def _rss_bytes() -> Optional[int]:
    # Prefer psutil if available.
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    # Fallback: resource (ru_maxrss is peak; platform-dependent units).
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux ru_maxrss is KiB.
        return int(r) * 1024
    except Exception:
        return None


def _ensure_open() -> bool:
    global _fh, _path, _flush
    if not _enabled():
        return False

    path = os.environ.get("PYFLOW_MEMLOG_FILE")
    if not path:
        return False

    if _fh is not None and _path == path:
        return True

    # (Re)open
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        # If directory creation fails, we still try to open (may be relative path).
        pass

    _fh = open(path, "a", encoding="utf-8")
    _path = path
    _flush = os.environ.get("PYFLOW_MEMLOG_FLUSH", "0") in ("1", "true", "True")
    return True


def memlog(tag: str, step: Optional[int] = None, l: Optional[float] = None, **fields: Any) -> None:
    """
    Append one JSONL record with RSS memory and optional metadata.
    """
    if not _ensure_open():
        return

    rec: dict[str, Any] = {
        "t_wall": time.time(),
        "t_mono": time.monotonic(),
        "pid": os.getpid(),
        "tag": tag,
    }
    if step is not None:
        rec["step"] = int(step)
    if l is not None:
        rec["l"] = float(l)

    rss = _rss_bytes()
    if rss is not None:
        rec["rss_bytes"] = int(rss)
        rec["rss_mb"] = float(rss) / (1024.0 * 1024.0)

    if fields:
        rec.update(fields)

    _fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if _flush:
        _fh.flush()


def close() -> None:
    global _fh, _path
    if _fh is None:
        return
    try:
        _fh.flush()
        _fh.close()
    finally:
        _fh = None
        _path = None


