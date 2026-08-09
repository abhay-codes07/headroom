"""Return freed-but-retained allocator pages to the OS on long-lived proxies.

Large concurrent Anthropic bodies (0.5-1 MB of JSON parsed, deep-copied and
re-serialized per in-flight request) drive libmalloc and pymalloc to a
high-water mark that is never returned to the OS: after a burst the malloc
zones keep entire regions resident but empty (``vmmap`` lists them as
``MALLOC_LARGE (empty)`` / ``MALLOC_SMALL (empty)``), so process RSS only
ratchets upward. Over a multi-day proxy lifetime under Claude Code traffic
this reaches double-digit GB and starves the host.

Neither runtime returns these pages on its own. macOS exposes
``malloc_zone_pressure_relief(NULL, 0)`` to purge every zone's free pages;
glibc has ``malloc_trim(0)``. ``trim()`` runs a full ``gc.collect()`` first so
obmalloc can unmap empty arenas and freed CPython blocks reach the allocator
free lists before the purge.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import logging
import sys
import time

logger = logging.getLogger(__name__)

# Lazily resolved (platform_tag, foreign_function | None). ``None`` function
# means the platform has no supported trim call and trim() is a no-op.
_relief: tuple[str, object | None] | None = None


def _resolve() -> tuple[str, object | None]:
    global _relief
    if _relief is not None:
        return _relief
    try:
        libc = ctypes.CDLL(None)
        if sys.platform == "darwin":
            fn = libc.malloc_zone_pressure_relief
            fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            fn.restype = ctypes.c_size_t
            _relief = ("darwin", fn)
        else:
            fn = libc.malloc_trim
            fn.argtypes = [ctypes.c_size_t]
            fn.restype = ctypes.c_int
            _relief = ("glibc", fn)
    except (OSError, AttributeError):
        _relief = ("unsupported", None)
    return _relief


def trim() -> int:
    """Collect garbage and return allocator free pages to the OS.

    Returns the number of bytes freed on macOS (glibc's ``malloc_trim``
    reports only success, so 0 is returned there and on unsupported
    platforms).
    """
    kind, fn = _resolve()
    if fn is None:
        return 0
    gc.collect()
    if kind == "darwin":
        return int(fn(None, 0))  # type: ignore[operator]
    fn(0)  # type: ignore[operator]
    return 0


async def trim_periodically(interval_seconds: int = 60) -> None:
    """Background task that periodically returns allocator free pages to the OS.

    Runs in every worker process (allocator state is per-process). The
    ``gc.collect()`` inside ``trim()`` holds the GIL for the duration of the
    collection, so the pause lands between requests just as any other
    collection would.

    Args:
        interval_seconds: How often to trim (default: 60 seconds).
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            start = time.perf_counter()
            freed = trim()
            elapsed_ms = (time.perf_counter() - start) * 1000
            log = logger.info if freed >= (16 << 20) else logger.debug
            log(
                "MallocTrim: returned %.1f MB to OS in %.0f ms",
                freed / 1048576,
                elapsed_ms,
            )
        except Exception as e:
            logger.debug("MallocTrim failed: %s", e)
