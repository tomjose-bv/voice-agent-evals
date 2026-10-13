"""Stop Windows from sleeping (and the display from turning off) during long eval runs; no-op elsewhere.

Uses SetThreadExecutionState, which only lasts while this process runs and
does not change the user's power settings.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
# Laptops with Modern Standby still suspend when the display turns off, so keep it on too.
ES_DISPLAY_REQUIRED = 0x00000002


@contextmanager
def keep_awake() -> Iterator[None]:
    if sys.platform != "win32":
        yield
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
