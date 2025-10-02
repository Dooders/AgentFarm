from __future__ import annotations

"""
Simple timing instrumentation.

Provides a context manager that records wall-clock duration of a block.
"""

import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def time_block(metrics: Dict[str, float], key: str = "duration_s"):
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics[key] = time.perf_counter() - start

