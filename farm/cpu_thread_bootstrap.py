"""Pre-import CPU math-thread pinning.

This module pins the thread-pool environment variables consulted by the CPU
math backends (OpenMP, MKL, OpenBLAS, NumExpr, Accelerate/vecLib) *before* those
libraries are imported and initialize their pools. Once ``numpy``/``torch`` (and
the BLAS libraries underneath them) have imported, these variables are read-only
no-ops for the current process, so timing matters: call ``pin_cpu_math_threads``
from the very top of a process entry point, before importing any scientific
library.

``DeviceManager`` still pins the in-process *PyTorch* pool at device-resolution
time via ``torch.set_num_threads`` (which can re-size a live pool). This module
complements that by capping the parent process's non-torch BLAS/OpenMP pools,
which ``torch.set_num_threads`` cannot resize after import.

IMPORTANT: keep this module importable without importing ``numpy``, ``torch``,
or any other scientific library. It may only depend on the standard library and
``yaml`` (which is pure Python and does not pull in numpy).
"""

import os
from typing import Optional

import yaml

# Environment variables consulted by CPU math backends. Each backend reads its
# variable once, when it first initializes its thread pool.
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def validate_cpu_threads(num_threads: Optional[int]) -> Optional[int]:
    """Validate a ``cpu_threads`` value, returning it unchanged when valid.

    Mirrors the validation in ``DeviceConfig``/``DeviceManager`` so the two code
    paths agree on what is acceptable.
    """
    if num_threads is None:
        return None
    # bool is a subclass of int in Python; reject bool explicitly.
    if type(num_threads) is bool or not isinstance(num_threads, int):
        raise ValueError(f"cpu_threads must be an int >= 1 or None, got {num_threads!r}")
    if num_threads < 1:
        raise ValueError(f"cpu_threads must be >= 1 or None, got {num_threads}")
    return num_threads


def pin_cpu_math_threads(
    num_threads: Optional[int], *, override: bool = False
) -> Optional[int]:
    """Set CPU math-library thread env vars before numpy/torch import.

    Args:
        num_threads: Thread cap (``>= 1``) to apply, or ``None`` to leave the
            environment untouched (keep library defaults).
        override: When ``False`` (default), environment variables already set by
            the user/launcher are respected and left as-is. When ``True``, every
            variable is overwritten.

    Returns:
        The applied integer thread count, or ``None`` if nothing was set
        (because ``num_threads`` was ``None`` or every variable was already set
        and ``override`` was ``False``).
    """
    num_threads = validate_cpu_threads(num_threads)
    if num_threads is None:
        return None

    value = str(num_threads)
    applied = False
    for var in CPU_THREAD_ENV_VARS:
        if override or var not in os.environ:
            os.environ[var] = value
            applied = True
    return num_threads if applied else None


def _extract_cpu_threads(config: object) -> Optional[int]:
    """Pull ``cpu_threads`` out of a parsed config mapping, flat or nested."""
    if not isinstance(config, dict):
        return None
    if "cpu_threads" in config:
        return config["cpu_threads"]
    device = config.get("device")
    if isinstance(device, dict) and "cpu_threads" in device:
        return device["cpu_threads"]
    return None


def resolve_cpu_threads_from_config(
    environment: str = "development",
    profile: Optional[str] = None,
    config_dir: str = "farm/config",
    default: Optional[int] = 1,
) -> Optional[int]:
    """Resolve the configured ``cpu_threads`` from the centralized YAML files.

    This intentionally re-reads the YAML files with the standard library + PyYAML
    instead of going through the full config loader, because the full loader
    imports ``numpy``/``torch`` transitively and would defeat pre-import pinning.

    Files are consulted in *descending* precedence (profile > environment >
    base), matching the deep-merge order used by the real config loader, and the
    first file that specifies ``cpu_threads`` wins. ``cpu_threads: null`` is a
    valid, explicit "keep defaults" value and is returned as ``None``.

    Falls back to ``default`` if no file specifies the key or the files cannot be
    read, so a misconfigured/missing config never blocks startup.
    """
    candidate_paths = []
    if profile:
        candidate_paths.append(os.path.join(config_dir, "profiles", f"{profile}.yaml"))
    candidate_paths.append(
        os.path.join(config_dir, "environments", f"{environment}.yaml")
    )
    candidate_paths.append(os.path.join(config_dir, "default.yaml"))

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(parsed, dict) and (
            "cpu_threads" in parsed
            or (isinstance(parsed.get("device"), dict) and "cpu_threads" in parsed["device"])
        ):
            return _extract_cpu_threads(parsed)

    return default
