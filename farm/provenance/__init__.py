"""Optional run notarization via FarmNotary (pip install farm-notary)."""

from farm.provenance.notary import (
    farm_notary_available,
    notarize,
    notarize_run_dir,
    reproduce,
    verify,
)

__all__ = [
    "farm_notary_available",
    "notarize",
    "notarize_run_dir",
    "reproduce",
    "verify",
]
