"""Optional run notarization via FarmNotary (pip install farm-notary)."""

from farm.provenance.notary import (
    OFFICIAL_PUBLISH_PATTERNS,
    farm_notary_available,
    notarize,
    notarize_run_dir,
    reproduce,
    verify,
)

__all__ = [
    "OFFICIAL_PUBLISH_PATTERNS",
    "farm_notary_available",
    "notarize",
    "notarize_run_dir",
    "reproduce",
    "verify",
]
