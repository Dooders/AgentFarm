"""
Genetics analysis utility helpers.
"""

from __future__ import annotations

from typing import List

from farm.database.data_types import GenomeId
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def parse_parent_ids(genome_id: str) -> List[str]:
    """Return parent agent IDs encoded in ``genome_id``.

    Returns an empty list when parsing fails or the genome ID encodes no
    parents (e.g. initial agents).
    """
    try:
        return GenomeId.from_string(genome_id).parent_ids
    except Exception as exc:
        logger.warning("parse_parent_ids failed for genome_id=%r: %s", genome_id, exc)
        return []
