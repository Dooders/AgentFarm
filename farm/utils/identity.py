"""Centralized identity generation utilities.

Provides a single API for generating, parsing, and validating identifiers
across the codebase, with optional deterministic behavior.
"""

from dataclasses import dataclass
from typing import NewType, Optional, Sequence, Tuple
import hashlib

from .short_id import ShortUUID


# Strongly-typed string aliases for clarity
AgentId = NewType("AgentId", str)
SimulationId = NewType("SimulationId", str)
ExperimentId = NewType("ExperimentId", str)
GenomeIdStr = NewType("GenomeIdStr", str)
RunId = NewType("RunId", str)
AgentStateId = NewType("AgentStateId", str)


@dataclass
class IdentityConfig:
    """Configuration for identity generation.

    deterministic_seed: If provided, agent IDs generated sequentially via
        Identity.agent_id() will be deterministic and reproducible for the
        same seed and creation order.
    alphabet: Optional custom alphabet for short IDs (forwarded to ShortUUID).
    default_length: Default length for short IDs when truncating.
    """

    deterministic_seed: Optional[int] = None
    alphabet: Optional[str] = None
    default_length: int = 10


class Identity:
    """Identity service for generating and parsing IDs.

    Create a single instance per context (e.g., per Environment) and use it to
    create all identifiers. When configured with a deterministic seed, agent
    IDs created through this instance will be reproducible.
    """

    def __init__(self, config: Optional[IdentityConfig] = None) -> None:
        self.config = config or IdentityConfig()
        self._short_uuid = ShortUUID(self.config.alphabet)
        self._agent_counter = 0

    # ----- Core helpers -----
    def short(self, length: Optional[int] = None) -> str:
        """Generate a short, random identifier with a default length."""
        value = self._short_uuid.uuid()
        target_len = length if length is not None else self.config.default_length
        return value[:target_len]

    def short_deterministic(self, key: str, length: Optional[int] = None) -> str:
        """Generate a deterministic short identifier from a key and the seed.

        Uses BLAKE2b for stable, fast hashing. The same seed and key yield the
        same output; different seeds or keys produce different outputs.
        """
        seed_str = "" if self.config.deterministic_seed is None else str(self.config.deterministic_seed)
        digest = hashlib.blake2b(f"{seed_str}:{key}".encode("utf-8"), digest_size=16).hexdigest()
        target_len = length if length is not None else self.config.default_length
        return digest[:target_len]

    # ----- Namespaced factories -----
    def simulation_id(self, prefix: str = "sim") -> SimulationId:
        return SimulationId(f"{prefix}_{self.short()}")

    def run_id(self, length: int = 8) -> RunId:
        return RunId(self.short(length))

    def experiment_id(self) -> ExperimentId:
        # Slightly longer by default to reduce accidental collisions in dashboards
        return ExperimentId(self.short(22))

    def agent_id(self) -> AgentId:
        if self.config.deterministic_seed is not None:
            value = self.short_deterministic(f"agent:{self._agent_counter}")
            self._agent_counter += 1
            return AgentId(f"agent_{value}")
        return AgentId(f"agent_{self.short()}")

    def agent_state_id(self, agent_id: str, step_number: int) -> AgentStateId:
        return AgentStateId(f"{agent_id}-{step_number}")

    def genome_id(
        self,
        agent_type: str,
        generation: int,
        parents: Sequence[str],
        time_step: int,
    ) -> GenomeIdStr:
        parents_part = "none" if not parents else "_".join(parents)
        return GenomeIdStr(f"{agent_type}:{generation}:{parents_part}:{time_step}")

    # ----- Parsers/validators -----
    @staticmethod
    def parse_agent_state_id(agent_state_id: str) -> Tuple[str, int]:
        """Parse an agent state id of the form '<agent_id>-<step_number>'."""
        agent_id, step_str = agent_state_id.rsplit("-", 1)
        return agent_id, int(step_str)

