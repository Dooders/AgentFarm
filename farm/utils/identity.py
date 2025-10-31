"""Centralized identity generation utilities.

Provides a single API for generating, parsing, and validating identifiers
across the codebase, with optional deterministic behavior.

This module implements short ID generation directly without depending on
the legacy ShortUUID class, centralizing all ID generation logic here.
"""

import hashlib
import math
import uuid as _uu
from dataclasses import dataclass
from typing import Callable, NewType, Optional, Sequence, Tuple

# Default alphabet for short ID generation (same as ShortUUID)
DEFAULT_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _int_to_string(number: int, alphabet: str, padding: Optional[int] = None) -> str:
    """
    Convert a number to a string, using the given alphabet.
    The output has the most significant digit first.
    """
    output = ""
    alpha_len = len(alphabet)
    while number:
        number, digit = divmod(number, alpha_len)
        output += alphabet[digit]
    if padding:
        remainder = max(padding - len(output), 0)
        output = output + alphabet[0] * remainder
    return output[::-1]


def _generate_short_uuid(alphabet: str, pad_length: Optional[int] = None) -> str:
    """
    Generate and return a short UUID using the given alphabet.
    """
    alpha_len = len(alphabet)
    if pad_length is None:
        # Calculate the necessary length to fit the entire UUID
        pad_length = int(math.ceil(math.log(2**128, alpha_len)))

    # Generate a random UUID and convert to int
    u = _uu.uuid4()
    return _int_to_string(u.int, alphabet, padding=pad_length)


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
    alphabet: Optional custom alphabet for short IDs.
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
        self._alphabet = (
            self.config.alphabet
            if self.config.alphabet is not None
            else DEFAULT_ALPHABET
        )
        self._agent_counter = 0
        self._resource_counter = 0
        # Registry to track base genome IDs and their counters
        # Format: base_genome_id (e.g., "agent_a:", "agent_a:agent_b", "::") -> max_counter (int)
        self._genome_id_registry: dict[str, int] = {}

    # ----- Core helpers -----
    def short(self, length: Optional[int] = None) -> str:
        """Generate a short, random identifier with a default length."""
        target_len = length if length is not None else self.config.default_length
        value = _generate_short_uuid(self._alphabet, pad_length=target_len)
        return value[:target_len]

    def short_deterministic(self, key: str, length: Optional[int] = None) -> str:
        """Generate a deterministic short identifier from a key and the seed.

        Uses BLAKE2b for stable, fast hashing. The same seed and key yield the
        same output; different seeds or keys produce different outputs.
        """
        seed_str = (
            ""
            if self.config.deterministic_seed is None
            else str(self.config.deterministic_seed)
        )
        digest = hashlib.blake2b(
            f"{seed_str}:{key}".encode("utf-8"), digest_size=16
        ).hexdigest()
        target_len = length if length is not None else self.config.default_length
        return digest[:target_len]

    # ----- Namespaced factories -----
    def simulation_id(self, prefix: str = "sim") -> SimulationId:
        if self.config.deterministic_seed is not None:
            # Use deterministic generation for reproducible simulation IDs
            value = self.short_deterministic(f"simulation:{prefix}")
            return SimulationId(f"{prefix}_{value}")
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

    def resource_id(self) -> str:
        """Generate a unique resource identifier following the pattern resource_{shortid}."""
        if self.config.deterministic_seed is not None:
            value = self.short_deterministic(f"resource:{self._resource_counter}")
            self._resource_counter += 1
            return f"resource_{value}"
        return f"resource_{self.short()}"

    @staticmethod
    def agent_state_id(agent_id: str, step_number: int) -> AgentStateId:
        return AgentStateId(f"{agent_id}-{step_number}")

    def genome_id(
        self,
        parent_ids: Sequence[str],
        existing_genome_checker: Optional[Callable[[str], bool]] = None,
    ) -> GenomeIdStr:
        """Generate a genome ID based on parent agent IDs.
        
        Format:
        - No parents (initial agents): `::1`, `::2`, etc.
        - Single parent (cloning): `{parent_id}:1`, `{parent_id}:2`, etc.
        - Two parents (sexual reproduction): `{parent1}:{parent2}:1`, `{parent1}:{parent2}:2`, etc.
        - Counters start from 1 (first offspring is :1, second is :2, etc.)
        
        Args:
            parent_ids: List of parent agent IDs. Empty for initial agents.
            existing_genome_checker: Optional callback to check if a genome_id exists
                in the database. Should return True if genome_id exists, False otherwise.
        
        Returns:
            GenomeIdStr in the format `parent1:parent2:counter` where counter >= 1
        """
        # Generate base genome ID from parent IDs
        if not parent_ids:
            # No parents: initial agent
            base_genome_id = "::"
        elif len(parent_ids) == 1:
            # Single parent: cloning
            base_genome_id = f"{parent_ids[0]}:"
        elif len(parent_ids) == 2:
            # Two parents: sexual reproduction
            base_genome_id = f"{parent_ids[0]}:{parent_ids[1]}"
        else:
            # More than 2 parents (shouldn't happen normally, but handle gracefully)
            # Use first two parents
            base_genome_id = f"{parent_ids[0]}:{parent_ids[1]}"
        
        # Determine separator for counter (depends on whether base ends with ':')
        separator = "" if base_genome_id.endswith(":") else ":"
        
        # Check in-memory registry for max counter
        max_counter_in_registry = self._genome_id_registry.get(base_genome_id, 0)
        
        # Check database if checker provided
        max_counter_in_db = 0
        
        if existing_genome_checker:
            # Find highest counter in database, starting from 1
            # Format with counter: base:counter if base doesn't end with ':', else basecounter
            counter = 1
            while existing_genome_checker(f"{base_genome_id}{separator}{counter}"):
                max_counter_in_db = counter
                counter += 1
        
        # Determine max counter from both sources
        # Registry stores the max counter used, so next should be max + 1
        max_counter = max(max_counter_in_registry, max_counter_in_db)
        
        # Always use a counter starting from 1
        # Next counter is max_counter + 1
        next_counter = max_counter + 1
        self._genome_id_registry[base_genome_id] = next_counter
        
        return GenomeIdStr(f"{base_genome_id}{separator}{next_counter}")

    # ----- Parsers/validators -----
    @staticmethod
    def parse_agent_state_id(agent_state_id: str) -> Tuple[str, int]:
        """Parse an agent state id of the form '<agent_id>-<step_number>'."""
        agent_id, step_str = agent_state_id.rsplit("-", 1)
        return agent_id, int(step_str)
