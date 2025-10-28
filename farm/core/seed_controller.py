"""
Centralized seed controller for deterministic simulations.

Provides per-agent and per-component RNG instances derived from a global seed,
ensuring deterministic behavior independent of agent processing order.
"""

import hashlib
import random
from typing import Optional, Tuple

import numpy as np
import torch

from farm.utils.logging import get_logger

logger = get_logger(__name__)


class SeedController:
    """
    Controller for managing deterministic random number generation.
    
    Creates per-agent RNG instances derived from a global seed plus agent ID,
    ensuring that each agent's randomness is independent of processing order
    and reproducible across simulation runs.
    """
    
    def __init__(self, global_seed: int):
        """
        Initialize seed controller with global seed.
        
        Args:
            global_seed: Global seed value for deterministic behavior
        """
        self.global_seed = global_seed
        logger.debug("seed_controller_initialized", global_seed=global_seed)
    
    def get_agent_rng(self, agent_id: str) -> Tuple[random.Random, np.random.Generator, torch.Generator]:
        """
        Get deterministic RNG instances for a specific agent.
        
        Uses deterministic hashing to derive agent-specific seeds from the
        global seed and agent ID, ensuring reproducible yet diverse randomness
        for each agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Tuple of (python_rng, numpy_rng, torch_generator) instances
            seeded with agent-specific values
        """
        # Derive agent-specific seed using cryptographic hash for determinism
        # Using hashlib instead of built-in hash() to ensure determinism across process runs
        hash_input = f"{self.global_seed}:{agent_id}".encode('utf-8')
        hash_digest = hashlib.blake2b(hash_input, digest_size=4).digest()
        agent_seed = int.from_bytes(hash_digest, byteorder='big') % (2**32)
        
        # Create seeded RNG instances
        py_rng = random.Random(agent_seed)
        np_rng = np.random.default_rng(agent_seed)
        torch_gen = torch.Generator().manual_seed(agent_seed)
        
        logger.debug(
            "agent_rng_created",
            agent_id=agent_id,
            agent_seed=agent_seed,
            global_seed=self.global_seed
        )
        
        return py_rng, np_rng, torch_gen
    
    def get_component_rng(self, agent_id: str, component_name: str) -> Tuple[random.Random, np.random.Generator, torch.Generator]:
        """
        Get deterministic RNG instances for a specific component.
        
        Useful for components that need their own randomness streams
        independent of the agent's main RNG.
        
        Args:
            agent_id: Unique identifier for the agent
            component_name: Name of the component (e.g., 'movement', 'perception')
            
        Returns:
            Tuple of (python_rng, numpy_rng, torch_generator) instances
            seeded with component-specific values
        """
        # Derive component-specific seed using cryptographic hash for determinism
        # Using hashlib instead of built-in hash() to ensure determinism across process runs
        hash_input = f"{self.global_seed}:{agent_id}:{component_name}".encode('utf-8')
        hash_digest = hashlib.blake2b(hash_input, digest_size=4).digest()
        component_seed = int.from_bytes(hash_digest, byteorder='big') % (2**32)
        
        # Create seeded RNG instances
        py_rng = random.Random(component_seed)
        np_rng = np.random.default_rng(component_seed)
        torch_gen = torch.Generator().manual_seed(component_seed)
        
        logger.debug(
            "component_rng_created",
            agent_id=agent_id,
            component_name=component_name,
            component_seed=component_seed,
            global_seed=self.global_seed
        )
        
        return py_rng, np_rng, torch_gen
