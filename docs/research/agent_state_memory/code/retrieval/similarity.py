"""Similarity-based memory retrieval mechanisms.

This module provides methods for retrieving memories based on semantic
similarity using vector embeddings.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from ..storage.redis_stm import RedisSTMStore
from ..storage.redis_im import RedisIMStore
from ..storage.sqlite_ltm import SQLiteLTMStore
from ..embeddings.vector_store import VectorStore
from ..embeddings.autoencoder import AutoencoderEmbeddingEngine

logger = logging.getLogger(__name__)


class SimilarityRetrieval:
    """Retrieval mechanisms based on semantic similarity.
    
    This class provides methods for retrieving memories by comparing
    their vector embeddings to find semantically similar content.
    
    Attributes:
        vector_store: Vector store for similarity queries
        embedding_engine: Engine for generating embeddings from states
        stm_store: Short-Term Memory store
        im_store: Intermediate Memory store
        ltm_store: Long-Term Memory store
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_engine: AutoencoderEmbeddingEngine,
        stm_store: RedisSTMStore,
        im_store: RedisIMStore,
        ltm_store: SQLiteLTMStore,
    ):
        """Initialize the similarity retrieval.
        
        Args:
            vector_store: Vector store for similarity queries
            embedding_engine: Engine for generating embeddings
            stm_store: Short-Term Memory store
            im_store: Intermediate Memory store
            ltm_store: Long-Term Memory store
        """
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store
    
    def retrieve_similar_to_state(
        self, 
        state: Dict[str, Any],
        limit: int = 10,
        min_score: float = 0.7,
        memory_type: Optional[str] = None,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories similar to a given state.
        
        Args:
            state: State to compare against
            limit: Maximum number of memories to retrieve
            min_score: Minimum similarity score threshold
            memory_type: Optional filter for memory type
            tier: Memory tier to search ("stm", "im", "ltm")
            
        Returns:
            List of similar memories
        """
        # Generate embedding for the state
        if tier == "stm":
            query_vector = self.embedding_engine.encode_stm(state)
        elif tier == "im":
            query_vector = self.embedding_engine.encode_im(state)
        else:  # ltm
            query_vector = self.embedding_engine.encode_ltm(state)
        
        # Set up metadata filter if memory_type specified
        metadata_filter = {}
        if memory_type:
            metadata_filter = {"memory_type": memory_type}
        
        # Find similar vectors
        similar_vectors = self.vector_store.find_similar_memories(
            query_vector,
            tier=tier,
            limit=limit * 2,  # Get extra results to allow for score filtering
            metadata_filter=metadata_filter
        )
        
        # Filter by score
        filtered_vectors = [v for v in similar_vectors if v["score"] >= min_score]
        
        # Limit results
        results = filtered_vectors[:limit]
        
        # Fetch full memory entries
        memories = []
        for result in results:
            memory_id = result["id"]
            memory = None
            
            # Look up in the appropriate store
            if tier == "stm":
                memory = self.stm_store.get(memory_id)
            elif tier == "im":
                memory = self.im_store.get(memory_id)
            else:  # ltm
                memory = self.ltm_store.get(memory_id)
            
            if memory:
                # Attach similarity score
                if "metadata" not in memory:
                    memory["metadata"] = {}
                memory["metadata"]["similarity_score"] = result["score"]
                memories.append(memory)
        
        return memories
    
    def retrieve_similar_to_memory(
        self, 
        memory_id: str,
        limit: int = 10,
        min_score: float = 0.7,
        exclude_self: bool = True,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories similar to a given memory.
        
        Args:
            memory_id: ID of the memory to compare against
            limit: Maximum number of memories to retrieve
            min_score: Minimum similarity score threshold
            exclude_self: Whether to exclude the query memory from results
            tier: Memory tier to search ("stm", "im", "ltm")
            
        Returns:
            List of similar memories
        """
        # Retrieve the reference memory
        reference_memory = None
        if tier == "stm":
            reference_memory = self.stm_store.get(memory_id)
        elif tier == "im":
            reference_memory = self.im_store.get(memory_id)
        else:  # ltm
            reference_memory = self.ltm_store.get(memory_id)
        
        if not reference_memory:
            logger.warning("Reference memory %s not found in %s", memory_id, tier)
            return []
        
        # Get the embedding directly if available
        query_vector = None
        if "embeddings" in reference_memory:
            if tier == "stm" and "full_vector" in reference_memory["embeddings"]:
                query_vector = reference_memory["embeddings"]["full_vector"]
            elif tier == "im" and "compressed_vector" in reference_memory["embeddings"]:
                query_vector = reference_memory["embeddings"]["compressed_vector"]
            elif tier == "ltm" and "abstract_vector" in reference_memory["embeddings"]:
                query_vector = reference_memory["embeddings"]["abstract_vector"]
        
        # If no embedding available, generate from contents
        if not query_vector and "contents" in reference_memory:
            if tier == "stm":
                query_vector = self.embedding_engine.encode_stm(reference_memory["contents"])
            elif tier == "im":
                query_vector = self.embedding_engine.encode_im(reference_memory["contents"])
            else:  # ltm
                query_vector = self.embedding_engine.encode_ltm(reference_memory["contents"])
        
        if not query_vector:
            logger.warning("Could not obtain embedding for memory %s", memory_id)
            return []
        
        # Find similar vectors
        similar_vectors = self.vector_store.find_similar_memories(
            query_vector,
            tier=tier,
            limit=limit * 2,  # Get extra results to allow for filtering
        )
        
        # Filter by score and exclude self if needed
        filtered_vectors = []
        for v in similar_vectors:
            if v["score"] >= min_score:
                if exclude_self and v["id"] == memory_id:
                    continue
                filtered_vectors.append(v)
        
        # Limit results
        results = filtered_vectors[:limit]
        
        # Fetch full memory entries
        memories = []
        for result in results:
            memory_id = result["id"]
            memory = None
            
            # Look up in the appropriate store
            if tier == "stm":
                memory = self.stm_store.get(memory_id)
            elif tier == "im":
                memory = self.im_store.get(memory_id)
            else:  # ltm
                memory = self.ltm_store.get(memory_id)
            
            if memory:
                # Attach similarity score
                if "metadata" not in memory:
                    memory["metadata"] = {}
                memory["metadata"]["similarity_score"] = result["score"]
                memories.append(memory)
        
        return memories
    
    def retrieve_by_example(
        self, 
        example: Dict[str, Any],
        limit: int = 10,
        min_score: float = 0.7,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories similar to an example pattern.
        
        This method enables query-by-example where you provide a partial
        or synthetic memory pattern to match against.
        
        Args:
            example: Example pattern to match
            limit: Maximum number of memories to retrieve
            min_score: Minimum similarity score threshold
            tier: Memory tier to search ("stm", "im", "ltm")
            
        Returns:
            List of matching memories
        """
        # Generate embedding for the example
        if tier == "stm":
            query_vector = self.embedding_engine.encode_stm(example)
        elif tier == "im":
            query_vector = self.embedding_engine.encode_im(example)
        else:  # ltm
            query_vector = self.embedding_engine.encode_ltm(example)
        
        # Find similar vectors
        similar_vectors = self.vector_store.find_similar_memories(
            query_vector,
            tier=tier,
            limit=limit * 2,  # Get extra results to allow for score filtering
        )
        
        # Filter by score
        filtered_vectors = [v for v in similar_vectors if v["score"] >= min_score]
        
        # Limit results
        results = filtered_vectors[:limit]
        
        # Fetch full memory entries
        memories = []
        for result in results:
            memory_id = result["id"]
            memory = None
            
            # Look up in the appropriate store
            if tier == "stm":
                memory = self.stm_store.get(memory_id)
            elif tier == "im":
                memory = self.im_store.get(memory_id)
            else:  # ltm
                memory = self.ltm_store.get(memory_id)
            
            if memory:
                # Attach similarity score
                if "metadata" not in memory:
                    memory["metadata"] = {}
                memory["metadata"]["similarity_score"] = result["score"]
                memories.append(memory)
        
        return memories 