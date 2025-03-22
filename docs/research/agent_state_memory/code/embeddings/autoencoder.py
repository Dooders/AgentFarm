"""Autoencoder-based embeddings for agent memory states."""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ..config import AutoencoderConfig

logger = logging.getLogger(__name__)


class StateAutoencoder(nn.Module):
    """Neural network autoencoder for agent state vectorization and compression.
    
    The autoencoder consists of:
    1. An encoder that compresses input features to the embedding space
    2. A decoder that reconstructs original features from embeddings
    3. Multiple "bottlenecks" for different compression levels
    """
    
    def __init__(self, input_dim: int, stm_dim: int = 384, im_dim: int = 128, ltm_dim: int = 32):
        """Initialize the multi-resolution autoencoder.
        
        Args:
            input_dim: Dimension of the flattened input features
            stm_dim: Dimension for Short-Term Memory (STM) embeddings
            im_dim: Dimension for Intermediate Memory (IM) embeddings
            ltm_dim: Dimension for Long-Term Memory (LTM) embeddings
        """
        super(StateAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Multi-resolution bottlenecks
        self.stm_bottleneck = nn.Linear(256, stm_dim)
        self.im_bottleneck = nn.Linear(stm_dim, im_dim)
        self.ltm_bottleneck = nn.Linear(im_dim, ltm_dim)
        
        # Expansion layers (from LTM to IM to STM)
        self.ltm_to_im = nn.Linear(ltm_dim, im_dim)
        self.im_to_stm = nn.Linear(im_dim, stm_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(stm_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )
    
    def encode_stm(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to STM embedding space.
        
        Args:
            x: Input tensor
            
        Returns:
            STM embedding
        """
        x = self.encoder(x)
        return self.stm_bottleneck(x)
    
    def encode_im(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to IM embedding space.
        
        Args:
            x: Input tensor
            
        Returns:
            IM embedding
        """
        x = self.encoder(x)
        x = self.stm_bottleneck(x)
        return self.im_bottleneck(x)
    
    def encode_ltm(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to LTM embedding space.
        
        Args:
            x: Input tensor
            
        Returns:
            LTM embedding
        """
        x = self.encoder(x)
        x = self.stm_bottleneck(x)
        x = self.im_bottleneck(x)
        return self.ltm_bottleneck(x)
    
    def decode_stm(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from STM embedding space.
        
        Args:
            z: STM embedding
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z)
    
    def decode_im(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from IM embedding space.
        
        Args:
            z: IM embedding
            
        Returns:
            Reconstructed input
        """
        z = self.im_to_stm(z)
        return self.decoder(z)
    
    def decode_ltm(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from LTM embedding space.
        
        Args:
            z: LTM embedding
            
        Returns:
            Reconstructed input
        """
        z = self.ltm_to_im(z)
        z = self.im_to_stm(z)
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor, level: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            level: Compression level (0=STM, 1=IM, 2=LTM)
            
        Returns:
            Tuple of (reconstructed output, embedding)
        """
        # Encode
        encoded = self.encoder(x)
        
        if level == 0:  # STM
            embedding = self.stm_bottleneck(encoded)
            decoded = self.decoder(embedding)
        elif level == 1:  # IM
            stm_embedding = self.stm_bottleneck(encoded)
            embedding = self.im_bottleneck(stm_embedding)
            expanded = self.im_to_stm(embedding)
            decoded = self.decoder(expanded)
        else:  # LTM
            stm_embedding = self.stm_bottleneck(encoded)
            im_embedding = self.im_bottleneck(stm_embedding)
            embedding = self.ltm_bottleneck(im_embedding)
            expanded_im = self.ltm_to_im(embedding)
            expanded_stm = self.im_to_stm(expanded_im)
            decoded = self.decoder(expanded_stm)
        
        return decoded, embedding


class AgentStateDataset(Dataset):
    """Dataset for training the autoencoder on agent states."""
    
    def __init__(self, states: List[Dict[str, Any]]):
        """Initialize the dataset.
        
        Args:
            states: List of agent state dictionaries
        """
        self.states = states
        self.vectors = self._prepare_vectors()
    
    def _prepare_vectors(self) -> np.ndarray:
        """Convert agent states to input vectors.
        
        Returns:
            Numpy array of flattened state vectors
        """
        # Extract numeric values from states
        vectors = []
        for state in self.states:
            # Extract all numeric values from the state dictionary
            vector = []
            for key, value in self._flatten_dict(state).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    vector.append(float(value))
            vectors.append(vector)
        
        # Pad to ensure uniform length
        max_len = max(len(v) for v in vectors)
        padded_vectors = []
        for v in vectors:
            padded = v + [0.0] * (max_len - len(v))
            padded_vectors.append(padded)
        
        return np.array(padded_vectors, dtype=np.float32)
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            prefix: Key prefix for nested dictionaries
            
        Returns:
            Flattened dictionary
        """
        result = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten_dict(v, key))
            else:
                result[key] = v
        return result
    
    def __len__(self) -> int:
        """Get the number of samples.
        
        Returns:
            Dataset size
        """
        return len(self.vectors)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tensor representation of the state
        """
        return torch.tensor(self.vectors[idx], dtype=torch.float32)


class AutoencoderEmbeddingEngine:
    """Engine for generating embeddings using the autoencoder model.
    
    This class handles the training and inference of the autoencoder model,
    providing methods to encode and decode agent states at different
    compression levels.
    
    Attributes:
        model: The autoencoder model
        input_dim: Dimension of the input features
        stm_dim: Dimension of STM embeddings
        im_dim: Dimension of IM embeddings
        ltm_dim: Dimension of LTM embeddings
        device: Device to run the model on
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_dim: int = 64,
        stm_dim: int = 384,
        im_dim: int = 128,
        ltm_dim: int = 32,
    ):
        """Initialize the embedding engine.
        
        Args:
            model_path: Path to saved model (if available)
            input_dim: Dimension of the input features
            stm_dim: Dimension of STM embeddings
            im_dim: Dimension of IM embeddings
            ltm_dim: Dimension of LTM embeddings
        """
        self.input_dim = input_dim
        self.stm_dim = stm_dim
        self.im_dim = im_dim
        self.ltm_dim = ltm_dim
        
        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = StateAutoencoder(input_dim, stm_dim, im_dim, ltm_dim).to(self.device)
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info("Loaded autoencoder model from %s", model_path)
        else:
            logger.warning("No pre-trained model found. Using untrained model.")
    
    def encode_stm(self, state: Dict[str, Any]) -> List[float]:
        """Encode state to STM embedding space.
        
        Args:
            state: Agent state dictionary
            
        Returns:
            STM embedding as list of floats
        """
        vector = self._state_to_vector(state)
        with torch.no_grad():
            x = torch.tensor(vector, dtype=torch.float32).to(self.device)
            embedding = self.model.encode_stm(x)
        return embedding.cpu().numpy().tolist()
    
    def encode_im(self, state: Dict[str, Any]) -> List[float]:
        """Encode state to IM embedding space.
        
        Args:
            state: Agent state dictionary
            
        Returns:
            IM embedding as list of floats
        """
        vector = self._state_to_vector(state)
        with torch.no_grad():
            x = torch.tensor(vector, dtype=torch.float32).to(self.device)
            embedding = self.model.encode_im(x)
        return embedding.cpu().numpy().tolist()
    
    def encode_ltm(self, state: Dict[str, Any]) -> List[float]:
        """Encode state to LTM embedding space.
        
        Args:
            state: Agent state dictionary
            
        Returns:
            LTM embedding as list of floats
        """
        vector = self._state_to_vector(state)
        with torch.no_grad():
            x = torch.tensor(vector, dtype=torch.float32).to(self.device)
            embedding = self.model.encode_ltm(x)
        return embedding.cpu().numpy().tolist()
    
    def _state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert a state dictionary to a vector.
        
        Args:
            state: Agent state dictionary
            
        Returns:
            Numpy array of state features
        """
        # Create a dataset with just this state
        dataset = AgentStateDataset([state])
        
        # Get the vector
        vector = dataset.vectors[0]
        
        # Ensure correct dimension
        if len(vector) < self.input_dim:
            vector = np.pad(vector, (0, self.input_dim - len(vector)))
        elif len(vector) > self.input_dim:
            vector = vector[:self.input_dim]
        
        return vector
    
    def train(
        self,
        states: List[Dict[str, Any]],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict[str, List[float]]:
        """Train the autoencoder on agent states.
        
        Args:
            states: List of agent state dictionaries
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary of training metrics
        """
        # Create dataset and dataloader
        dataset = AgentStateDataset(states)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        metrics = {"loss": [], "stm_loss": [], "im_loss": [], "ltm_loss": []}
        
        for epoch in range(epochs):
            running_loss = 0.0
            running_stm_loss = 0.0
            running_im_loss = 0.0
            running_ltm_loss = 0.0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Train on all compression levels
                stm_output, _ = self.model(batch, level=0)
                im_output, _ = self.model(batch, level=1)
                ltm_output, _ = self.model(batch, level=2)
                
                # Calculate losses
                stm_loss = criterion(stm_output, batch)
                im_loss = criterion(im_output, batch)
                ltm_loss = criterion(ltm_output, batch)
                
                # Combined loss (weighted by importance)
                loss = stm_loss + 0.5 * im_loss + 0.25 * ltm_loss
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Update metrics
                running_loss += loss.item()
                running_stm_loss += stm_loss.item()
                running_im_loss += im_loss.item()
                running_ltm_loss += ltm_loss.item()
            
            # Record epoch metrics
            metrics["loss"].append(running_loss / len(dataloader))
            metrics["stm_loss"].append(running_stm_loss / len(dataloader))
            metrics["im_loss"].append(running_im_loss / len(dataloader))
            metrics["ltm_loss"].append(running_ltm_loss / len(dataloader))
            
            logger.info(
                f"Epoch {epoch+1}/{epochs}, Loss: {metrics['loss'][-1]:.4f}, "
                f"STM: {metrics['stm_loss'][-1]:.4f}, "
                f"IM: {metrics['im_loss'][-1]:.4f}, "
                f"LTM: {metrics['ltm_loss'][-1]:.4f}"
            )
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "stm_dim": self.stm_dim,
            "im_dim": self.im_dim,
            "ltm_dim": self.ltm_dim,
        }, path)
        logger.info("Model saved to %s", path)
    
    def load_model(self, path: str) -> None:
        """Load the model from a file.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.stm_dim = checkpoint["stm_dim"]
        self.im_dim = checkpoint["im_dim"]
        self.ltm_dim = checkpoint["ltm_dim"]
        
        # Re-initialize model with loaded dimensions
        self.model = StateAutoencoder(
            self.input_dim, 
            self.stm_dim, 
            self.im_dim, 
            self.ltm_dim
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Model loaded from %s", path)