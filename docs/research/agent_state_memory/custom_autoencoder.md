# Custom Autoencoder for AgentMemory Embeddings

## Overview

This document outlines the implementation of a custom autoencoder for generating and compressing agent state embeddings in the hierarchical memory system. The autoencoder provides high-quality vectorization of agent states while enabling efficient dimension reduction between memory tiers.

## Architecture

```python
class StateAutoencoder(nn.Module):
    """Neural network autoencoder for agent state vectorization and compression.
    
    The autoencoder consists of:
    1. An encoder that compresses input features to the embedding space
    2. A decoder that reconstructs original features from embeddings
    3. Multiple "bottlenecks" for different compression levels
    """
    
    def __init__(self, input_dim, stm_dim=384, im_dim=128, ltm_dim=32):
        """
        Initialize the multi-resolution autoencoder.
        
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
            nn.Sigmoid()  # For normalized features
        )
    
    def encode_stm(self, x):
        """Encode input to STM resolution."""
        x = self.encoder(x)
        return self.stm_bottleneck(x)
    
    def encode_im(self, x):
        """Encode input to IM resolution."""
        x = self.encoder(x)
        stm_embedding = self.stm_bottleneck(x)
        return self.im_bottleneck(stm_embedding)
    
    def encode_ltm(self, x):
        """Encode input to LTM resolution (highest compression)."""
        x = self.encoder(x)
        stm_embedding = self.stm_bottleneck(x)
        im_embedding = self.im_bottleneck(stm_embedding)
        return self.ltm_bottleneck(im_embedding)
    
    def decode_from_stm(self, z):
        """Decode from STM embedding."""
        return self.decoder(z)
    
    def decode_from_im(self, z):
        """Decode from IM embedding."""
        stm_z = self.im_to_stm(z)
        return self.decoder(stm_z)
    
    def decode_from_ltm(self, z):
        """Decode from LTM embedding."""
        im_z = self.ltm_to_im(z)
        stm_z = self.im_to_stm(im_z)
        return self.decoder(stm_z)
    
    def forward(self, x):
        """Full forward pass with reconstruction from STM embedding."""
        stm_embedding = self.encode_stm(x)
        reconstruction = self.decode_from_stm(stm_embedding)
        return reconstruction, stm_embedding
```

## Training the Autoencoder

```python
class AutoencoderTrainer:
    """Handles training and evaluation of the state autoencoder."""
    
    def __init__(self, input_dim, stm_dim=384, im_dim=128, ltm_dim=32, 
                 learning_rate=0.001, model_path="models/state_autoencoder.pt"):
        """
        Initialize the trainer.
        
        Args:
            input_dim: Dimension of the flattened input features
            stm_dim: Dimension for STM embeddings
            im_dim: Dimension for IM embeddings  
            ltm_dim: Dimension for LTM embeddings
            learning_rate: Learning rate for optimization
            model_path: Path to save/load the model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StateAutoencoder(input_dim, stm_dim, im_dim, ltm_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.model_path = model_path
        
    def prepare_batch(self, batch_data):
        """
        Convert a batch of agent states to model input tensor.
        
        Args:
            batch_data: List of agent state dictionaries
            
        Returns:
            torch.Tensor: Batch of normalized, flattened features
        """
        # Extract and normalize features from agent states
        feature_extractor = FeatureExtractor()
        features = [feature_extractor.extract_features(state) for state in batch_data]
        
        # Convert to tensor and move to device
        return torch.tensor(features, dtype=torch.float32).to(self.device)
    
    def train_epoch(self, data_loader):
        """
        Train for one epoch.
        
        Args:
            data_loader: DataLoader providing batches of agent states
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_data in data_loader:
            # Prepare batch
            x = self.prepare_batch(batch_data)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, _ = self.model(x)
            
            # Calculate loss with multi-level reconstruction
            stm_embedding = self.model.encode_stm(x)
            im_embedding = self.model.encode_im(x)
            ltm_embedding = self.model.encode_ltm(x)
            
            # Calculate reconstruction loss from each level
            stm_recon = self.model.decode_from_stm(stm_embedding)
            im_recon = self.model.decode_from_im(im_embedding)
            ltm_recon = self.model.decode_from_ltm(ltm_embedding)
            
            # Combined loss with higher weight on STM accuracy
            loss = (
                0.5 * self.criterion(stm_recon, x) +
                0.3 * self.criterion(im_recon, x) +
                0.2 * self.criterion(ltm_recon, x)
            )
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def train(self, train_loader, val_loader=None, epochs=50, save_best=True):
        """
        Train the autoencoder.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            epochs: Number of training epochs
            save_best: Whether to save the best model based on validation loss
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            
            if val_loader:
                val_loss = self.evaluate(val_loader)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model()
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
                
                if save_best and epoch % 5 == 0:
                    self.save_model()
        
        if not save_best:
            self.save_model()
    
    def evaluate(self, data_loader):
        """
        Evaluate the autoencoder.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            float: Average loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_data in data_loader:
                x = self.prepare_batch(batch_data)
                reconstructed, _ = self.model(x)
                loss = self.criterion(reconstructed, x)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def save_model(self):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load the model from disk."""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"Model loaded from {self.model_path}")
            return True
        return False
```

## Feature Extraction for Autoencoder Input

```python
class FeatureExtractor:
    """Extracts and normalizes features from agent states for autoencoder input."""
    
    def __init__(self):
        """Initialize the feature extractor with normalization parameters."""
        # Define standard normalization ranges for known numerical features
        self.numerical_ranges = {
            "health": (0, 100),
            "energy": (0, 100),
            "resources": (0, 1000),
            "x_position": (-500, 500),
            "y_position": (-500, 500),
            # Add other numerical features as needed
        }
        
        # Define known categorical features and their possible values
        self.categorical_features = {
            "agent_type": ["harvester", "explorer", "defender", "builder"],
            "status": ["idle", "active", "sleeping", "engaged"],
            "current_action": ["move", "gather", "attack", "defend", "build", "rest", "none"],
            # Add other categorical features as needed
        }
    
    def extract_features(self, state_data):
        """
        Extract normalized features from an agent state.
        
        Args:
            state_data: Dictionary containing agent state
            
        Returns:
            numpy.ndarray: Flattened, normalized feature vector
        """
        # Extract and normalize numerical features
        numerical_features = self._extract_numerical(state_data)
        
        # Extract and one-hot encode categorical features
        categorical_features = self._extract_categorical(state_data)
        
        # Process special structures like perception grid if present
        perception_features = self._extract_perception(state_data)
        
        # Combine all features into a single vector
        combined = np.concatenate([
            numerical_features,
            categorical_features,
            perception_features
        ])
        
        return combined
    
    def _extract_numerical(self, state_data):
        """Extract and normalize numerical features."""
        features = []
        
        # Process known numerical features with standard normalization
        for feature, (min_val, max_val) in self.numerical_ranges.items():
            if feature in state_data:
                # Normalize to [0,1] range
                normalized = (state_data[feature] - min_val) / (max_val - min_val)
                features.append(max(0, min(1, normalized)))  # Clip to [0,1]
            else:
                # Use default value if feature is missing
                features.append(0.5)  # Middle of the range as default
        
        # Process any additional numerical features with standard scaling
        for key, value in state_data.items():
            if isinstance(value, (int, float)) and key not in self.numerical_ranges:
                # Use sigmoid-like normalization for unknown ranges
                features.append(value / (1 + abs(value)))
        
        return np.array(features)
    
    def _extract_categorical(self, state_data):
        """Extract and one-hot encode categorical features."""
        features = []
        
        # One-hot encode known categorical features
        for feature, possible_values in self.categorical_features.items():
            if feature in state_data:
                value = state_data[feature]
                # One-hot encode
                encoding = [1 if value == val else 0 for val in possible_values]
                features.extend(encoding)
            else:
                # Use all zeros if feature is missing
                features.extend([0] * len(possible_values))
        
        return np.array(features)
    
    def _extract_perception(self, state_data):
        """Extract features from perception grid if present."""
        if "perception" not in state_data:
            return np.array([0])  # Default if no perception
        
        perception = state_data["perception"]
        
        if isinstance(perception, dict) and "grid" in perception:
            # If perception is a grid, flatten and normalize
            grid = perception["grid"]
            if isinstance(grid, list):
                flattened = []
                for row in grid:
                    if isinstance(row, list):
                        flattened.extend(row)
                
                # Normalize grid values
                return np.array(flattened) / max(max(flattened), 1)
        
        # Fallback: return a single default value
        return np.array([0])
```

## Integration with Memory System

```python
class AutoencoderEmbeddingEngine:
    """Uses trained autoencoder to generate embeddings for agent memory system."""
    
    def __init__(self, model_path="models/state_autoencoder.pt", input_dim=512, 
                 stm_dim=384, im_dim=128, ltm_dim=32):
        """
        Initialize the embedding engine with a trained autoencoder.
        
        Args:
            model_path: Path to the trained autoencoder model
            input_dim: Input dimension for the autoencoder
            stm_dim: STM embedding dimension
            im_dim: IM embedding dimension
            ltm_dim: LTM embedding dimension
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StateAutoencoder(input_dim, stm_dim, im_dim, ltm_dim).to(self.device)
        self.feature_extractor = FeatureExtractor()
        
        # Load trained model
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded autoencoder model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def create_state_embedding(self, state_data):
        """
        Generate embeddings at all resolution levels for an agent state.
        
        Args:
            state_data: Dictionary with agent state information
            
        Returns:
            dict: Dictionary with embeddings at different resolutions
        """
        # Extract features
        features = self.feature_extractor.extract_features(state_data)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Generate embeddings at all resolution levels
        with torch.no_grad():
            stm_embedding = self.model.encode_stm(features_tensor)
            im_embedding = self.model.encode_im(features_tensor)
            ltm_embedding = self.model.encode_ltm(features_tensor)
        
        # Convert to numpy arrays
        embeddings = {
            "full_vector": stm_embedding.squeeze().cpu().numpy(),
            "compressed_vector": im_embedding.squeeze().cpu().numpy(),
            "abstract_vector": ltm_embedding.squeeze().cpu().numpy()
        }
        
        return embeddings
    
    def create_action_embedding(self, action_data):
        """
        Generate embeddings for action data.
        
        Args:
            action_data: Dictionary with action information
            
        Returns:
            dict: Dictionary with embeddings at different resolutions
        """
        # Convert action data to a state-like format for feature extraction
        state_like_format = self._action_to_state_format(action_data)
        
        # Use the state embedding function
        return self.create_state_embedding(state_like_format)
    
    def _action_to_state_format(self, action_data):
        """Convert action data to a format compatible with the feature extractor."""
        state_format = {}
        
        # Map action type to a categorical field
        if "action_type" in action_data:
            state_format["current_action"] = action_data["action_type"]
        
        # Map action parameters to appropriate fields
        if "action_params" in action_data:
            params = action_data["action_params"]
            for key, value in params.items():
                state_format[f"param_{key}"] = value
        
        # Include context if available
        if "context" in action_data:
            context = action_data["context"]
            for key, value in context.items():
                state_format[f"context_{key}"] = value
        
        # Include outcome if available
        if "outcome" in action_data:
            outcome = action_data["outcome"]
            if isinstance(outcome, dict):
                for key, value in outcome.items():
                    state_format[f"outcome_{key}"] = value
            else:
                state_format["outcome"] = outcome
        
        return state_format
    
    def reconstruct_from_embedding(self, embedding, tier="stm"):
        """
        Reconstruct approximate state from an embedding.
        
        Args:
            embedding: The embedding vector
            tier: The memory tier ("stm", "im", or "ltm")
            
        Returns:
            numpy.ndarray: Reconstructed feature vector
        """
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if tier == "stm":
                reconstructed = self.model.decode_from_stm(embedding_tensor)
            elif tier == "im":
                reconstructed = self.model.decode_from_im(embedding_tensor)
            elif tier == "ltm":
                reconstructed = self.model.decode_from_ltm(embedding_tensor)
            else:
                raise ValueError(f"Unknown tier: {tier}")
        
        return reconstructed.squeeze().cpu().numpy()
```

## Data Collection and Training Pipeline

```python
def collect_training_data(simulation, num_samples=10000):
    """
    Collect agent state data for training the autoencoder.
    
    Args:
        simulation: Simulation environment to collect data from
        num_samples: Number of samples to collect
        
    Returns:
        list: List of agent state dictionaries
    """
    training_data = []
    
    # Run simulation to collect data
    for _ in range(num_samples):
        simulation.step()
        
        # Collect states from all agents
        for agent in simulation.agents:
            state_data = agent.get_state_data()
            training_data.append(state_data)
    
    print(f"Collected {len(training_data)} agent state samples")
    return training_data

def train_autoencoder_pipeline(training_data, input_dim, batch_size=64, epochs=50):
    """
    Run the complete autoencoder training pipeline.
    
    Args:
        training_data: List of agent state dictionaries
        input_dim: Input dimension for the autoencoder
        batch_size: Batch size for training
        epochs: Number of training epochs
        
    Returns:
        AutoencoderTrainer: Trained model trainer
    """
    # Split into training and validation sets
    train_size = int(0.8 * len(training_data))
    train_data = training_data[:train_size]
    val_data = training_data[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Initialize and train the autoencoder
    trainer = AutoencoderTrainer(input_dim=input_dim)
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    return trainer
```

## Usage in the Memory System

```python
# Example of integrating the autoencoder with the memory system
def initialize_memory_system(agent_id):
    """Initialize the memory system with autoencoder-based embedding generation."""
    # Configure memory system
    memory_config = DefaultMemoryConfig()
    
    # Create embedding engine with trained autoencoder
    embedding_engine = AutoencoderEmbeddingEngine(
        model_path="models/state_autoencoder.pt",
        input_dim=512,  # Must match feature extractor output dimension
        stm_dim=384,
        im_dim=128,
        ltm_dim=32
    )
    
    # Create memory logger with custom embedding engine
    memory_logger = RedisAgentMemoryLogger(agent_id, memory_config)
    memory_logger.embedding_engine = embedding_engine
    
    return memory_logger
```

## Benefits of the Custom Autoencoder Approach

1. **Unified Embedding**: Generates consistent embeddings across all memory tiers from a single model

2. **Semantic Compression**: Learns to compress state information while preserving important semantic relationships

3. **Reconstruction Capability**: Allows approximate reconstruction of original state from compressed embeddings

4. **Dimensionality Flexibility**: Can be adjusted to balance between embedding quality and storage efficiency

5. **Feature Learning**: Automatically learns important features and relationships rather than using hand-engineered features

6. **Adaptability**: Can be retrained as the agent state representation evolves

7. **Efficiency**: Once trained, generates embeddings with a single forward pass through the network

## References

### Internal References

- [Core Concepts](core_concepts.md): Fundamental architecture and memory tier structure
- [Memory Agent](memory_agent.md): Integration of the autoencoder with memory management
- [Agent State Storage](agent_state_storage.md): Using embeddings for memory indexing and retrieval
- [Redis Integration](redis_integration.md): Storage of embeddings in the Redis backend
- [AgentMemory API](agent_memory_api.md): API methods for working with embeddings
- [Glossary](glossary.md): Definition of memory embedding and related terms

### Academic References

1. Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504-507.

2. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.

3. Chollet, F. (2016). Building autoencoders in keras. *The Keras Blog*. 

4. Li, C., Ovsjanikov, M., & Chazal, F. (2018). Persistence-based structural recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(4), 965-980.

5. Wang, Y., Yao, H., & Zhao, S. (2016). Auto-encoder based dimensionality reduction. *Neurocomputing*, 184, 232-242.

6. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.

7. Rae, J. W., Hunt, J. J., Danihelka, I., Harley, T., Senior, A. W., Wayne, G., ... & Lillicrap, T. P. (2016). Scaling memory-augmented neural networks with sparse reads and writes. *Advances in Neural Information Processing Systems*, 29.

### Implementation Resources

1. PyTorch Documentation: [Neural Networks Tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

2. Redis Vector Similarity Documentation: [Vector Similarity](https://redis.io/docs/stack/search/reference/vectors/)

3. FAISS Library: [Efficient Similarity Search](https://github.com/facebookresearch/faiss)

4. PyTorch Lightning: [Training Autoencoders](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html)

5. Scikit-learn: [Dimensionality Reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html) 