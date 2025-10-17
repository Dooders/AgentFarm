import torch
import torch.nn as nn
import torch.nn.functional as F

class GenomeEncoder(nn.Module):
    """
    Neural encoder that converts genome metadata (generation, parent hash, trait hash, etc.)
    into fixed-dimension embeddings.
    """

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Example sub-embeddings: generation embedding, parent hash embedding, trait hash embedding
        # Adjust sizes to fit your data's ranges
        self.generation_embed = nn.Embedding(num_embeddings=1000, embedding_dim=8)  # Example

        # Linear layers to process parent and trait hash bits
        self.parent_fc = nn.Linear(64, 16)  # Example: assume 64 bits for parent
        self.trait_fc = nn.Linear(64, 16)   # Example: assume 64 bits for traits

        # Combine embeddings into final representation
        combined_size = 8 + 16 + 16
        self.fc_out = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim),
        )

    def forward(self, generation: torch.Tensor, parent_bits: torch.Tensor, trait_bits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
            generation: LongTensor of shape (batch_size,) - assumed discrete generation indices
            parent_bits: FloatTensor of shape (batch_size, 64)
            trait_bits: FloatTensor  of shape (batch_size, 64)

        Returns:
            embeddings: FloatTensor of shape (batch_size, embedding_dim)
        """
        gen_emb = self.generation_embed(generation)                        # (batch_size, 8)
        parent_emb = F.relu(self.parent_fc(parent_bits))                   # (batch_size, 16)
        trait_emb = F.relu(self.trait_fc(trait_bits))                      # (batch_size, 16)

        combined = torch.cat([gen_emb, parent_emb, trait_emb], dim=1)      # (batch_size, 8+16+16)
        embeddings = self.fc_out(combined)                                 # (batch_size, embedding_dim)
        return embeddings 
