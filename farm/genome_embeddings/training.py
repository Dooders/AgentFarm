import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

class ContrastiveTrainer:
    """
    Handles the training loop for contrastive learning of genome embeddings.
    """

    def __init__(self, model, margin=1.0, lr=1e-3, device=None):
        """
        Args:
            model: GenomeEncoder instance
            margin: float, margin for contrastive loss
            lr: float, learning rate
            device: torch.device
        """
        self.model = model
        self.margin = margin
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device if device else torch.device("cpu")
        self.model.to(self.device)

    def contrastive_loss(self, emb1, emb2, label):
        """
        Basic contrastive loss:
          - label=1 for similar samples, label=0 for dissimilar
        """
        euclidean_distance = F.pairwise_distance(emb1, emb2)
        # L = 0.5 * [ label * distance^2 + (1-label) * max(margin - distance, 0)^2 ]
        pos_loss = label * torch.pow(euclidean_distance, 2)
        neg_loss = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = 0.5 * (pos_loss + neg_loss)
        return loss.mean()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            # We form pairs (item_i, item_j) for contrastive learning
            # For simplicity, let's assume the DataLoader or a collate_fn organizes pairs & labels
            emb1 = self.model(
                batch['generation1'].to(self.device),
                batch['parent_bits1'].to(self.device),
                batch['trait_bits1'].to(self.device),
            )
            emb2 = self.model(
                batch['generation2'].to(self.device),
                batch['parent_bits2'].to(self.device),
                batch['trait_bits2'].to(self.device),
            )
            labels = batch['label'].float().to(self.device)

            loss = self.contrastive_loss(emb1, emb2, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def fit(self, train_loader, val_loader=None, epochs=10):
        """Train loop with optional validation."""
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)

            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss}")

    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                emb1 = self.model(
                    batch['generation1'].to(self.device),
                    batch['parent_bits1'].to(self.device),
                    batch['trait_bits1'].to(self.device),
                )
                emb2 = self.model(
                    batch['generation2'].to(self.device),
                    batch['parent_bits2'].to(self.device),
                    batch['trait_bits2'].to(self.device),
                )
                labels = batch['label'].float().to(self.device)

                loss = self.contrastive_loss(emb1, emb2, labels)
                total_loss += loss.item()
        return total_loss / len(loader) 