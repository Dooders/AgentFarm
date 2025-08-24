import cv2
import faiss
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel


class LatentSpace:
    def __init__(self, dimension=768):
        """
        Centralized latent space for storing and retrieving embeddings from different senses.
        """
        self.index = faiss.IndexFlatL2(
            dimension
        )  # Vector database for storing embeddings
        self.embeddings = []  # Stores embeddings
        self.metadata = []  # Stores metadata for each embedding

    def add_embedding(self, embedding: np.ndarray, metadata: dict):
        """
        Add an embedding to the centralized latent space.
        """
        # Ensure embedding is 2D array for FAISS
        embedding_2d = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(x=embedding_2d, n=1)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)

    def search(self, embedding: np.ndarray, top_k: int = 5):
        """
        Search for the most similar embeddings in the latent space.
        """
        # Ensure embedding is 2D array for FAISS
        embedding_2d = embedding.reshape(1, -1).astype(np.float32)
        distances = np.zeros((1, top_k), dtype=np.float32)
        indices = np.zeros((1, top_k), dtype=np.int64)
        self.index.search(
            x=embedding_2d, k=top_k, n=1, distances=distances, labels=indices
        )
        results = [
            (self.metadata[i], distances[0][idx]) for idx, i in enumerate(indices[0])
        ]
        return results


class Senses:
    def __init__(self, latent_space, model_name="google/vit-base-patch16-224-in21k"):
        """
        Initialize the Senses module for visual chart embedding and comparison.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device).eval()  # type: ignore
        self.latent_space = latent_space  # Connect to the centralized latent space

    def preprocess_image(self, image_path):
        """
        Preprocess the chart image for embedding extraction.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.extractor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def generate_embedding(self, image_path):
        """
        Convert an image into a latent space embedding using ViT.
        """
        pixel_values = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(pixel_values)
            embedding = (
                outputs.last_hidden_state[:, 0, :].cpu().numpy()
            )  # Extract CLS token embedding
        return embedding

    def store_embedding(self, image_path, chart_id):
        """
        Generate and store an embedding for a chart image in the central latent space.
        """
        embedding = self.generate_embedding(image_path)
        self.latent_space.add_embedding(embedding, {"id": chart_id, "type": "visual"})

    def compare_charts(self, image_path, top_k=5):
        """
        Compare a new chart to stored embeddings and find the most similar ones.
        """
        embedding = self.generate_embedding(image_path)
        return self.latent_space.search(embedding, top_k)


# Example usage
if __name__ == "__main__":
    latent_space = LatentSpace()
    senses = Senses(latent_space)

    # Store a few sample charts (modify paths as needed)
    senses.store_embedding("chart1.png", "Chart 1")
    senses.store_embedding("chart2.png", "Chart 2")
    senses.store_embedding("chart3.png", "Chart 3")

    # Compare a new chart
    results = senses.compare_charts("new_chart.png")
    print("Most similar charts:", results)
