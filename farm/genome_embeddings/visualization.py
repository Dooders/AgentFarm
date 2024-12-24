import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(embeddings, labels, title="t-SNE Genome Embeddings"):
    """
    Use t-SNE to reduce embeddings to 2D for visualization.
    embeddings: np.ndarray of shape (num_samples, embedding_dim)
    labels: list or array of annotation labels
    """
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show() 