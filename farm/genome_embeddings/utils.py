import torch

def save_model(model, path):
    """Save the model state_dict to disk."""
    torch.save(model.state_dict(), path)

def load_model(model, path, device=None):
    """Load a model state_dict from disk."""
    checkpoint = torch.load(path, map_location=device or 'cpu')
    model.load_state_dict(checkpoint)
    return model

def compute_embedding(model, generation, parent_bits, trait_bits, device=None):
    """Compute a single embedding vector using the provided model."""
    model.eval()
    if device is None:
        device = torch.device("cpu")

    generation_tensor = torch.tensor([generation], dtype=torch.long).to(device)
    parent_tensor = torch.FloatTensor([parent_bits]).to(device)
    trait_tensor = torch.FloatTensor([trait_bits]).to(device)

    with torch.no_grad():
        embedding = model(generation_tensor, parent_tensor, trait_tensor)
    return embedding.cpu().numpy()[0]  # return as numpy array 
