import torch
from torch.utils.data import Dataset

class GenomeDataset(Dataset):
    """
    PyTorch dataset for genome metadata, intended for contrastive training of embeddings.
    Each sample includes generation index, parent signature bits, trait signature bits,
    as well as a label/family ID for grouping or comparison.
    """

    def __init__(self, genome_records):
        """
        Args:
            genome_records (List[dict]): A list of dicts with keys like
                {
                    'generation': int,
                    'parent_bits': List[0/1 or float],
                    'trait_bits': List[0/1 or float],
                    'family_id': int,
                    ...
                }
        """
        self.genome_records = genome_records

    def __len__(self):
        return len(self.genome_records)

    def __getitem__(self, idx):
        record = self.genome_records[idx]

        generation = record['generation']
        parent_bits = torch.FloatTensor(record['parent_bits'])
        trait_bits = torch.FloatTensor(record['trait_bits'])
        family_id = record['family_id']  # used for labeling in contrastive training

        return {
            'generation': torch.tensor(generation, dtype=torch.long),
            'parent_bits': parent_bits,
            'trait_bits': trait_bits,
            'family_id': family_id
        } 
