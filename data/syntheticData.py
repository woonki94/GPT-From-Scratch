# synthetic_data.py

import torch
from torch.utils.data import Dataset, DataLoader

# Vocabulary
def build_vocab():
    vocab = {
        "dog": 0,
        "cat": 1,
        "man": 2,
        "woman": 3,
        "eats": 4,
        "loves": 5,
        "hates": 6,
        "meat": 7,
        "fish": 8,
        "rice": 9,
        "PAD": 10
    }
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab, len(vocab)

# Synthetic sentences (SVO grammar)
def build_sentences():
    return [
        ["dog", "eats", "meat"],
        ["cat", "eats", "fish"],
        ["man", "loves", "woman"],
        ["woman", "hates", "man"],
        ["cat", "hates", "rice"],
        ["dog", "loves", "fish"],
        ["man", "eats", "rice"],
        ["woman", "eats", "meat"],
        ["dog", "eats", "rice"],
        ["cat", "loves", "man"]
    ]

# Tokenize and create input/target pairs
def tokenize_sentences(sentences, vocab):
    tokenized = [[vocab[word] for word in sent] for sent in sentences]
    inputs = [seq[:-1] for seq in tokenized]   # e.g., ["dog", "eats"]
    targets = [seq[1:] for seq in tokenized]   # e.g., ["eats", "meat"]
    return inputs, targets

# Dataset class
class TinyWordDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Factory function to get data loader
def get_synthetic_dataloader(batch_size=4, shuffle=True):
    vocab, inv_vocab, vocab_size = build_vocab()
    sentences = build_sentences()
    inputs, targets = tokenize_sentences(sentences, vocab)
    dataset = TinyWordDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), vocab, inv_vocab, vocab_size

# Exported objects
__all__ = [
    "get_synthetic_dataloader",
    "build_vocab",
    "build_sentences",
    "tokenize_sentences",
    "TinyWordDataset"
]
