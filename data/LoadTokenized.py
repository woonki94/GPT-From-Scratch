import torch
import numpy as np
import pickle

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]

def load_tokenized_dataset(bin_path, meta_path, block_size=256):
    token_ids = np.memmap(bin_path, dtype=np.uint16, mode="r")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)


    vocab_size = meta["vocab_size"]

    num_blocks = len(token_ids) // block_size
    token_ids = token_ids[:num_blocks * block_size]
    token_ids = token_ids.reshape((num_blocks, block_size))

    token_tensor = torch.from_numpy(token_ids).long()
    return token_tensor, vocab_size

def get_tokenized_dataloader(batch_size, bin_path, meta_path, block_size=256):
    data_tensor, vocab_size = load_tokenized_dataset(bin_path, meta_path, block_size)
    dataset = TokenizedDataset(data_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,)
    return loader, vocab_size
