import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    def __init__(self, tokens, block_size, stride=None):
        self.tokens = tokens
        self.block_size = block_size
        self.stride = stride or block_size  # ← non-overlapping windows

    def __len__(self):
        return (len(self.tokens) - self.block_size) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        x = torch.tensor(
            self.tokens[start:start+self.block_size],
            dtype=torch.long
        )
        y = torch.tensor(
            self.tokens[start+1:start+self.block_size+1],
            dtype=torch.long
        )
        return x, y