import torch

def collate_fn(batch):
    if isinstance(batch, tuple) and isinstance(batch[0], list):
        return batch
    elif isinstance(batch, list):
        transposed = list(zip(*batch))
        return [collate_fn(samples) for samples in transposed]
    return torch.utils.data.default_collate(batch)
