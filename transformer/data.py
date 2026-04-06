import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_batch(dataset, batch_size, context_length, device):
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # 计算最大起始索引（确保有足够的长度）
    max_start_idx = len(dataset) - context_length
    
    # 随机选择起始索引
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # 使用向量化操作提取输入和标签
    input_sequences = []
    label_sequences = []
    
    for start_idx in start_indices:
        # 输入：从start_idx到start_idx+context_length-1
        input_seq = dataset[start_idx:start_idx + context_length]
        # 标签：从start_idx+1到start_idx+context_length
        label_seq = dataset[start_idx + 1:start_idx + context_length + 1]
        
        input_sequences.append(input_seq)
        label_sequences.append(label_seq)
    
    # 转换为张量
    inputs = torch.tensor(np.array(input_sequences), dtype=torch.long)
    labels = torch.tensor(np.array(label_sequences), dtype=torch.long)
    
    # 移动到指定设备
    return inputs.to(device), labels.to(device)


class CausalMemmapDataset(Dataset):
    def __init__(self, data_path, context_length, start_block=0, end_block=None):

        # 确保dtype一致，通常语料索引使用int32足够
        self.data = np.memmap(data_path, mode='r', dtype=np.int32)
        self.context_length = context_length

        total_blocks = (len(self.data) - context_length - 1) // context_length

        if end_block is None:
            end_block = total_blocks

        self.start_block = start_block
        self.end_block = end_block
        self.num_blocks = end_block - start_block

        # 简单的边界检查
        if self.num_blocks <= 0:
            print(f"Warning: Dataset has 0 blocks. (Start: {start_block}, End: {end_block})")

    def __len__(self):
        return max(0, self.num_blocks)

    def __getitem__(self, idx):
        block_idx = self.start_block + idx
        start_idx = block_idx * self.context_length

        # 转换为int64给torch使用
        x = torch.from_numpy(
            self.data[start_idx: start_idx + self.context_length].astype(np.int64)
        )

        y = torch.from_numpy(
            self.data[start_idx + 1: start_idx + self.context_length + 1].astype(np.int64)
        )

        return x, y