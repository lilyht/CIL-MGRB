from torch.utils.data import Dataset
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class FaultDataset2d(Dataset):

    def __init__(self, data_array, label_array, transform=None):
        """
        Args:
            data_array (string): array of data which have been cached.
            label_array (string): array of labels which have been cached.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_array = data_array
        self.labels = np.array(label_array)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data_array[:, :, idx]
        label = self.labels[idx]

        data = torch.tensor(data, dtype=torch.float32)
        data = data.reshape(1, data.shape[0], data.shape[1])

        return data, label
