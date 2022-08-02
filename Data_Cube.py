import torch
from torch.utils.data import Dataset


class DataCube(Dataset):

    def __init__(self, data, labels, transform=None):
        super().__init__()
        self.data = torch.from_numpy(data).to(torch.float)
        self.labels = labels
        self.labels = torch.from_numpy(self.labels).to(torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index, :]
        label = self.labels[index]
        if self.transform is not None:
            d = self.transform(d)
        return d, label
