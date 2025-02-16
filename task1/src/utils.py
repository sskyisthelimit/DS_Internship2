import torch
from idx2numpy import convert_from_file
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    def __init__(self, images_filepath, labels_filepath):
        super().__init__()
        self.images = convert_from_file(images_filepath)
        self.labels = convert_from_file(labels_filepath)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0) / 255
        label = torch.tensor(self.labels[idx], dtype=torch.int32)

        return image, label