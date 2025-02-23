from torch.utils.data import  Dataset
from PIL import Image
import os
import requests


class AnimalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'Filepath']
        image = Image.open(img_path).convert('RGB')
        label = int(self.dataframe.loc[idx, 'Label_idx'])
        if self.transform:
            image = self.transform(image)
        return image, label


def download_file(url, target_path):
    """Download file from URL if it doesn't exist locally."""
    if not os.path.exists(target_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        print(f"Downloading weights from {url} to {target_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # raise an error on failure
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        print("Download complete!")
    else:
        print("Weights file already exists.")
