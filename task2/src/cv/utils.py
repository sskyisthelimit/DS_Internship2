from torch.utils.data import  Dataset
from PIL import Image

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

