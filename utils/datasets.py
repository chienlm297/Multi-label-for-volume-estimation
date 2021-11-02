import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        image = Image.open(self.img_dir / d.image).convert("RGB")
        label = torch.tensor(d[1:].tolist(), dtype=torch.float32)

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.df)
