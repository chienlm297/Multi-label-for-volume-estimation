import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import json
import os


class MyDataset(Dataset):
    def __init__(self, label_file, img_dir, img_dir2, transforms=None):
        # self.df = pd.read_csv(csv_file)
        self.label_file = open(label_file).readlines()
        self.img_dir = img_dir
        self.img_dir2 = img_dir2
        self.transforms = transforms

    def __getitem__(self, idx):
        label_dict = json.loads(self.label_file[idx])
        image_name = label_dict["image_name"]
        label = label_dict["label"]
        image = Image.open(os.path.join(self.img_dir, image_name)).convert("RGB")
        image_2 = Image.open(os.path.join(self.img_dir, image_name)).convert("RGB")
        label = torch.tensor(label, dtype=torch.float32)

        # d = self.df.iloc[idx]
        # image = Image.open(self.img_dir / d.image).convert("RGB")
        # image_2 = Image.open(self.img_dir / d.image).convert("RGB")
        # label = torch.tensor(d[1:].tolist(), dtype=torch.float32)
        if self.transforms is not None:
            image = self.transforms(image)
            image_2 = self.transforms(image_2)
        return image, image_2, label

    def __len__(self):
        return len(self.label_file)
        # return len(self.df)
