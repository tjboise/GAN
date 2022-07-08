import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class myDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.img_names = os.listdir(self.data_dir)

    def __getitem__(self, index):
        path_img = os.path.join(self.data_dir, self.img_names[index])
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_names)

data_dir = "C:/Users/tjzhang/Documents/TJzhang/gan_for_crack/data/cracks/transverse"


mydata = myDataset(data_dir, transforms.ToTensor())
img,label=mydata[1]