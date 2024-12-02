import os
import glob
import numpy as np
import cv2 as cv
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random


class MiniImagenetDataset(Dataset):
    def __init__(self, path, train=True, img_res=512):

        if isinstance(img_res, int):
            img_res = [img_res, img_res]

        if train:
            img_dir = os.path.join(path, "train")
        else:
            img_dir = os.path.join(path, "test")
        
        # Collect all image paths
        self.img_paths = glob.glob(f"{img_dir}/*.jpeg")
    
        self.transform = transforms.Compose(
            [
                transforms.AutoAugment(
                    transforms.AutoAugmentPolicy.IMAGENET,
                    transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomResizedCrop(img_res),
                transforms.ToTensor()
            ]
        )

    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        # Convert image to RGB if it is not already
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.transform(img)
        return img


if __name__ == "__main__":
    dataset = MiniImagenetDataset(path='./mini-imagenet/', train=False, img_res=512)
    print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for m in dataloader:
        print(m.shape)
        break