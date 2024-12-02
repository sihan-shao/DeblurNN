import torch

def gaussian_noise(image, std_dev=0.001):
    return image + torch.randn_like(image) * std_dev

    