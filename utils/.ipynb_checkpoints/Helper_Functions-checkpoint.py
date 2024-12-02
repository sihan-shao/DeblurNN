import matplotlib.pyplot as plt
import numpy as np
import pylab as plt

import torch
import torch.nn as nn

import imageio as io

import torch.nn.functional as F

from utils.units import *

import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def float_to_unit_identifier(val):
    """
    Takes a float value (e.g. 5*mm) and identifies which range it is
    e.g. mm , m, um etc.

    We always round up to the next 1000er decimal

    e.g.
    - 55mm will return mm
    - 100*m will return m
    - 0.1*mm will return um
    """
    exponent = np.floor(np.log10( val) / 3)
    unit_val = 10**(3*exponent)

    if unit_val == m:
        unit = "m"
    elif unit_val == mm:
        unit = "mm"
    elif unit_val == um:
        unit = "um"
    elif unit_val == nm:
        unit = "nm"
    return unit_val, unit


def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def ft2(input, delta=1, norm = 'ortho', pad = False):
    """
    Helper function computes a shifted fourier transform with optional scaling
    """
    return perform_ft(
        input=input,
        delta = delta,
        norm = norm,
        pad = pad,
        flag_ifft=False
    )

def ift2(input, delta=1, norm = 'ortho', pad = False):
    
    return perform_ft(
        input=input,
        delta = delta,
        norm = norm,
        pad = pad,
        flag_ifft=True
    )

def perform_ft(input, delta=1, norm = 'ortho', pad = False, flag_ifft : bool = False):
    
    # Get the initial shape (used later for transforming 6D to 4D)
    tmp_shape = input.shape

    # Save Size for later crop
    Nx_old = int(input.shape[-2])
    Ny_old = int(input.shape[-1])
        
    # Pad the image for avoiding convolution artifacts
    if pad == True:
        
        pad_scale = 1
        
        pad_nx = int(pad_scale * Nx_old / 2)
        pad_ny = int(pad_scale * Ny_old / 2)
        
        input = torch.nn.functional.pad(input, (pad_ny,pad_ny,pad_nx,pad_nx), mode='constant', value=0)
    
    if flag_ifft == False:
        myfft = torch.fft.fft2
        my_fftshift = torch.fft.fftshift
    else:
        myfft = torch.fft.ifft2
        my_fftshift = torch.fft.ifftshift


    
    # Compute the Fourier Transform
    out = (delta**2)* my_fftshift( myfft (  my_fftshift (input, dim=(-2,-1))  , dim=(-2,-1), norm=norm)  , dim=(-2,-1))
    
    if pad == True:
        input_size = [Nx_old, Ny_old]
        pool = torch.nn.AdaptiveAvgPool2d(input_size)
        
        if out.is_complex():
            out = pool(out.real) + 1j * pool(out.imag)
        else:
            out = pool(out)
    return out

def fftconv2d(image, kernel, mode="same"):
    """
    2D convolution using FFT.
    image and kernel should have same ndim.

    the Fourier transform of a convolution equals the multiplication of the Fourier transforms of the convolved signals.
    A similar property holds for the Laplace and z-transforms.
    However, it does not, in general, hold for the discrete Fourier transform.
    Instead, multiplication of discrete Fourier transforms corresponds to the 'circular convolution' of the corresponding time-domain signals.
    In order to compute the linear convolution of two sequences using the DFT, the sequences must be zero-padded to a length equal to the sum of the lengths of the two sequences minus one, i.e. N+M-1.
    """


    imH, imW = image.shape[-2], image.shape[-1]
    kH, kW = kernel.shape[-2], kernel.shape[-1]
    # zero-padded to a length equal to the sum of the lengths of the two sequences minus one, i.e. N+M-1
    size = (imH + kH - 1, imW + kW - 1)

    Fimage = torch.fft.fft2(image, s=size)
    Fkernel = torch.fft.fft2(kernel, s=size)

    Fconv = Fimage * Fkernel

    conv = torch.fft.ifft2(Fconv)

    if mode == "same":
        conv = conv[..., (kH // 2) : imH + (kH // 2), (kW // 2) : imW + (kW // 2)]
    if mode == "valid":
        conv = conv[..., kH - 1 : imH, kW - 1 : imW]
    # otherwise, full
    return conv


# ==================================
# Image batch quality evaluation
# ==================================
def batch_PSNR(img_clean, img):
    """Compute PSNR for image batch."""
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    Img_clean = (
        img_clean.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    )
    PSNR = 0.0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Img_clean[i, :, :, :], Img[i, :, :, :])
    return round(PSNR / Img.shape[0], 4)


def batch_SSIM(img, img_clean, multichannel=True):
    """Compute SSIM for image batch."""
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    Img_clean = (
        img_clean.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    )
    SSIM = 0.0
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Img_clean[i, ...], Img[i, ...], channel_axis=0)
    return round(SSIM / Img.shape[0], 4)

# ==================================
# Image batch normalization
# ==================================
def normalize_ImageNet(batch):
    """Normalize dataset by ImageNet(real scene images) distribution."""
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = (batch - mean) / std
    return batch_out


def denormalize_ImageNet(batch):
    """Convert normalized images to original images to compute PSNR."""
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = batch * std + mean
    return batch_out