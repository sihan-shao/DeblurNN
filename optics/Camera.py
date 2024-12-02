import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.fft import fft2, fftshift, ifft2, ifftshift
from optics.ElectricField import ElectricField, PlaneWave
from utils.Visualization_Helper import float_to_unit_identifier, add_colorbar
from optics.Element import *


class Camera(nn.Module):
    "Simple Camera Model"
    def __init__(self, configs):
    
        super().__init__()

        self.configs = configs

        aperture_diameter = float(self.configs['aperture_diameter'])  # the size of aperture
        delta_s = float(self.configs['delta_s']) # sampling interval of aperture
        N = 2 * int(aperture_diameter / delta_s)
        wavelengths = torch.tensor(self.configs['wavelength'])
        
        self.source = PlaneWave(height=N, width=N, A=1.0, wavelengths=wavelengths, spacing=delta_s)

        ref_foclen = self.configs['f_green']
        ref_n = self.configs['n_lambda'][1]
        zf = [self.configs['z_near'], self.configs['z_far']]
        zs = self.configs['zs']
        self.pupil = GeneralizedPupil(ref_foclen=ref_foclen, 
                                      ref_n=ref_n, 
                                      zf=zf, 
                                      zs=zs, 
                                      d=aperture_diameter)
        
        self.sensor = Sensor(noise=0.001)  # simplified sensor model with additive Gaussian noise
        
    
    def forward(self, imgs, batchsize=1):

        input_wave = self.source()
        
        self.pupil.calc_defocus(np.array(self.configs['n_lambda']), batchsize)

        psfs = self.pupil.to_psf(input_wave)

        blurred_imgs = self.sensor(imgs, psfs)

        return blurred_imgs # scale [0, 1]















