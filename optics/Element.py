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
from optics.ElectricField import ElectricField
from utils.Visualization_Helper import float_to_unit_identifier, add_colorbar

class GeneralizedPupil(nn.Module):
    "Generalized Pupil Function, considering the chromatic abberation"
    def __init__(self, ref_foclen, ref_n, zf, zs, d):
        """
        Parameters

        ref_foclen: reference focal length 
        ref_n : reference refractive index 
        zf: distance from the object to the lens list [zf_min, zf_max]
        zs: distance from the lens to the image sensor
        d: aperture diameter
		"""
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ref_foclen = ref_foclen
        self.ref_n = ref_n

        zf = [zf, zf] if isinstance(zf, float) else zf
        self.zf_min = zf[0]
        self.zf_max = zf[1]
        
        self.zs = zs
        self.r = d / 2
        
        self.aberration = None

    def calc_defocus(self, ns, batch_size=1):
        """
        Parameters:

        ns: list of refractive index
        batch_size: the number of sampled distance zf

            if batch_size = 1, only one object distance is used in one batch_size during training
            otherwise, we samples different object distance in one batch size
        """
        
        # calculate the focal lengths of corresponding R G B wavelength
        self.foclens = torch.from_numpy(
            self.ref_foclen * (self.ref_n - 1) / (np.array(ns) - 1)
            ).view(1, -1, 1, 1)
        
        # sample object distances
        self.zfs = torch.from_numpy(
            np.random.uniform(self.zf_min, self.zf_max, batch_size)
            ).view(-1, 1, 1, 1)

        # calculate the defocus parameter
        self.delta_Ds = (1 / self.zfs + 1 / self.zs - 1 / self.foclens).to(self.device)

    def to_aberration(self, field):

        wavelengths = field.wavelengths[None, :, None, None]
        dx, dy = field.spacing[0], field.spacing[1]
        height, width = field.height, field.width

        X, Y = torch.meshgrid(torch.linspace(-dx * width / 2, dx * width / 2, width, dtype=dx.dtype), 
                              torch.linspace(-dy * height / 2, dy * height / 2, height, dtype=dy.dtype), 
                              indexing='xy')
        R = torch.sqrt(X**2 + Y**2).to(self.device)

        # apply phase aberration to the field
        phi_lens = 1j * torch.pi / wavelengths * self.delta_Ds * R[None, None, :, :]**2
        aberration = field.data * torch.exp(phi_lens)

        # apply aperture to the field
        aper_lens = R < self.r
        aberration *= aper_lens

        self.aberration = ElectricField(
				data=aberration,
				wavelengths=field.wavelengths,
				spacing = field.spacing
		)

        return self.aberration

    def to_psf(self, field, padding=True):
        
        if self.aberration is None:
            field = self.to_aberration(field)
        else:
            field = self.aberration
        # padding 
        if padding:
            Horg, Worg = field.height, field.width
            Hpad, Wpad = Horg // 4, Worg // 4
            Himg, Wimg = Horg + 2 * Hpad, Worg + 2 * Wpad
            padded_field = pad(field.data, (Wpad, Wpad, Hpad, Hpad), mode='constant', value=0)
        
        else:
            Himg, Wimg = field.height, field.width
        
        # Computational fourier optics. Chapter 5, section 5.5.
        # obs sample interval
        self.dx_obs = field.wavelengths * self.zs / Himg / field.spacing[0]
        self.dy_obs = field.wavelengths * self.zs / Himg / field.spacing[1]
        
        field_data = ifftshift(fft2(fftshift(padded_field)))


        if padding:
            center_crop = torchvision.transforms.CenterCrop([Horg, Worg])
            field_data = center_crop(field_data)
        
        self.psfs = field_data.abs()**2 / torch.sum(field_data.abs()**2, dim=[2, 3], keepdim=True)

        return self.psfs

    def show_psf(self, wavelength, flag_axis=True, figsize=(8, 8)):

        if figsize is not None:
            fig = plt.figure(figsize=figsize, constrained_layout=True)
        
        grid = fig.add_gridspec(4, 4, hspace=0.1, wspace=0.1)
        wavelength = float(wavelength)
        idx = (self.aberration.wavelengths == wavelength).nonzero()[0]
        z_obj = float(self.zfs[0,0,0,0])
        psf = self.psfs[0, idx, :, :].squeeze()
        dx, dy = self.dx_obs[idx].detach().cpu(), self.dy_obs[idx].detach().cpu()
        height, width = psf.shape[0], psf.shape[1]
        cross_section_x = psf[int(height / 2), :]
        cross_section_y = psf[:, int(width / 2)]

        if flag_axis == True:
            size_x = np.array(dx / 2.0 * height)
            size_y =np.array(dy / 2.0 * width)
            
            unit_val, unit_axis = float_to_unit_identifier(max(size_x,size_y))
            size_x = size_x / unit_val
            size_y = size_y / unit_val
            
            extent = [-size_y, size_y, -size_x, size_x]

        else:
            extent = None
            size_x = height
            size_y = width

        unit_val, unit = float_to_unit_identifier(wavelength)
        wavelength = wavelength / unit_val

        # plot intensity of point spread function
        main_ax = fig.add_subplot(grid[:-1, :-1])  # Main plot spans most of the grid
        main_plot = main_ax.imshow(psf.detach().cpu(), cmap='viridis', extent=extent, origin='lower', aspect='auto')
        main_ax.set_title("Intensity| wavelength = " + str(round(wavelength, 2)) + str(unit) + "| z_obj = " + str(round(z_obj, 2)) + 'm')

        if flag_axis:
            if unit != "":
                main_ax.set_xlabel("Position (" + unit_axis + ")")
                main_ax.set_ylabel("Position (" + unit_axis + ")")

        # Cross-section on the right
        right_ax = fig.add_subplot(grid[:-1, -1], sharey=main_ax)
        right_ax.plot(cross_section_y.detach().cpu(), np.linspace(-size_y, size_y, width), 'r')
        right_ax.set_xlabel('Intensity')
        right_ax.set_title('Vertical cross-section')
        right_ax.tick_params(labelleft=False)

        # Cross-section at the bottom
        bottom_ax = fig.add_subplot(grid[-1, :-1], sharex=main_ax)
        bottom_ax.plot(np.linspace(-size_x, size_x, height), cross_section_x.detach().cpu(), 'b')
        bottom_ax.set_ylabel('Intensity')
        bottom_ax.set_title('Horizontal cross-section')
        bottom_ax.tick_params(labelbottom=False)
        
        if flag_axis:
            plt.axis("on")
        else:
            plt.axis("off")


class Sensor(nn.Module):
    "Image sensor class. only square sensor is considered"
    def __init__(self, res=None, ps=None, noise=None):
        """
        Parameters

        res: resolution of camera sensor
        ps: pixle size of camera sensor
        noise: std_dev of guassin noise
        conv_type: convolution in frequency or spatial domain
		"""
        super().__init__()

        self.res = [res, res] if isinstance(res, int) else res
        self.ps = ps
        self.noise = noise

    def area_resampling(self, psfs):
        """
        Usually the sample interval of wave propagation is not the same as the interval of camera sensor
        Thus, we need to resampling the point spread functions with resolution and pixel size of sensor

        In this case, we don't need this function right now (can be implemented later)
        """
        return NotImplemented

    def img_psf_fftconv(self, image, kernel, mode="same"):
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
            conv = conv[..., (kH // 2) : imH + (kH // 2), (kW // 2) : imW + (kW // 2)].abs()
        if mode == "valid":
            conv = conv[..., kH - 1 : imH, kW - 1 : imW].abs()
        # otherwise, full

        if self.noise is not None:
            conv += torch.randn_like(conv) * self.noise

        return conv

    def forward(self, imgs, psfs):
        
        if self.res and self.ps is not None:
            psfs = self.area_resampling(psfs)
        
        return self.img_psf_fftconv(imgs, psfs)

class ApertureElement(nn.Module):
    
    def __init__(self,
                 aperture_type      : str = 'circ',
                 aperture_size      : float = None, 
                 device             : torch.device = None
                 ):
        """
		This implements an aperture
        aperture_type:
            circ: add a circle aperture to the field
            rect: add a rectangle aperture to the field
        
        aperture_size:
            The size of defined aperture, can't be larger than field size in the simulation
            if aperture_type is circ, the aperture_size should be the radius
            if aperture_type is rect, the aperture_size should be width and length

		"""
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.aperture_type = aperture_type
        self.aperture_size = aperture_size

    
    
    def add_circ_aperture_to_field(self,
                                   input_field    : ElectricField, 
                                   radius=None)-> torch.Tensor: 
        

        dx, dy = input_field.spacing[0], input_field.spacing[1]
        height, width = input_field.height, input_field.width
        radius = torch.tensor(radius)
        if radius==None:
            r = min([dx * height, dy * width]) / 2.0
            
        if radius!=None and radius < min([dx * height, dy * width]) / 2.0:
            r = radius
        else:
            ValueError('The radius should not larger than the physical length of E-field ')
            
            
        x = torch.linspace(-dx * height / 2, dx * height / 2, height, dtype=dx.dtype)
            
        y = torch.linspace(-dy * width / 2, dy * width / 2, width, dtype=dy.dtype)
                
        X, Y = torch.meshgrid(x, y, indexing='xy')
            
        R = torch.sqrt(X**2 + Y**2)
            
        # Create a mask that is 1 inside the circle and 0 outside
        Mask = torch.where(R <= r, 1, 0)
        Mask = Mask[None, None, :, :]
        
        return Mask.to(self.device)
    
    def add_rect_aperture_to_field(self,
                                   input_field, 
                                   rect_width=None, 
                                   rect_height=None)-> torch.Tensor:
        
        
        dx, dy = input_field.spacing[0], input_field.spacing[1]
        height, width = input_field.height, input_field.width
            
        if rect_width is None:
            rect_width = dx * width / 2
        if rect_height is None:
            rect_height = dy * height / 2
            
        # Ensure the rectangle dimensions are within the field's dimensions
        rect_width = min(rect_width, dx * width)
        rect_height = min(rect_height, dy * height)
            
        x = torch.linspace(-dx * width / 2, dx * width / 2, width, dtype=dx.dtype)
        y = torch.linspace(-dy * height / 2, dy * height / 2, height, dtype=dy.dtype)

        X, Y = torch.meshgrid(x, y, indexing='xy')

        # Create a mask that is 1 inside the rectangle and 0 outside
        Mask = torch.where((torch.abs(X) <= rect_width / 2) & (torch.abs(Y) <= rect_height / 2), 1, 0)
        Mask = Mask[None, None, :, :]
        
        return Mask.to(self.device)
    
    
    def forward(self,
                field: ElectricField
                ) -> ElectricField:
        
        """
		Args:
			field(torch.complex128) : Complex field (MxN).
		"""

        if self.aperture_type == 'circ':
            self.aperture = self.add_circ_aperture_to_field(field, 
                                                            radius=self.aperture_size)
        
        elif self.aperture_type == 'rect':
            self.aperture = self.add_rect_aperture_to_field(field, 
                                                            rect_height=self.aperture_size, 
                                                            rect_width=self.aperture_size)
        elif self.aperture_type == None:
            self.aperture = torch.ones_like(field.data)
        
        else:
            ValueError('No exisiting aperture shape, please define by yourself')
            
        out_field = self.aperture * field.data
        
        Eout = ElectricField(
				data=out_field,
				wavelengths=field.wavelengths,
				spacing = field.spacing
		)
        
        return Eout


class ApertureElement(nn.Module):
    
    def __init__(self,
                 aperture_type      : str = 'circ',
                 aperture_size      : float = None, 
                 device             : torch.device = None
                 ):
        """
		This implements an aperture
        aperture_type:
            circ: add a circle aperture to the field
            rect: add a rectangle aperture to the field
        
        aperture_size:
            The size of defined aperture, can't be larger than field size in the simulation
            if aperture_type is circ, the aperture_size should be the radius
            if aperture_type is rect, the aperture_size should be width and length

		"""
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.aperture_type = aperture_type
        self.aperture_size = aperture_size

    
    
    def add_circ_aperture_to_field(self,
                                   input_field    : ElectricField, 
                                   radius=None)-> torch.Tensor: 
        

        dx, dy = input_field.spacing[0], input_field.spacing[1]
        height, width = input_field.height, input_field.width
        radius = torch.tensor(radius)
        if radius==None:
            r = min([dx * height, dy * width]) / 2.0
            
        if radius!=None and radius < min([dx * height, dy * width]) / 2.0:
            r = radius
        else:
            ValueError('The radius should not larger than the physical length of E-field ')
            
            
        x = torch.linspace(-dx * height / 2, dx * height / 2, height, dtype=dx.dtype)
            
        y = torch.linspace(-dy * width / 2, dy * width / 2, width, dtype=dy.dtype)
                
        X, Y = torch.meshgrid(x, y)
            
        R = torch.sqrt(X**2 + Y**2)
            
        # Create a mask that is 1 inside the circle and 0 outside
        Mask = torch.where(R <= r, 1, 0)
        Mask = Mask[None, None, :, :]
        
        return Mask.to(self.device)
    
    def add_rect_aperture_to_field(self,
                                   input_field, 
                                   rect_width=None, 
                                   rect_height=None)-> torch.Tensor:
        
        
        dx, dy = input_field.spacing[0], input_field.spacing[1]
        height, width = input_field.height, input_field.width
            
        if rect_width is None:
            rect_width = dx * width / 2
        if rect_height is None:
            rect_height = dy * height / 2
            
        # Ensure the rectangle dimensions are within the field's dimensions
        rect_width = min(rect_width, dx * width)
        rect_height = min(rect_height, dy * height)
            
        x = torch.linspace(-dx * width / 2, dx * width / 2, width, dtype=dx.dtype)
        y = torch.linspace(-dy * height / 2, dy * height / 2, height, dtype=dy.dtype)

        X, Y = torch.meshgrid(x, y, indexing='xy')

        # Create a mask that is 1 inside the rectangle and 0 outside
        Mask = torch.where((torch.abs(X) <= rect_width / 2) & (torch.abs(Y) <= rect_height / 2), 1, 0)
        Mask = Mask[None, None, :, :]
        
        return Mask.to(self.device)
    
    
    def forward(self,
                field: ElectricField
                ) -> ElectricField:
        
        """
		Args:
			field(torch.complex128) : Complex field (MxN).
		"""

        if self.aperture_type == 'circ':
            self.aperture = self.add_circ_aperture_to_field(field, 
                                                            radius=self.aperture_size)
        
        elif self.aperture_type == 'rect':
            self.aperture = self.add_rect_aperture_to_field(field, 
                                                            rect_height=self.aperture_size, 
                                                            rect_width=self.aperture_size)
        elif self.aperture_type == None:
            self.aperture = torch.ones_like(field.data)
        
        else:
            ValueError('No exisiting aperture shape, please define by yourself')
            
        out_field = self.aperture * field.data
        
        Eout = ElectricField(
				data=out_field,
				wavelengths=field.wavelengths,
				spacing = field.spacing
		)
        
        return Eout
