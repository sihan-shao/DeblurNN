import numpy as np

aperture_diameter = 7e-3
f_green = 35e-3  # focal length of the lens for green channel
zf = 2  # object to lens distance in meters
zs = (1 / f_green - 1 / zf) ** (-1)  # lens to image distance in meters

wavelength = np.array([630.0e-9, 525.0e-9, 458.0e-9])  # red, green, blue
n_lambda = [1.4571, 1.4610, 1.4650]  # refractive indices for red, green, blue wavelengths
n_green = n_lambda[1]  # green
f_lambda = f_green * (n_green - 1) / (n_lambda - 1)
delta_s = 50e-6 # lens spatial sampling interval in meters
z_near = 1.84  # nearest object depth in meters
z_far = 2.20  # farthest object depth in meters
