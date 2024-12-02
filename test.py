# Evaluation function
import yaml
import os
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from dataset import MiniImagenetDataset
from torch.utils.data import DataLoader
from optics.Element import *
from optics.Camera import Camera
from utils.Helper_Functions import *
from networks.unet import UNet
from networks.restormer import Restormer

path = os.path.dirname(
                    os.path.abspath(__file__),
    )
    
# Load .yaml vars
config_path = './checkpoint/model_pretrain/config_unet.txt'
with open(config_path, 'r') as file:
    loaded_config = yaml.safe_load(file)

camera_config = loaded_config['optics']
network_config = loaded_config['network']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define a imaging system 
camera = Camera(camera_config).to(device)

# define a neural networt
if network_config['net'] == 'unet':
    network = UNet(in_channels=3, out_channels=3).to(device)
        
elif network_config['net'] == 'restormer':
    network = Restormer(inp_channels=3, out_channels=3).to(device)

# load pre-trained parameters of NN
model_path = './checkpoint/model_pretrain/Unet_L1.pth'
state_dict = torch.load(model_path, map_location=device)
NN_param = state_dict['network_state_dict']
network.load_state_dict(NN_param)

batch_size = 8
data_path = './mini-imagenet'
valid_dataset = MiniImagenetDataset(data_path, train=False, img_res=network_config['image_size'])
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

# evaluate on test dataset
validate_epoch_psnr = []
validate_epoch_ssim = []
with tqdm(valid_dataloader, unit="batch") as tepoch:
    for data in tepoch:
        valid_inputs = data
        valid_inputs = valid_inputs.to(device)

        blurred_img = camera(valid_inputs).float()
        deblurred_img = network(blurred_img)
                        
        validate_epoch_psnr.append(batch_PSNR(valid_inputs, deblurred_img))
        validate_epoch_ssim.append(batch_SSIM(valid_inputs, deblurred_img))


validate_psnr = np.stack(validate_epoch_psnr).mean()
validate_ssim = np.stack(validate_epoch_ssim).mean()

print("Structural Similarity Index: {}".format(validate_ssim))
print("Peak Signal-to-Noise Ratio: {}".format(validate_psnr))







