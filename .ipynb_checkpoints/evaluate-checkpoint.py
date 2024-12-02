# Evaluation function
import yaml
import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from optics.Element import *
from optics.Camera import Camera
from utils.Helper_Functions import *
from networks.unet import UNet

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


def evaluate(sharp_image: np.ndarray, z: float) -> np.ndarray:
    # sharp_image: the ground truth image to be evaluated with a shape of (H, W, 3)
    # z: the object depth in meters
    # returns: deblurred image with a shape of (H, W, 3) through neural network
    H, W, C = sharp_image.shape

    transform = transforms.Compose(
                [
                    transforms.Resize([256, 256]),
                    transforms.ToTensor()
                ]
            )

    # reshape the input image to [1, 3, H, W]
    #input = torch.from_numpy(sharp_image).reshape(1, C, H, W).to(device)
    #pil_image = Image.fromarray(sharp_image)
    input = transform(Image.fromarray(sharp_image)).reshape(1, C, H, W).to(device)

    # only one distance can be sampled
    camera.pupil.zf_min = z
    camera.pupil.zf_max = z

    blurred_img = camera(input).float()
    deblurred_img = network(blurred_img)

    output_input_gt = torch.cat((input, blurred_img, deblurred_img), dim=0)

    save_image(output_input_gt, f"./gt_blurred_deblurred_results.png")

    # reshape and detach from GPU, then convert to numpy array
    output = deblurred_img.reshape(H, W, C).detach()

    # rescale image from [0, 1] to [0, 255]
    output= output.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()

    return output


# -----------------------------------------------------------------------------------------

# read a image
img_path = './image_44285.jpeg'
image = np.array(Image.open(img_path))


# evaluate the imaging model and neural network
output = evaluate(image, z=2.0)
print(output.shape)

# Evaluation Metrics
ssim = compare_ssim(image, output, channel_axis=2)
psnr = compare_psnr(image, output)
print("Structural Similarity Index: {}".format(ssim))
print("Peak Signal-to-Noise Ratio: {}".format(psnr))







