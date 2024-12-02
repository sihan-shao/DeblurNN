import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import yaml
import numpy as np
import scipy
from tqdm import tqdm
import time
from datetime import datetime
from optics.Element import *
from optics.Camera import Camera
from utils.Helper_Functions import *
from utils.losses import *
from networks.unet import UNet
from networks.nafnet import NAFNet
from networks.restormer import Restormer
from dataset import MiniImagenetDataset
## Distributed Parallel Training 
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class DDPTrainer:
    
    def __init__(self, 
                 config_file,
                 data_path, 
                 model_dir, 
                 logs_dir) -> None:
        
        init_process_group(backend="nccl")
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        
        self.verbose = dist.get_rank() == 0
        # configuration files
        self.config_file = config_file
        self.camera_config = self.config_file['optics']
        self.network_config = self.config_file['network']
        self.data_path = data_path
        self.model_dir = model_dir
        self.logs_dir  = logs_dir
        
        # Build a camera
        self.camera = Camera(self.camera_config).cuda()

        self.network_config['net'] = 'nafnet'                             
        # Build reconstruction NN
        if self.network_config['net'] == 'unet':
            self.network = UNet(in_channels=3, out_channels=3)

        elif self.network_config['net'] == 'nafnet':
            self.network = NAFNet(in_chan=3, out_chan=3)
        
        elif self.network_config['net'] == 'restormer':
            self.network = Restormer(inp_channels=3, out_channels=3)

        else:
            NotImplemented
        
        # define the loss function
        self.L1 = nn.L1Loss()
        self.SSIMLoss = SSIMLoss()
        self.PSNRLoss = PSNRLoss()

        
        self.network = nn.SyncBatchNorm.convert_sync_batchnorm(self.network).cuda()
        print("------- Imaging System ---------")
        print(self.camera)
        print("------- Network ---------")
        count_parameters(self.network)
        
        # Optimizers
        self.network_optimiser = torch.optim.Adam(self.network.parameters(), lr=self.network_config['nn_lr'])
        self.network_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.network_optimiser, gamma=0.995) #MultiStepLR(self.network_optimiser, milestones=[50, 100, 150], gamma=0.1)

        # warp model by using DistributedDataParallel
        #self.camera = DDP(self.camera, device_ids=[local_rank])
        self.network = DDP(self.network, device_ids=[local_rank])
        
    def load_dataset(self, data_path):
        
        train_dataset = MiniImagenetDataset(data_path, train=True, img_res=self.network_config['image_size'])
        train_dataloader = DataLoader(train_dataset, batch_size=int(self.network_config['batch_size']), shuffle=False, pin_memory=True, sampler=DistributedSampler(train_dataset))
        
        valid_dataset = MiniImagenetDataset(data_path, train=False, img_res=self.network_config['image_size'])
        valid_dataloader = DataLoader(valid_dataset, batch_size=int(self.network_config['batch_size']), shuffle=False, pin_memory=True, sampler=DistributedSampler(valid_dataset), drop_last=True)

        return train_dataloader, valid_dataloader
    
    def create_experiments(self):
        # *** establish folders for saving model parameters ***
        date = datetime.now()
        self.model_dir = os.path.join(self.model_dir, "%s_total_epoch_%s_batch_size_%s_lr_%s" % (date.strftime("%Y%m%d-%H%M%S"), 
                                                                                   str(self.network_config['epoch']),
                                                                                   str(self.network_config['batch_size']),
                                                                                   str(self.network_config['nn_lr'])))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
    
        # *** establish folders for saving experiment results ***
        logs_dir =  os.path.join(self.model_dir, "%s_total_epoch_%s_batch_size_%s_lr_%s" % (date.strftime("%Y%m%d-%H%M%S"), 
                                                                                   str(self.network_config['epoch']),
                                                                                   str(self.network_config['batch_size']),
                                                                                   str(self.network_config['nn_lr'])))
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        
        # *** saving configuration files into the experoment folder to record training details ***
        config_path = os.path.join(logs_dir, 'config.txt')
        # Write the YAML content to the config file
        if self.verbose:
            with open(config_path, 'w') as file:
                yaml.dump(self.config_file, file)
    
        # run directory for tensorboard information
        self.writer = SummaryWriter(logs_dir)

    def _run_batch(self, inputs, validate=False):
        
        batch_size = int(self.network_config['batch_size'])

        #inputs = normalize_ImageNet(inputs)

        measurements = self.camera(inputs, batchsize=batch_size).float()
        x_output = self.network(measurements)
        
        # Compute and print loss
        loss = self.L1(inputs, x_output) #+ 0.7 * self.SSIMLoss(x_output, inputs) #+ 0.5 * self.PSNRLoss(x_output, inputs)#+ 0.5 * self.PerceptualLoss(x_output, inputs) #+ 0.1 * self.SSIMLoss(x_output, inputs)
        
        if validate == False:
            self.network_optimiser.zero_grad()
            loss.backward()
            self.network_optimiser.step()
        
        #x_output = denormalize_ImageNet(x_output)  # de-normalize the imgs to [0, 1] for visulization

        return x_output, loss
    
    def train(self):
        
        self.create_experiments()
        train_dataloader, valid_dataloader = self.load_dataset(self.data_path)
        print("Start training.")
        
        best_metric = 10
        iters = 0
        total_epochs = int(self.network_config['epoch'])
        
        for epoch in range(1, total_epochs+1):
        #  =====================Train and validate============================
            start = time.time()
            train_dataloader.sampler.set_epoch(epoch)
            with tqdm(train_dataloader, unit='batch') as tepoch:
                for itr, data in enumerate(tepoch, 0):
                    
                    inputs = data
                    inputs = inputs.float().cuda()  # normalize the imgs when using ImageNet pretraining model
                    preds, loss = self._run_batch(inputs, validate=False)
                    
                    if iters % 200 == 0 and self.verbose:
                        self.writer.add_scalar('Loss', loss.item(), iters)
                        
                    if iters % 200 == 0 and self.verbose: 
                        output_input_gt = torch.cat((preds, inputs), dim=0)
                        grid = torchvision.utils.make_grid(output_input_gt,
                                                           scale_each=True,
                                                           nrow=int(self.network_config['batch_size']),
                                                           normalize=True).cpu().detach().numpy()
                        self.writer.add_image("Output_vs_gt | Training", grid, iters)
                    
                    if self.verbose:
                        tepoch.set_postfix(epoch=epoch, loss=loss.item())
                
                    iters += 1
        
            self.network_scheduler.step()

            # =====================valid============================
            if epoch % 2 == 0:
                self.camera.eval()
                self.network.eval()
                validate_epoch_loss = []
                validate_epoch_psnr = []
                validate_epoch_ssim = []
            
                with tqdm(valid_dataloader, unit="batch") as tepoch:
                    valid_dataloader.sampler.set_epoch(epoch)
                    for data in tepoch:
                        valid_inputs = data
                        valid_inputs = valid_inputs.cuda()
                        
                        with torch.no_grad():
                            preds, loss = self._run_batch(valid_inputs, validate=True)
                        
                        validate_epoch_psnr.append(batch_PSNR(valid_inputs, preds))
                        validate_epoch_ssim.append(batch_SSIM(valid_inputs, preds))
                        validate_epoch_loss.append(loss.item())
                        if self.verbose:
                            tepoch.set_postfix(epoch=epoch, loss=loss.item())
                
                validate_loss = np.stack(validate_epoch_loss).mean()
                self.writer.add_scalars('loss per epoch ', {'validate': validate_loss}, epoch)
                validate_psnr = np.stack(validate_epoch_psnr).mean()
                self.writer.add_scalars('PSNR per epoch ', {'validate': validate_psnr}, epoch)
                validate_ssim = np.stack(validate_epoch_ssim).mean()
                self.writer.add_scalars('SSIM per epoch ', {'validate': validate_ssim}, epoch)
                
                if self.verbose:
                    output_input_gt = torch.cat((preds, valid_inputs), dim=0)
                    grid = torchvision.utils.make_grid(output_input_gt,
                                                    scale_each=True,
                                                    nrow=int(self.network_config['batch_size']),
                                                    normalize=True).cpu().detach().numpy()
                    self.writer.add_image("Output_vs_gt | Validation", grid, epoch)

                if self.verbose and validate_loss < best_metric:
                    best_metric = validate_loss
                    best_epoch = epoch
                    torch.save({"network_state_dict": self.network.module.state_dict(),
                                "network_optim_state_dict": self.network_optimiser.state_dict(),
                                "epoch": best_epoch
                                }, os.path.join(self.model_dir, 'model_best.pth'))
        
        if self.verbose:
            torch.save({"network_state_dict": self.network.module.state_dict(),
                        "network_optim_state_dict": self.network_optimiser.state_dict()
                        }, os.path.join(self.model_dir, 'model_last.pth'))


    def text_model(self):
        NotImplemented

        
def main(loaded_config, data_path, model_dir, logs_dir):

    trainer = DDPTrainer(config_file=loaded_config, 
                         data_path=data_path,
                         model_dir=model_dir, 
                         logs_dir=logs_dir)
    
    trainer.train()


if __name__ == '__main__':
    
    
    torch.manual_seed(3407)
    torch.random.seed()
    path = os.path.dirname(
                    os.path.abspath(__file__),
    )
    print(path)
    
    # Load .yaml vars
    config_path = os.path.join(path, 'parameters.yaml')
    with open(config_path, 'r') as file:
        loaded_config = yaml.safe_load(file)
        print(loaded_config)
    # load dataset root
    data_path = os.path.join(path, 'mini-imagenet')
    model_dir = os.path.join(path, 'checkpoint')
    logs_dir = os.path.join(path, 'logs')

    torch.cuda.empty_cache()
    # The total GPU We have
    world_size = torch.cuda.device_count()
    print("Total utilized GPUs {}".format(world_size))
    
    main(loaded_config, data_path, model_dir, logs_dir)