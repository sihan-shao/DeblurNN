#!/bin/bash
#SBATCH --account=project_2010479
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:v100:4

module purge
module load pytorch
which torchrun

#srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 joint_design_trainer.py
srun torchrun --standalone --nnodes=1 --nproc_per_node=4 train_restormer.py