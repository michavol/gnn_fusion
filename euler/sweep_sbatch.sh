#!/bin/bash

#SBATCH --time=00:10:00
##SBATCH --gpus=1
##SBATCH --gres=gpumem:10g
#SBATCH --mem-per-cpu=20G
#SBATCH --job-name=dslab_training
#SBATCH --output=log/%x.out                                                                         
#SBATCH --error=log/%x.err

# export JAX_PLATFORMS="cuda"
export JAX_PLATFORMS=""
module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

cd ..

# Activate virtual environment
source dl_euler/bin/activate

# Pass sweep id as first arguemtn - get id by running sh get_agent_id.sh
wandb agent $1

cd euler
