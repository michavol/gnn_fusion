#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=gnn_fusion_optimization
#SBATCH --ntasks=8
#SBATCH --output=log/%x.out                                                                         
#SBATCH --error=log/%x.err

#export JAX_PLATFORMS="cuda"
#export JAX_PLATFORMS=""

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

cd ..

# Activate virtual environment
source dl_euler/bin/activate

# Pass sweep id as first argument - get id by running sh get_agent_id.sh
wandb agent $1

