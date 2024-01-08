# Euler Setup for Sweeps

For running the hyperparameter sweeps we used Euler. The reason is that we had issues with installing necessary
libraries on Student Cluster.

You may want to consider using the new JupyterHub (https://scicomp.ethz.ch/wiki/JupyterHub). I can recommend running
JupyterLab (there are multiple options).

Go to: https://jupyter.euler.hpc.ethz.ch to start a session

## 0. Clone Repository

Git clone the repository into the euler directory.

## 1. Euler Modules

Execute in the terminal (euler home directory):

```shell
env2lmod
module purge
module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy
``` 

## 2. Virtual Environment

Execute in the terminal (gnn_fusion root directory):

```shell
python -m venv --system-site-packages dl_euler
source dl_euler/bin/activate
pip install --upgrade pip
pip install -r euler/requirements.txt

```

## 3. Install DGL

```shell
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

[//]: # (## 4. Ensure Compatibility)

[//]: # (To ensure compatibility of benchmarking code with newer dgl version uncomment the marked line)

[//]: # (in: ```gnn_benchmarking/layers/gcn_layer.py```)

## 4. Download Compatible ZINC Dataset

Execute in terminal (from ```gnn_fusion/euler```directory):

```shell
sh download_molecules_dgl_latest.sh
```

## 5. Setup Wandb

Go to http://wandb.ai/, setup a free account (remember you team/username), get API key and execute

```shell
echo "export WANDB_API_KEY=\"YOUR_API_KEY\"" >> ~/.bashrc
source ~/.bashrc
wandb login
```

In all configs under ```gnn_fusion/src/conf/experiments/optimization``` replace entity `deep-talk` with your
team/username.

## 6. Prepare Directory Structure

Create the following folders for the sweep to store the results:

```shell
mkdir src/conf/models/optimization_models
mkdir models/optimization_models
mkdir src/conf/models/samplesize_models
mkdir models/samplesize_models
```

## 7. Run Sweep

Execute in terminal (from ```gnn_fusion/euler```directory):

```
sh get_wandb_sweep_ids.sh
```

Copy the SWEEP_ID from the output of the previous statement and run:

```
sbatch sweep_sbatch.sh SWEEP_ID
```

You can run multiple jobs at once. Make sure that there is at least one job per SWEEP_ID returned by the previous
command.
