# EULER SETUP
You may want to consider using the new JupyterHub (https://scicomp.ethz.ch/wiki/JupyterHub)

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
source dl_euler/bin/activate #to activate
# pip list #to check packages
pip install --upgrade pip #install packages via pip
pip install -r requirements.txt
```
## 3. Install DGL
```shell
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## 4. Ensure Compatibility
To ensure compatibility of benchmarking code with newer dgl version uncomment the marked line
in: ```gnn_benchmarking/layers/gcn_layer.py```

## 5. Download Compatible ZINC Dataset
Execute from within the ```euler``` directory.
```shell
sh download_molecules_dgl_latest.sh
```
## 5. Setup Wandb
Go to http://wandb.ai/authorize to get API key and execute
```shell
echo "export WANDB_API_KEY=\"YOUR_API_KEY\"" >> ~/.bashrc
source ~/.bashrc
wandb login
```