# Ot-Fusion of GCNs

## Repository Structure

## Running the code

## Reproducing Sweeps

## Reproducing Only Report Results Locally

To reproduce the results from the report locally, follow the steps below. All commands should be executed from
gnn_fusion root directory.

### 1. Install Dependencies

```shell
python -m venv --system-site-packages dl_euler
source dl_euler/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download ZINC Dataset

```shell
sh reproduce_results/download_molecules_dgl.sh
```

### 2. Reproduce Results

To reproduce results from Table 1:

```shell
sh reproduce_results/table_1.sh
```

To reproduce results from other tables, just substitute the table number in the above command.

#TODO: Running the code
#TODO: How to reproduce sweep results on Euler
#TODO: How to reproduce table results locally - test it
#TODO: Describe repository structure
#TODO: Authors and aknowledgement


