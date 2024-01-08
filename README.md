# Ot-Fusion of GCNs

## Repository Structure

## Running the code

## Reproducing Sweeps

Results of sweeps are presented in sections 5.1 and 5.2 of the report.

### 1. Setup Sweeps on Euler
Follow the instructions from ```gnn_fusion/euler/setup_instructions.py```.

## 2. Evaluate Fused Models
After all the sweeps are finished, execute (from ```gnn_fusion```directory):

```
python src/optimization_experiment.py
```

The above command should generate a csv file with the results in the ```gnn_fusion/results```

## Reproducing Only Report Results Locally

To reproduce the results from the report locally, follow the steps below. All commands should be executed from
gnn_fusion root directory.

### 1. Install Dependencies

```shell
python -m venv --system-site-packages .venv
source .venv/bin/activate
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


