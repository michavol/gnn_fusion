# Ot-Fusion of GCNs

## Repository Structure

Here we describe the contents of the most important folders in this repository.

* `src` - Source code that performs the OT-Fusion. Parts of the code were inspired by the
  original [Model Fusion via Optimal Transport](https://github.com/sidak/otfusion).
* `gnn_benchmarking` - Slightly modified code
  of [Benchmarking Graph Neural Networks](https://github.com/graphdeeplearning/benchmarking-gnns) used for defining,
  training and evaluating graph neural networks.
* `results_reproduction` - Scripts for reproducing the results from the project report.
* `euler` - Code and configs for setting up the parameter sweeps on Euler.
* `models` - Small pretrained models for experiments.

## Running the code

To run the code

## Reproducing Result

### Reproducing Sweep Results

Results of sweeps are presented in sections 5.1 and 5.2 of the report. We recommend running the sweeps on Euler, as the
number of configurations makes the process computationally heavy.

#### 1. Setup Sweeps on Euler

Follow the instructions from ```gnn_fusion/euler/setup_instructions.py```.

#### 2. Evaluate Fused Models

After all the sweeps are finished, uncomment the first two lines in ```results_reproduction/evaluate_sweeps.sh```, and
execute (from ```gnn_fusion```directory):

```
 source results_reproduction/evaluate_sweeps.sh 
```

The above command should generate csv files with results in the ```gnn_fusion/results```, print the contents of Table 1
in the terminal and save Figure 1 in ```gnn_fusion/report/figures``` folder.

### Reproducing Remaining Tables

Results from sections 5.3, 5.4 and 5.5 can be easily reproduced on ones laptop. If you want to setup the code on your
laptop, follow all the steps in this section. If you already configured Euler, you can continue in Euler terminal from
step 3.

All commands should be executed from gnn_fusion root directory.

#### 1. Install Dependencies

Execute in the terminal:

```shell
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Download ZINC Dataset

```shell
sh reproduce_results/download_molecules_dgl.sh
```

#### 3. Reproduce Results

To reproduce results from Table 2 and Table 3 execute:

```shell
sh reproduce_results/table_2_3.sh
```

To reproduce results from Table 4 execute:

```shell
sh reproduce_results/table_4.sh
```

To reproduce results from Table 5 execute:

```shell
sh reproduce_results/table_5.sh
```

#TODO: Running the code
#TODO: How to reproduce table results locally - test it
#TODO: Authors and aknowledgement


