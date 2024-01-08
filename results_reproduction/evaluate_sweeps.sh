# Uncomment the lines below if you have run a new sweep and wish to overwrite previous results
#python src/optimization_experiment.py config_dir=../../../src/conf/models/optimization_models/ results_file=MAE_general_sweep.csv
#python src/optimization_experiment.py config_dir=../../../src/conf/models/samplesize_models/ results_file=MAE_emd_batchsize.csv

echo "Best configurations for Sinkhorn (Table 1)"
python report/create_tables.py results/MAE_general_sweep.csv --algo sinkhorn --sample_size_lp_q 100 --sample_size_gw 2

echo "Best configurations for EMD (Table 1)"
python report/create_tables.py results/MAE_general_sweep.csv --algo emd --sample_size_lp_q 100 --sample_size_gw 2

echo "Generating sample size plot (Figure 1)"
python report/lineplot.py