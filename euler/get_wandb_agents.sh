cd ..
echo "EMD:"
wandb sweep src/conf/experiments/optimization/sweep_emd.yaml

echo "Sinkhorn:"
wandb sweep src/conf/experiments/optimization/sweep_sinkhorn.yaml
cd euler
