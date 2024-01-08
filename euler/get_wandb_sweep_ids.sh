cd ..

echo "emd:"
wandb sweep src/conf/experiments/optimization/sweep_emd.yaml

echo "sinkhorn gw:"
wandb sweep src/conf/experiments/optimization/sweep_sinkhorn_gw.yaml

echo "sinkhorn no gw:"
wandb sweep src/conf/experiments/optimization/sweep_sinkhorn_no_gw.yaml

