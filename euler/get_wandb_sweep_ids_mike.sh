cd ..
# echo "emd fast lp:"
# wandb sweep src/conf/experiments/optimization/sweep_emd_fast_lp.yaml

# echo "emd:"
# wandb sweep src/conf/experiments/optimization/sweep_emd.yaml

echo "sinkhorn fast lp:"
wandb sweep src/conf/experiments/optimization/sweep_sinkhorn_fast_lp.yaml

# echo "sinkhorn gw:"
# wandb sweep src/conf/experiments/optimization/sweep_sinkhorn_gw.yaml

echo "sinkhorn no gw:"
wandb sweep src/conf/experiments/optimization/sweep_sinkhorn_no_gw.yaml

